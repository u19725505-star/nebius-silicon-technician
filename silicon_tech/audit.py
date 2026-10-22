"""Judge-readiness audit.
Run:  python -m silicon_tech.audit            (no tokens)
      python -m silicon_tech.audit --live     (adds held-out Nemotron + fallback test)
Options: --n 20 (held-out cases)  --seed 777
"""
import os
import re
import sys
import time
import random
import multiprocessing as mp
from statistics import median
from silicon_tech.simulator import Fault
from silicon_tech.diagnoser import diagnose
from silicon_tech.hard import HardProcessor, START, LAST_TICK, WINDOW, NONE
from silicon_tech import scoreboard

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []


def record(status, text):
    RESULTS.append((status, text))
    print(f"[{status}] {text}", flush=True)


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# ---------- 1. submission basics ----------
def static_checks():
    for f in ("LICENSE", "README.md", "requirements.txt", "results_hard_v2.txt"):
        record("PASS" if os.path.exists(os.path.join(REPO, f)) else "FAIL", f"file present: {f}")
    readme = ""
    p = os.path.join(REPO, "README.md")
    if os.path.exists(p):
        readme = open(p, errors="ignore").read()
    record("PASS" if len(readme) >= 1500 else "FAIL",
           f"README length {len(readme)} chars (judges read it first; aim for 1500+)")
    for kw in ("Nemotron", "Nebius", "limitation", "feedback"):
        record("PASS" if kw.lower() in readme.lower() else "FAIL", f"README mentions '{kw}'")
    pat = re.compile(r"""(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['"][A-Za-z0-9_\-\.]{20,}['"]""")
    leaks = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".venv", ".git", "__pycache__")]
        for f in files:
            if f.endswith((".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".env")):
                try:
                    text = open(os.path.join(root, f), errors="ignore").read()
                except OSError:
                    continue
                if pat.search(text):
                    leaks.append(f)
    record("FAIL" if leaks else "PASS", f"no hard-coded secrets found" if not leaks else f"possible secret in: {leaks}")


# ---------- 2. clean simulator ----------
def rules_clean():
    rows = scoreboard.run(seed=42, nodes=12, verbose=False)
    faulty = [r for r in rows if r[1] != "healthy"]
    hits = [r for r in faulty if r[2] is not None]
    record("PASS" if len(hits) == len(faulty) else "FAIL",
           f"clean simulator: rules diagnosed {len(hits)}/{len(faulty)} faulty nodes")


# ---------- 3. real processor, zero tokens ----------
def live_host_check():
    from silicon_tech.collector import LiveHost, read_schedstat
    from silicon_tech.live_agent import summarize, gate
    from silicon_tech.live_demo import hog, collect
    record("PASS" if read_schedstat() else "FAIL", "/proc/schedstat readable on this host")
    host = LiveHost()
    host.step()
    base = collect(host, 6)
    quiet = collect(host, 4)
    sb, sq = summarize(base), summarize(quiet)
    record("PASS" if not gate(sb, sq) else "FAIL", "gate stays closed on an idle host (0 tokens)")
    core = host.cores[0]
    proc = mp.get_context("fork").Process(target=hog, args=(core,), daemon=True)
    proc.start()
    try:
        time.sleep(3)
        busy = collect(host, 4)
    finally:
        proc.terminate()
        proc.join(timeout=2)
    sh = summarize(busy)
    record("PASS" if gate(sb, sh) else "FAIL", f"gate opens under a real CPU hog on core {core}")
    b, h = sb["cores"][core]["wake_us"], sh["cores"][core]["wake_us"]
    visible = b is not None and h is not None and h > 2 * max(b, 5)
    record("PASS" if visible else "WARN",
           f"real scheduler delay on core {core}: idle {b} us -> hog {h} us"
           + ("" if visible else "  (no clear latency signal; do not claim one)"))
    u = sh["cores"][core]["util"]
    record("PASS" if u is not None and u > 90 else "FAIL", f"hogged core utilization {u}%")


# ---------- 4. held-out Nemotron test ----------
def _cores_text(cs):
    if len(cs) == 1:
        return f"core {cs[0]}"
    return "cores " + ", ".join(str(c) for c in cs[:-1]) + f" and {cs[-1]}"


def make_cases(n, seed):
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        r = rng.random()
        name = f"case{i:02d}"
        if r < 0.55:
            label, fault, (lo, hi) = rng.choice([
                ("memory_stall", Fault.MEMORY_STALL, (0.4, 0.7)),
                ("noisy_neighbor", Fault.NOISY_NEIGHBOR, (0.35, 0.6)),
                ("irq_storm", Fault.IRQ_STORM, (0.1, 0.4)),
                ("runaway_process", Fault.RUNAWAY_PROCESS, (1.0, 1.0)),
            ])
            k = 1 if label == "runaway_process" else rng.randint(1, 2)
            cores = sorted(rng.sample(range(8), k))
            sev = round(rng.uniform(lo, hi), 2)
            ctx = "No batch jobs are scheduled on this node." if label == "runaway_process" else NONE
            cases.append((f"{name}-{label}", fault, cores, sev, None, {label: cores}, ctx))
        elif r < 0.75:
            cores = sorted(rng.sample(range(8), rng.randint(1, 2)))
            job = rng.choice(["log-compression", "backup checksum", "video transcode"])
            ctx = f"Scheduled: nightly {job} job pinned to {_cores_text(cores)} (100% load is expected)."
            cases.append((f"{name}-benign", Fault.RUNAWAY_PROCESS, cores, 1.0, None, {}, ctx))
        elif r < 0.88:
            cases.append((f"{name}-healthy-glitch", None, [], 1.0, rng.randrange(8), {}, NONE))
        else:
            cores = [rng.randrange(8)]
            g = rng.choice([c for c in range(8) if c not in cores])
            cases.append((f"{name}-thermal-glitch", Fault.THERMAL_THROTTLE, cores, 1.0, g,
                          {"thermal_throttle": cores}, NONE))
    return cases


def run_spec(spec, seed):
    name, fault, cores, sev, glitch, truth, ctx = spec
    p = HardProcessor(name, 8, seed=seed, severity=sev, glitch_core=glitch)
    if fault is not None:
        p.inject(fault, cores, start=START)
    samples = [p.step()[0] for _ in range(LAST_TICK + 1)]
    return samples[-WINDOW:], truth, ctx


def diagnose_safe(window, context, model=None):
    """Nemotron first; if it errors or returns nothing usable, fall back to the rules."""
    from silicon_tech import agent
    try:
        faults, parsed, usage = agent.diagnose_with_llm(window, context, model=model)
        if faults is not None:
            return faults, "nemotron", usage.prompt_tokens, usage.completion_tokens
        return diagnose(window[-1]), "rules-fallback (empty reply)", 0, 0
    except Exception as e:
        return diagnose(window[-1]), f"rules-fallback ({type(e).__name__})", 0, 0


def held_out(n, seed, live):
    cases = make_cases(n, seed)
    rules_ok = nemo_ok = tin = tout = fallbacks = 0
    times, misses = [], []
    for i, spec in enumerate(cases):
        window, truth, ctx = run_spec(spec, seed * 100 + i)
        rules_ok += diagnose(window[-1]) == truth
        if live:
            t0 = time.time()
            got, src, a, b = diagnose_safe(window, ctx)
            dt = time.time() - t0
            times.append(dt)
            tin += a
            tout += b
            fallbacks += src != "nemotron"
            ok = got == truth
            nemo_ok += ok
            if not ok:
                misses.append((spec[0], truth, got, src))
            print(f"   {spec[0]:30} {'OK  ' if ok else 'MISS'} {dt:5.1f}s", flush=True)
    record("INFO", f"held-out set: {n} unseen random cases (seed {seed}); rules got {rules_ok}/{n}")
    if live:
        acc = nemo_ok / n
        record("PASS" if acc >= 0.8 and nemo_ok > rules_ok else "FAIL",
               f"Nemotron held-out accuracy {nemo_ok}/{n} = {acc:.0%} (target 80%+ and above rules)")
        record("INFO", f"time per diagnosis: median {median(times):.1f}s, max {max(times):.1f}s")
        record("INFO", f"tokens: {tin} in, {tout} out; fallbacks used: {fallbacks}")
        for m in misses:
            print(f"   MISS {m[0]}: truth={m[1] or 'none'} got={m[2] or 'none'} via {m[3]}", flush=True)


# ---------- 5. failure test ----------
def fallback_check():
    window, truth, ctx = run_spec(make_cases(1, 5)[0], 5)
    faults, src, _, _ = diagnose_safe(window, ctx, model="nvidia/does-not-exist")
    record("PASS" if src.startswith("rules-fallback") else "FAIL",
           f"bad model name degrades gracefully to rules: {src}")


def main():
    live = "--live" in sys.argv
    n, seed = int(arg("--n", 20)), int(arg("--seed", 777))
    if live and not (os.environ.get("NEBIUS_API_KEY") and os.environ.get("NEBIUS_MODEL")):
        sys.exit("Set NEBIUS_API_KEY and NEBIUS_MODEL first.")
    print("== submission basics"); static_checks()
    print("== clean simulator"); rules_clean()
    print("== real processor (no tokens)")
    try:
        live_host_check()
    except Exception as e:
        record("FAIL", f"live host check crashed: {type(e).__name__}: {e}")
    print("== held-out test"); held_out(n, seed, live)
    if live:
        print("== failure test"); fallback_check()
    c = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "WARN", "INFO")}
    print(f"\nSUMMARY: {c['PASS']} pass, {c['FAIL']} fail, {c['WARN']} warn, {c['INFO']} info")
    print("READY" if c["FAIL"] == 0 else "NOT READY: fix the FAIL lines above")


if __name__ == "__main__":
    main()