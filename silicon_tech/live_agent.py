"""Nemotron technician for a LIVE host: baseline vs now, with a zero-token gate."""
from statistics import median
from silicon_tech import llm
from silicon_tech.agent import parse

SYSTEM_LIVE = """You are a senior data center technician diagnosing a live Linux server.
You get a BASELINE (medians from a healthy period) and the CURRENT window medians.
Per core: util % and wake latency (p90 microseconds that a 1 ms sleep wakes late when pinned to that core).
System-wide: load average, context switches/s, interrupts/s. Missing counters show as n/a (no hardware counters here).
Compare every core with its own baseline and with its peers.
Fault signatures:
- runaway_process: one core near 100% util while peers stay near baseline; its wake latency is far above baseline.
- noisy_neighbor: several cores busy at once, load average well above baseline, many more context switches.
- irq_storm: interrupts/s far above baseline.
- healthy: everything within normal variation of the baseline.
Reply with ONLY a JSON object:
{"status": "healthy" or "degraded", "faults": {"fault_name": [core numbers]}, "summary": "one sentence", "action": "one sentence"}
Use {} for faults when healthy. One fault per core."""


def summarize(window):
    cores = [c["core"] for c in window[0]["cores"]]
    out = {
        "load1": median(s["load1"] for s in window),
        "ctx_per_s": median(s["ctx_per_s"] for s in window),
        "irq_per_s": median(s["irq_per_s"] for s in window),
        "cores": {},
    }
    for i, c in enumerate(cores):
        def med(k):
            vals = [s["cores"][i][k] for s in window if s["cores"][i][k] is not None]
            return round(median(vals), 1) if vals else None
        out["cores"][c] = {"util": med("util"), "wake_us": med("wake_latency_us"),
                           "freq": med("freq_mhz"), "temp": med("temp_c")}
    return out


def _fmt(v, unit=""):
    return "n/a" if v is None else f"{v}{unit}"


def describe(name, s):
    lines = [f"{name}: load {s['load1']}  ctx/s {s['ctx_per_s']:.0f}  irq/s {s['irq_per_s']:.0f}"]
    for c, d in s["cores"].items():
        lines.append(f"  core {c}: util {_fmt(d['util'], '%')}  wake_latency {_fmt(d['wake_us'], 'us')}  "
                     f"freq {_fmt(d['freq'], 'MHz')}  temp {_fmt(d['temp'], 'C')}")
    return "\n".join(lines)


def gate(base, now):
    """Cheap check, zero tokens. True means the host changed enough to ask the model."""
    for c, d in now["cores"].items():
        b = base["cores"][c]
        if d["util"] is not None and b["util"] is not None and d["util"] - b["util"] >= 30:
            return True
        if d["wake_us"] is not None and b["wake_us"] is not None \
                and d["wake_us"] > 3 * max(b["wake_us"], 50):
            return True
    return now["irq_per_s"] > 5 * max(base["irq_per_s"], 500)


def diagnose_live(baseline_window, window, model=None):
    """Returns (result dict or None, tokens_in, tokens_out)."""
    base, now = summarize(baseline_window), summarize(window)
    if not gate(base, now):
        return ({"gated": True, "status": "healthy", "faults": {}, "action": "none",
                 "summary": "No change from baseline; model not called."}, 0, 0)
    prompt = describe("BASELINE (healthy)", base) + "\n\n" + describe("CURRENT", now)
    tin = tout = 0
    parsed = None
    for max_tokens in (2500, 6000):
        text, usage = llm.ask(SYSTEM_LIVE, prompt, model=model,
                              max_tokens=max_tokens, temperature=0.0)
        tin += usage.prompt_tokens
        tout += usage.completion_tokens
        parsed = parse(text)
        if parsed is not None:
            break
    return parsed, tin, tout