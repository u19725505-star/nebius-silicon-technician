"""Nemotron technician agent: reads telemetry + maintenance context, returns a diagnosis."""
import json
from statistics import median
from silicon_tech import llm

SYSTEM = """You are a senior data center technician diagnosing one server processor with 8 cores.
You get telemetry from the latest tick and the median of the last 8 ticks, plus maintenance context.
Compare every core with its peers. A healthy core looks like: util ~30%, temp 55-65C, 3000 MHz,
run queue ~1, ~2000 ctx/s, ~800 irq/s, scheduler latency ~40us, IPC ~1.8, cache miss ~3%.
Fault signatures (they can be mild or partial):
- thermal_throttle: temperature high (above ~85C) with a reduced clock.
- noisy_neighbor: high util, deeper run queue, MANY context switches (far above peers), high latency; IPC and cache misses are only mildly worse (IPC ~1.0-1.6, miss ~5-15%).
- runaway_process: one core pinned near 100% util with FEW context switches and moderate latency.
- irq_storm: interrupt rate far above normal, latency jitter.
- memory_stall: busy core with VERY low IPC (below ~1.0) and a high cache-miss rate (above ~15%) but NORMAL context switches and run queue.
Rules: ignore physically implausible sensor glitches (for example a 0C temperature reading) when the
window medians look normal. FIRST read the maintenance context: if a scheduled job is named for a core and that core shows the load that job would cause (for example one pinned core at ~100% util), that core is NOT faulty, so leave it out of faults entirely.
Report ONE fault per core (the single best explanation); never list a core under two faults.
Reply with ONLY a JSON object:
{"faults": {"fault_name": [core numbers]}, "summary": "one sentence", "action": "one sentence"}
Use {} for faults if the processor is healthy."""


def _line(c):
    return (f"core {c['core']}: util {c['util']:.0f} temp {c['temp_c']:.0f} MHz {c['freq_mhz']:.0f} "
            f"rq {c['run_queue']} ctx {c['ctx_switches']:.0f} irq {c['irq_per_s']:.0f} "
            f"lat {c['sched_latency_us']:.0f}us ipc {c['ipc']:.2f} miss {c['cache_miss_pct']:.0f}%")


def build_prompt(window, context):
    latest = window[-1]["cores"]
    meds = []
    for i in range(len(latest)):
        row = {"core": i}
        for k in latest[i]:
            if k != "core":
                row[k] = median(s["cores"][i][k] for s in window)
        meds.append(row)
    return (f"Maintenance context: {context}\n\n"
            f"LATEST TICK:\n" + "\n".join(_line(c) for c in latest) +
            f"\n\nMEDIAN OF LAST {len(window)} TICKS:\n" + "\n".join(_line(c) for c in meds))


def parse(text):
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def diagnose_with_llm(window, context, model=None):
    """Returns (faults dict or None, parsed json or None, usage)."""
    text, usage = llm.ask(SYSTEM, build_prompt(window, context), model=model,
                          max_tokens=2500, temperature=0.0)
    parsed = parse(text)
    if parsed is None:
        return None, None, usage
    faults = {}
    for name, cores in (parsed.get("faults") or {}).items():
        try:
            cs = sorted(int(x) for x in cores)
        except (TypeError, ValueError):
            continue
        if cs:
            faults[str(name).lower()] = cs
    return faults, parsed, usage

def diagnose_with_llm(window, context, model=None):
    """Retry once with a larger token budget if the model returns no usable JSON."""
    from types import SimpleNamespace
    prompt = build_prompt(window, context)
    tin = tout = 0
    parsed = None
    for max_tokens in (2500, 6000):
        text, usage = llm.ask(SYSTEM, prompt, model=model,
                              max_tokens=max_tokens, temperature=0.0)
        tin += usage.prompt_tokens
        tout += usage.completion_tokens
        parsed = parse(text)
        if parsed is not None:
            break
    total = SimpleNamespace(prompt_tokens=tin, completion_tokens=tout)
    if parsed is None:
        return None, None, total
    faults = {}
    for name, cores in (parsed.get("faults") or {}).items():
        try:
            cs = sorted(int(x) for x in cores)
        except (TypeError, ValueError):
            continue
        if cs:
            faults[str(name).lower()] = cs
    return faults, parsed, total
