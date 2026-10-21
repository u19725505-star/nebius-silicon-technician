"""Peer-comparison anomaly detector + rule-based diagnoser (zero LLM tokens)."""
from statistics import median

# Typical healthy spread per metric; a core is flagged when it sits
# Z_FLAG "spreads" away from the median of its peers on the same chip.
SCALES = {
    "util": 7.0,
    "temp_c": 3.0,
    "freq_mhz": 30.0,
    "run_queue": 0.8,
    "ctx_switches": 300.0,
    "irq_per_s": 120.0,
    "sched_latency_us": 8.0,
    "ipc": 0.1,
    "cache_miss_pct": 0.8,
}
Z_FLAG = 6.0


def core_findings(sample):
    """Return cores that deviate strongly from their peers."""
    cores = sample["cores"]
    med = {m: median(c[m] for c in cores) for m in SCALES}
    out = []
    for c in cores:
        z = {m: (c[m] - med[m]) / SCALES[m] for m in SCALES}
        flagged = {m: round(v, 1) for m, v in z.items() if abs(v) >= Z_FLAG}
        if flagged:
            out.append({"core": c["core"], "z": flagged, "raw": c, "peer_median": med})
    return out


def classify(f):
    """Map one flagged core's signature to a fault label."""
    r, z = f["raw"], f["z"]
    if r["temp_c"] > 85:
        return "thermal_throttle"
    if r["irq_per_s"] > 20000:
        return "irq_storm"
    if r["ipc"] < 0.8 and r["cache_miss_pct"] > 20:
        return "memory_stall"
    if r["run_queue"] >= 5 and r["ctx_switches"] > 8000:
        return "noisy_neighbor"
    if r["util"] >= 98 and r["ctx_switches"] < 1000:
        return "runaway_process"
    if z.get("temp_c", 0) >= Z_FLAG:
        return "thermal_throttle"   # early heat ramp, before throttling
    return "unknown"


def diagnose(sample):
    """Return {fault_label: [core, ...]} for one tick of one processor."""
    found = {}
    for f in core_findings(sample):
        found.setdefault(classify(f), []).append(f["core"])
    return {k: sorted(v) for k, v in found.items()}