"""Hard-mode scenarios where fixed rules break down.
Run:  python -m silicon_tech.hard          (rules only, no tokens)
      python -m silicon_tech.hard --live   (also calls Nemotron on Nebius)
"""
import sys
from silicon_tech.simulator import SimProcessor, Fault
from silicon_tech.diagnoser import diagnose

BASE = {"util": 30.0, "run_queue": 1.0, "ctx_switches": 2000.0, "irq_per_s": 800.0,
        "sched_latency_us": 40.0, "ipc": 1.8, "cache_miss_pct": 3.0}
START, LAST_TICK, WINDOW = 20, 40, 8


class HardProcessor(SimProcessor):
    """Adds fault severity (weakened signatures) and a stuck temperature sensor."""

    def __init__(self, name, cores, seed, severity=1.0, glitch_core=None):
        super().__init__(name, cores, seed)
        self.severity = severity
        self.glitch_core = glitch_core

    def step(self):
        sample, truth = super().step()
        if self.severity < 1.0:
            faulted = {c for t in truth if t["fault"] != "thermal_throttle" for c in t["cores"]}
            for row in sample["cores"]:
                if row["core"] in faulted:
                    for k, b in BASE.items():
                        row[k] = round(b + self.severity * (row[k] - b), 2)
                    for k in ("run_queue", "ctx_switches", "irq_per_s"):
                        row[k] = int(round(row[k]))
        if self.glitch_core is not None and sample["tick"] >= 38:
            sample["cores"][self.glitch_core]["temp_c"] = 0.0   # sensor reads 0C
        return sample, truth


NONE = "No maintenance scheduled on this node."
# (name, injected fault, cores, severity, glitch_core, correct answer, maintenance context)
CASES = [
    ("subtle-mem-a", Fault.MEMORY_STALL, [2], 0.45, None, {"memory_stall": [2]}, NONE),
    ("subtle-mem-b", Fault.MEMORY_STALL, [5, 6], 0.5, None, {"memory_stall": [5, 6]}, NONE),
    ("subtle-noisy-a", Fault.NOISY_NEIGHBOR, [1], 0.4, None, {"noisy_neighbor": [1]}, NONE),
    ("subtle-noisy-b", Fault.NOISY_NEIGHBOR, [3, 4], 0.4, None, {"noisy_neighbor": [3, 4]}, NONE),
    ("subtle-irq", Fault.IRQ_STORM, [7], 0.15, None, {"irq_storm": [7]}, NONE),
    ("batch-benign-a", Fault.RUNAWAY_PROCESS, [2, 3], 1.0, None, {},
     "Scheduled: nightly log-compression batch job pinned to cores 2 and 3 (100% load is expected)."),
    ("batch-benign-b", Fault.RUNAWAY_PROCESS, [6], 1.0, None, {},
     "Scheduled: nightly backup checksum job pinned to core 6 (100% load is expected)."),
    ("runaway-real", Fault.RUNAWAY_PROCESS, [4], 1.0, None, {"runaway_process": [4]},
     "No batch jobs are scheduled on this node."),
    ("healthy-glitch", None, [], 1.0, 5, {}, NONE),
    ("thermal-glitch", Fault.THERMAL_THROTTLE, [1], 1.0, 6, {"thermal_throttle": [1]}, NONE),
]


def run_case(i, spec):
    name, fault, cores, severity, glitch, truth, context = spec
    p = HardProcessor(name, 8, seed=1000 + i, severity=severity, glitch_core=glitch)
    if fault is not None:
        p.inject(fault, cores, start=START)
    samples = [p.step()[0] for _ in range(LAST_TICK + 1)]
    return samples[-WINDOW:], truth, context


def fmt(d):
    return str(d) if d else "none"


def main(live):
    agent = None
    if live:
        from silicon_tech import agent
    rules_ok = agent_ok = tok_in = tok_out = 0
    for i, spec in enumerate(CASES):
        window, truth, context = run_case(i, spec)
        rules = diagnose(window[-1])
        r_ok = rules == truth
        rules_ok += r_ok
        print(f"{spec[0]:15} truth={fmt(truth)}")
        print(f"   rules : {fmt(rules)}  {'OK' if r_ok else 'MISS'}")
        if live:
            got, parsed, usage = agent.diagnose_with_llm(window, context)
            a_ok = got == truth
            agent_ok += a_ok
            tok_in += usage.prompt_tokens
            tok_out += usage.completion_tokens
            print(f"   nemo  : {fmt(got)}  {'OK' if a_ok else 'MISS'}")
            if parsed:
                print(f"   note  : {str(parsed.get('summary', ''))[:110]}")
    print(f"\nrules    : {rules_ok}/{len(CASES)} correct")
    if live:
        print(f"nemotron : {agent_ok}/{len(CASES)} correct")
        print(f"tokens   : {tok_in} in, {tok_out} out")


if __name__ == "__main__":
    main("--live" in sys.argv)