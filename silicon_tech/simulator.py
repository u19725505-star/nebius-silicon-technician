"""Synthetic processor with injectable faults.

Each tick returns per-core telemetry (what the agent sees) plus ground truth
(what the scoreboard uses). Standard library only.
"""
import random
from dataclasses import dataclass, asdict
from enum import Enum


class Fault(str, Enum):
    NONE = "healthy"
    THERMAL_THROTTLE = "thermal_throttle"
    NOISY_NEIGHBOR = "noisy_neighbor"
    RUNAWAY_PROCESS = "runaway_process"
    IRQ_STORM = "irq_storm"
    MEMORY_STALL = "memory_stall"


BASE_FREQ_MHZ = 3000.0
AMBIENT_C = 35.0
THROTTLE_START_C = 88.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class CoreSample:
    core: int
    util: float            # percent busy
    freq_mhz: float
    temp_c: float
    run_queue: int         # runnable tasks waiting
    ctx_switches: int      # per second
    irq_per_s: int
    sched_latency_us: float
    ipc: float             # instructions per cycle
    cache_miss_pct: float


class SimProcessor:
    def __init__(self, name="cpu0", cores=8, seed=0):
        self.name = name
        self.n = cores
        self.rng = random.Random(seed)
        self.tick = 0
        self.temp = [50.0] * cores
        self.faults = []

    def inject(self, fault, cores, start, duration=None):
        self.faults.append({
            "fault": Fault(fault),
            "cores": list(cores),
            "start": start,
            "end": None if duration is None else start + duration,
        })

    def _active(self, core):
        for f in self.faults:
            if (f["start"] <= self.tick
                    and (f["end"] is None or self.tick < f["end"])
                    and core in f["cores"]):
                return f
        return None

    def _truth(self):
        out = []
        for f in self.faults:
            if f["start"] <= self.tick and (f["end"] is None or self.tick < f["end"]):
                out.append({"fault": f["fault"].value, "cores": f["cores"]})
        return out

    def step(self):
        """Advance one tick. Returns (sample, truth)."""
        r = self.rng
        rows = []
        for c in range(self.n):
            util = _clamp(r.gauss(30, 7), 5, 70)
            rq = max(0, round(r.gauss(1.2, 0.6)))
            ctx = int(r.gauss(2000, 250))
            irq = int(r.gauss(800, 100))
            lat = max(10.0, r.gauss(40, 6))
            ipc = _clamp(r.gauss(1.8, 0.08), 0.1, 4.0)
            miss = _clamp(r.gauss(3, 0.6), 0.1, 100.0)
            temp_boost = 0.0

            f = self._active(c)
            fault = f["fault"] if f else Fault.NONE

            if fault == Fault.THERMAL_THROTTLE:
                temp_boost = 45.0  # cooling failure
            elif fault == Fault.NOISY_NEIGHBOR:
                util = _clamp(r.gauss(90, 4), 0, 100)
                rq = max(0, round(r.gauss(8, 1.5)))
                ctx = int(ctx * 7)
                lat = max(50.0, r.gauss(420, 60))
                ipc = _clamp(r.gauss(1.1, 0.1), 0.1, 4.0)
                miss = _clamp(r.gauss(14, 2), 0.1, 100.0)
            elif fault == Fault.RUNAWAY_PROCESS:
                util = _clamp(r.gauss(99.8, 0.2), 0, 100)
                rq = max(0, round(r.gauss(3.5, 0.5)))
                ctx = int(r.gauss(400, 60))
                lat = max(50.0, r.gauss(160, 25))
            elif fault == Fault.IRQ_STORM:
                irq = int(r.gauss(110000, 15000))
                util = _clamp(r.gauss(62, 5), 0, 100)
                ctx = int(ctx * 3)
                lat = abs(r.gauss(700, 200)) + 100
                ipc = _clamp(r.gauss(1.3, 0.1), 0.1, 4.0)
            elif fault == Fault.MEMORY_STALL:
                util = _clamp(r.gauss(88, 4), 0, 100)
                ipc = _clamp(r.gauss(0.45, 0.05), 0.1, 4.0)
                miss = _clamp(r.gauss(36, 3), 0.1, 100.0)
                lat = max(50.0, r.gauss(110, 15))

            # Temperature follows load; throttling is emergent, not scripted.
            target = AMBIENT_C + 12 + 0.35 * util + temp_boost
            self.temp[c] += 0.15 * (target - self.temp[c]) + r.gauss(0, 0.3)
            temp = self.temp[c]

            freq = BASE_FREQ_MHZ + r.gauss(0, 15)
            if temp > THROTTLE_START_C:
                freq = max(800.0, BASE_FREQ_MHZ - (temp - THROTTLE_START_C) * 100)
            lat *= BASE_FREQ_MHZ / freq  # slower clock, worse latency

            rows.append(asdict(CoreSample(
                core=c,
                util=round(util, 1),
                freq_mhz=round(freq),
                temp_c=round(temp, 1),
                run_queue=int(rq),
                ctx_switches=ctx,
                irq_per_s=irq,
                sched_latency_us=round(lat, 1),
                ipc=round(ipc, 2),
                cache_miss_pct=round(miss, 1),
            )))

        sample = {"tick": self.tick, "processor": self.name, "cores": rows}
        truth = self._truth()
        self.tick += 1
        return sample, truth


def make_fleet(n=6, cores=8, seed=42):
    """Build n processors; roughly 70% get one injected fault."""
    rng = random.Random(seed)
    kinds = [f for f in Fault if f != Fault.NONE]
    fleet = []
    for i in range(n):
        p = SimProcessor(f"node{i:02d}", cores, seed=seed * 100 + i)
        if rng.random() < 0.7:
            kind = rng.choice(kinds)
            if kind in (Fault.THERMAL_THROTTLE, Fault.RUNAWAY_PROCESS):
                targets = rng.sample(range(cores), rng.randint(1, 2))
            else:
                targets = rng.sample(range(cores), rng.randint(1, 3))
            p.inject(kind, targets, start=rng.randint(10, 50))
        fleet.append(p)
    return fleet


def summarize(sample):
    cs = sample["cores"]
    return {
        "node": sample["processor"],
        "tick": sample["tick"],
        "util_avg": round(sum(c["util"] for c in cs) / len(cs), 1),
        "temp_max": max(c["temp_c"] for c in cs),
        "freq_min": min(c["freq_mhz"] for c in cs),
        "lat_max_us": max(c["sched_latency_us"] for c in cs),
        "irq_max": max(c["irq_per_s"] for c in cs),
        "ipc_min": min(c["ipc"] for c in cs),
        "miss_max": max(c["cache_miss_pct"] for c in cs),
        "rq_max": max(c["run_queue"] for c in cs),
    }


if __name__ == "__main__":
    fleet = make_fleet()
    print(f"{'node':7}{'tick':>5}{'util':>7}{'tempC':>7}{'MHz':>6}"
          f"{'lat_us':>8}{'irq/s':>8}{'ipc':>6}{'miss%':>7}{'rq':>4}  truth")
    for t in range(120):
        for p in fleet:
            sample, truth = p.step()
            if t % 40 == 39:
                s = summarize(sample)
                tag = ", ".join(f"{x['fault']}{x['cores']}" for x in truth) or "healthy"
                print(f"{s['node']:7}{s['tick']:>5}{s['util_avg']:>7}{s['temp_max']:>7}"
                      f"{s['freq_min']:>6}{s['lat_max_us']:>8}{s['irq_max']:>8}"
                      f"{s['ipc_min']:>6}{s['miss_max']:>7}{s['rq_max']:>4}  {tag}")
        if t % 40 == 39:
            print()
