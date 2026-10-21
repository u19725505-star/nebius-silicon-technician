"""Live host collector: real CPU telemetry from this machine (Linux).
Counters the host cannot provide are None."""
import os
import time
import psutil


def _wake_latency_us(core, samples=15, target_s=0.001):
    """p90 of how late a 1 ms sleep wakes up while pinned to `core`
    (a real scheduler-latency proxy)."""
    try:
        old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        return None
    try:
        late = []
        for _ in range(samples):
            t0 = time.perf_counter()
            time.sleep(target_s)
            late.append(max(0.0, (time.perf_counter() - t0 - target_s) * 1e6))
        late.sort()
        return round(late[int(len(late) * 0.9)], 1)
    finally:
        os.sched_setaffinity(0, old)


class LiveHost:
    def __init__(self):
        if hasattr(os, "sched_getaffinity"):
            self.cores = sorted(os.sched_getaffinity(0))
        else:
            self.cores = list(range(psutil.cpu_count()))
        psutil.cpu_percent(percpu=True)          # prime the counters
        self.prev = psutil.cpu_stats()
        self.prev_t = time.time()
        self.tick = 0

    def _temp(self):
        try:
            temps = psutil.sensors_temperatures()
            vals = [e.current for entries in temps.values() for e in entries]
            return round(max(vals), 1) if vals else None
        except (AttributeError, OSError):
            return None

    def step(self):
        util = psutil.cpu_percent(percpu=True)
        now = time.time()
        st = psutil.cpu_stats()
        dt = max(1e-6, now - self.prev_t)
        ctx = (st.ctx_switches - self.prev.ctx_switches) / dt
        irq = (st.interrupts - self.prev.interrupts) / dt
        self.prev, self.prev_t = st, now
        try:
            freqs = psutil.cpu_freq(percpu=True) or []
        except (AttributeError, OSError, NotImplementedError):
            freqs = []
        temp = self._temp()
        rows = []
        for c in self.cores:
            rows.append({
                "core": c,
                "util": round(util[c], 1) if c < len(util) else None,
                "freq_mhz": round(freqs[c].current) if c < len(freqs) else None,
                "temp_c": temp,
                "wake_latency_us": _wake_latency_us(c),
            })
        sample = {
            "tick": self.tick,
            "host": os.uname().nodename,
            "load1": round(os.getloadavg()[0], 2),
            "ctx_per_s": round(ctx),
            "irq_per_s": round(irq),
            "cores": rows,
        }
        self.tick += 1
        return sample

def _wake_latency_us(core, samples=6, spins=600000):
    """p90 scheduling delay (wall time minus CPU time, in microseconds) of a fixed
    compute probe pinned to `core`. Rises when other work competes for the core."""
    try:
        old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        return None
    try:
        delays = []
        for _ in range(samples):
            w0, c0 = time.perf_counter(), time.process_time()
            x = 0
            for i in range(spins):
                x += i
            wall = time.perf_counter() - w0
            cpu = time.process_time() - c0
            delays.append(max(0.0, wall - cpu) * 1e6)
        delays.sort()
        return round(delays[int(len(delays) * 0.9)], 1)
    finally:
        os.sched_setaffinity(0, old)
