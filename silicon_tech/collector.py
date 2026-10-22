"""Live host collector: real CPU telemetry from this machine (Linux).
Scheduler latency comes passively from /proc/schedstat (no probe overhead).
Counters the host cannot provide are None."""
import os
import time
import psutil


def read_schedstat():
    """{cpu: (run_time_ns, run_delay_ns, timeslices)} from /proc/schedstat, or {}."""
    out = {}
    try:
        with open("/proc/schedstat") as f:
            for line in f:
                if line.startswith("cpu"):
                    p = line.split()
                    out[int(p[0][3:])] = (int(p[-3]), int(p[-2]), int(p[-1]))
    except (OSError, ValueError, IndexError):
        pass
    return out


class LiveHost:
    def __init__(self):
        if hasattr(os, "sched_getaffinity"):
            self.cores = sorted(os.sched_getaffinity(0))
        else:
            self.cores = list(range(psutil.cpu_count()))
        psutil.cpu_percent(percpu=True)          # prime the counters
        self.prev = psutil.cpu_stats()
        self.prev_t = time.time()
        self.prev_sched = read_schedstat()
        self.tick = 0

    def _temp(self):
        try:
            temps = psutil.sensors_temperatures()
            vals = [e.current for entries in temps.values() for e in entries]
            return round(max(vals), 1) if vals else None
        except (AttributeError, OSError):
            return None

    def _sched_delay_us(self, cur, c):
        """Average run-queue wait per scheduling slice since the last sample."""
        a, b = self.prev_sched.get(c), cur.get(c)
        if not a or not b:
            return None
        slices = b[2] - a[2]
        if slices <= 0:
            return 0.0
        return round((b[1] - a[1]) / slices / 1000.0, 1)

    def step(self):
        util = psutil.cpu_percent(percpu=True)
        now = time.time()
        st = psutil.cpu_stats()
        dt = max(1e-6, now - self.prev_t)
        ctx = (st.ctx_switches - self.prev.ctx_switches) / dt
        irq = (st.interrupts - self.prev.interrupts) / dt
        self.prev, self.prev_t = st, now
        cur = read_schedstat()
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
                "wake_latency_us": self._sched_delay_us(cur, c),
            })
        self.prev_sched = cur
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
