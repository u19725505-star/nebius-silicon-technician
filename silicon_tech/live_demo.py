"""Live demo on a real machine: baseline -> quiet check -> real CPU hog -> diagnosis.
Run: python -m silicon_tech.live_demo"""
import os
import time
import multiprocessing as mp
from silicon_tech.collector import LiveHost
from silicon_tech.live_agent import diagnose_live, summarize, describe


def hog(core):
    os.sched_setaffinity(0, {core})
    while True:
        pass


def collect(host, n, interval=1.0):
    out = []
    for _ in range(n):
        time.sleep(interval)
        out.append(host.step())
    return out


def show(title, result, tin, tout):
    print(f"\n== {title}")
    if result is None:
        print("   model returned no usable answer")
        return
    print("   status :", result.get("status"))
    print("   faults :", result.get("faults") or "none")
    print("   summary:", result.get("summary"))
    print("   action :", result.get("action"))
    print(f"   tokens : {tin} in, {tout} out")


def main():
    host = LiveHost()
    host.step()                                   # warm-up sample, discarded
    print("Collecting healthy baseline (about 10 s)...")
    baseline = collect(host, 8)
    print(describe("BASELINE", summarize(baseline)))

    print("\nChecking a quiet window (about 8 s)...")
    quiet = collect(host, 6)
    show("QUIET WINDOW", *diagnose_live(baseline, quiet))

    core = host.cores[0]
    print(f"\nInjecting a real CPU hog on core {core}...")
    p = mp.get_context("fork").Process(target=hog, args=(core,), daemon=True)
    p.start()
    try:
        time.sleep(3)
        busy = collect(host, 6)
    finally:
        p.terminate()
    print(describe("\nWITH HOG", summarize(busy)))
    show("WITH HOG", *diagnose_live(baseline, busy))


if __name__ == "__main__":
    main()

def hog2(core):
    """Two CPU-bound tasks sharing one core: a stand-in for a serving engine's
    CPU thread competing with a noisy process, so runnable work has to wait."""
    import signal
    import multiprocessing as _mp
    os.sched_setaffinity(0, {core})
    child = _mp.get_context("fork").Process(target=hog, args=(core,), daemon=True)
    child.start()

    def _stop(*_):
        child.terminate()
        os._exit(0)

    signal.signal(signal.SIGTERM, _stop)
    while True:
        pass
