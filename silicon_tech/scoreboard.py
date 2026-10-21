"""Scores a diagnoser against the fault-injection ground truth."""
import sys
from silicon_tech.simulator import make_fleet
from silicon_tech.diagnoser import diagnose


def run(diagnose_fn=diagnose, seed=42, nodes=12, ticks=120, verbose=True):
    fleet = make_fleet(n=nodes, seed=seed)
    rows = []
    for p in fleet:
        start = p.faults[0]["start"] if p.faults else None
        label = p.faults[0]["fault"].value if p.faults else "healthy"
        first_ok, fp_ticks = None, 0
        for _ in range(ticks):
            sample, truth = p.step()
            want = {t["fault"]: sorted(t["cores"]) for t in truth}
            got = diagnose_fn(sample)
            if want and first_ok is None and got == want:
                first_ok = sample["tick"] - start
            if any(k not in want for k in got):
                fp_ticks += 1
        rows.append((p.name, label, first_ok, fp_ticks))

    faulty = [r for r in rows if r[1] != "healthy"]
    hits = [r for r in faulty if r[2] is not None]
    if verbose:
        print(f"{'node':8}{'truth':20}{'ticks_to_dx':>12}{'false_alarm_ticks':>19}")
        for name, label, ttd, fp in rows:
            t = "-" if label == "healthy" else ("MISSED" if ttd is None else ttd)
            print(f"{name:8}{label:20}{str(t):>12}{fp:>19}")
        print()
        print(f"diagnosed {len(hits)}/{len(faulty)} faulty nodes")
        if hits:
            print(f"mean ticks to diagnosis: {sum(r[2] for r in hits)/len(hits):.1f}")
        print(f"total false-alarm ticks: {sum(r[3] for r in rows)}")
    return rows


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run(seed=seed)