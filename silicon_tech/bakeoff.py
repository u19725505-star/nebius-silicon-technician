"""Model bake-off on the held-out audit cases.
Run:  python -m silicon_tech.bakeoff
      python -m silicon_tech.bakeoff --models "id1,id2" --n 20 --seed 777
"""
import os
import sys
import time
from statistics import median
from silicon_tech.audit import make_cases, run_spec
from silicon_tech.diagnoser import diagnose
from silicon_tech import agent

DEFAULT = [
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/nemotron-3-super-120b-a12b",
]


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def evaluate(model, cases, seed):
    ok = tin = tout = errors = 0
    times, misses = [], []
    for i, spec in enumerate(cases):
        window, truth, ctx = run_spec(spec, seed * 100 + i)
        t0 = time.time()
        try:
            faults, _, usage = agent.diagnose_with_llm(window, ctx, model=model)
            tin += usage.prompt_tokens
            tout += usage.completion_tokens
        except Exception as e:
            faults, errors = None, errors + 1
            print(f"   error: {type(e).__name__}", flush=True)
        dt = time.time() - t0
        times.append(dt)
        good = faults == truth
        ok += good
        if not good:
            misses.append((spec[0], truth, faults))
        print(f"   {spec[0]:30} {'OK  ' if good else 'MISS'} {dt:5.1f}s", flush=True)
    return {"model": model, "ok": ok, "n": len(cases), "med": median(times),
            "mx": max(times), "tin": tin, "tout": tout, "errors": errors, "misses": misses}


def main():
    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("Set NEBIUS_API_KEY first.")
    n, seed = int(arg("--n", 20)), int(arg("--seed", 777))
    models = arg("--models", ",".join(DEFAULT)).split(",")
    cases = make_cases(n, seed)
    rules_ok = sum(run_spec(s, seed * 100 + i)[1] == diagnose(run_spec(s, seed * 100 + i)[0][-1])
                   for i, s in enumerate(cases))
    results = []
    for m in models:
        print(f"\n== {m}", flush=True)
        results.append(evaluate(m, cases, seed))
    print(f"\n{'model':44}{'correct':>9}{'median':>8}{'max':>7}{'tok in':>9}{'tok out':>9}{'err':>5}")
    print(f"{'rules (no LLM)':44}{f'{rules_ok}/{n}':>9}{'-':>8}{'-':>7}{'0':>9}{'0':>9}{'0':>5}")
    for r in results:
        print(f"{r['model']:44}{str(r['ok'])+'/'+str(r['n']):>9}{r['med']:>7.1f}s{r['mx']:>6.1f}s"
              f"{r['tin']:>9}{r['tout']:>9}{r['errors']:>5}")
    for r in results:
        for name, truth, got in r["misses"]:
            print(f"MISS [{r['model'].split('/')[-1]}] {name}: truth={truth or 'none'} got={got or 'none'}")


if __name__ == "__main__":
    main()