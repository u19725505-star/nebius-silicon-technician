"""Rack-side technician co-pilot. Fuses an EMULATED BMC (Redfish), REAL host CPU
readings from this machine, and maintenance notes; diagnoses with Nemotron;
guardrails are enforced in code, not in the prompt.
Run:  python -m silicon_tech.copilot
"""
import os
import re
import time
import multiprocessing as mp
from silicon_tech import llm
from silicon_tech.agent import parse
from silicon_tech.redfish_sim import SCENARIOS, _start, _req, snapshot
from silicon_tech.collector import LiveHost
from silicon_tech.live_agent import summarize, describe, gate
from silicon_tech.live_demo import hog, collect

CAUSES = ["healthy", "fan_failure", "psu_degraded", "inlet_hot",
          "scheduled_maintenance", "host_cpu_starvation"]
ACTIONS = {"none", "locate_led", "request_reset"}
DEFAULT_MAINT = "No maintenance scheduled on this node."
MAINT = {"planned_fan_test":
         "Scheduled: fan diagnostic job running on this node (high fan speeds are expected)."}
TRUTH = {"healthy": "healthy", "fan_failure": "fan_failure", "psu_degraded": "psu_degraded",
         "inlet_hot": "inlet_hot", "planned_fan_test": "scheduled_maintenance",
         "cpu_starvation": "host_cpu_starvation"}
FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b"

SYSTEM = """You are a rack-side data center technician co-pilot. You are given:
(1) BMC readings (Redfish): temperatures with headroom already computed, fans, power supplies, event log;
(2) host CPU readings measured on the server itself: baseline and current, per core utilization and scheduler delay;
(3) maintenance notes.
Pick exactly one root cause from: healthy, fan_failure, psu_degraded, inlet_hot, scheduled_maintenance, host_cpu_starvation.
Rules: use only numbers that appear in the data and never compute new numbers. Work that the maintenance notes say is scheduled is not a fault. host_cpu_starvation means the BMC looks fine but a host core is saturated with scheduler delay far above its baseline. If unsure, say so with low confidence.
Allowed actions: none, locate_led (lights the locate LED so a technician can find the rack; safe), request_reset (asks a human to approve a reset; you never do it yourself).
List 1 to 3 physical checks a technician should do. Reply with ONLY JSON:
{"root_cause": "...", "evidence": ["short quote with numbers from the data"], "physical_checks": ["..."], "action": "none|locate_led|request_reset", "action_reason": "...", "confidence": "low|medium|high"}"""


def bmc_text(s):
    lines = [f"overall health {s['health']}, consumed {s['consumed_w']} W"]
    for n, t in s["temps"].items():
        lines.append(f"{n}: {t['c']}C, headroom to critical {t['headroom_c']}C, {t['health']}")
    for n, f in s["fans"].items():
        lines.append(f"{n}: {f['rpm']} RPM, {f['health']}")
    for n, p in s["psu"].items():
        lines.append(f"{n}: {p['watts']} W, {p['health']}")
    lines.append(f"power redundancy: {s['redundancy']}")
    lines += [f"log {sev}: {msg}" for sev, msg in s["log"]]
    return "\n".join(lines)


NUM = re.compile(r"\d+(?:\.\d+)?")


def grounded(evidence, prompt):
    """Every number the model quotes must appear (within rounding) in the data it saw."""
    have = [float(x) for x in NUM.findall(prompt)]
    bad = []
    for e in evidence:
        for n in NUM.findall(e):
            v = float(n)
            if not any(abs(v - h) <= max(0.06, 0.01 * abs(h)) for h in have):
                bad.append(n)
    return not bad, bad


def corroborated(cause, snap, gate_open, maint):
    a = " ".join(snap["anomalies"]).lower()
    return {
        "healthy": not snap["anomalies"] and not gate_open,
        "fan_failure": "fan" in a and "(critical)" in a,
        "psu_degraded": "psu" in a or "redundancy" in a,
        "inlet_hot": "inlet" in a,
        "scheduled_maintenance": maint != DEFAULT_MAINT and "fan" in a,
        "host_cpu_starvation": gate_open,
    }.get(cause, False)


def rules_cause(snap, gate_open, maint):
    a = " ".join(snap["anomalies"]).lower()
    if maint != DEFAULT_MAINT and "fan" in a:
        return "scheduled_maintenance"
    if "fan" in a and "(critical)" in a:
        return "fan_failure"
    if "psu" in a or "redundancy" in a:
        return "psu_degraded"
    if "inlet" in a:
        return "inlet_hot"
    return "host_cpu_starvation" if gate_open else "healthy"


def ask_model(prompt):
    """Primary model, then fallback model; returns (parsed, model_used, tokens_in, tokens_out)."""
    tin = tout = 0
    for model in (os.environ.get("NEBIUS_MODEL"), FALLBACK_MODEL):
        if not model:
            continue
        for budget in (3000, 8000):
            try:
                text, usage = llm.ask(SYSTEM, prompt, model=model,
                                      max_tokens=budget, temperature=0.0)
            except Exception as e:
                print(f"   [model error: {type(e).__name__}]")
                break
            tin += usage.prompt_tokens
            tout += usage.completion_tokens
            parsed = parse(text)
            if parsed is not None:
                return parsed, model, tin, tout
    return None, "rules-fallback", tin, tout


def main():
    srv, base = _start(8801)
    print("BMC is EMULATED. Host CPU readings below are REAL, from this machine.")
    host = LiveHost()
    host.step()
    print("Collecting host baseline (about 8 s)...")
    base_w = collect(host, 6)
    correct = tin_all = tout_all = 0
    for name in SCENARIOS:
        _req(base, "/emulator/scenario", {"name": name}, "POST")
        proc = None
        if name == "cpu_starvation":
            proc = mp.get_context("fork").Process(target=hog, args=(host.cores[0],), daemon=True)
            proc.start()
            time.sleep(3)
        else:
            time.sleep(5)
        now_w = collect(host, 3)
        if proc:
            proc.terminate()
            proc.join(timeout=2)
        snap = snapshot(base)
        maint = MAINT.get(name, DEFAULT_MAINT)
        bs, ns = summarize(base_w), summarize(now_w)
        gate_open = gate(bs, ns)
        prompt = (f"MAINTENANCE: {maint}\n\nBMC (emulated Redfish):\n{bmc_text(snap)}\n\n"
                  + describe("HOST BASELINE (healthy)", bs) + "\n\n" + describe("HOST NOW", ns))
        parsed, used, a, b = ask_model(prompt)
        tin_all += a
        tout_all += b
        print(f"\n== {name}   (injected: {TRUTH[name]})")
        if parsed is None:
            cause = rules_cause(snap, gate_open, maint)
            print(f"   model unavailable -> rules fallback says: {cause}   mode: NEEDS HUMAN")
            correct += cause == TRUTH[name]
            continue
        cause = parsed.get("root_cause")
        action = parsed.get("action") if parsed.get("action") in ACTIONS else "none"
        ev = [str(e) for e in (parsed.get("evidence") or [])][:4]
        g_ok, bad = grounded(ev, prompt)
        c_ok = corroborated(cause, snap, gate_open, maint)
        auto = (action == "locate_led" and g_ok and c_ok
                and cause not in ("healthy", "scheduled_maintenance"))
        correct += cause == TRUTH[name]
        print(f"   model: {used.split('/')[-1]}   root cause: {cause}   confidence: {parsed.get('confidence')}")
        for e in ev:
            print("   evidence:", e)
        for c in (parsed.get("physical_checks") or [])[:3]:
            print("   check   :", c)
        print(f"   guardrails: numbers grounded={g_ok}{'' if g_ok else ' ' + str(bad)}  "
              f"corroborated by data={c_ok}")
        if action == "locate_led" and auto:
            _req(base, "/redfish/v1/Systems/1", {"LocationIndicatorActive": True}, "PATCH")
            print("   action  : locate_led EXECUTED (safe action, all guardrails passed)")
        elif action == "request_reset":
            print("   action  : reset REQUESTED. Needs human approval; not executed")
        elif action == "none":
            print("   action  : none")
        else:
            print(f"   action  : {action} NOT executed (guardrails failed or no fault); needs human review")
    print(f"\nroot cause matched the injected scenario in {correct}/{len(SCENARIOS)} runs")
    print(f"tokens: {tin_all} in, {tout_all} out")
    print("audit log of actions taken:", _req(base, "/emulator/audit")["actions"])
    srv.shutdown()


if __name__ == "__main__":
    main()