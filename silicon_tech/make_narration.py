"""Ask Nemotron to write the demo narration from recorded facts; numbers are checked in code.
Run: python -m silicon_tech.make_narration   (writes docs/narration.json)"""
import json
import os
from silicon_tech import llm
from silicon_tech.copilot import grounded

SYSTEM = ("You are Silicon Technician, an AI data center co-pilot built on NVIDIA Nemotron, giving a live "
          "demo to an on-site data center manager. Speak in first person, plain spoken English, 3 to 5 short "
          "sentences, no markdown, no lists. Use ONLY the facts given. Never introduce a number that is not in "
          "the facts, never say 'step', and never claim anything the facts do not state. Say the action taken "
          "and, when given, the guardrail result.")
FACTS = [
 "Server looks healthy. CPU Temp 57.4C, headroom to critical 37.6C. Inlet 24.0C, headroom 18.0C. Fan 1 9103 RPM OK. PSU 1 699 W OK. Management controller (BMC) is an emulator. Guardrails passed. Action: none.",
 "Fan 3 reads 0 RPM, Critical. Event log: Fan 3 speed below lower critical threshold. CPU Temp 88.6C, headroom to critical 6.4C, Warning. Guardrails passed. Action: lit the blue locate LED (safe action) so a technician can find the rack.",
 "PSU 2 outputs 0 W, Critical. Log: Power supply 2 output degraded; redundancy lost. Power redundancy Warning. Guardrails passed. Action: lit the blue locate LED. Checks: status LEDs, cable seating, replacement.",
 "Inlet temperature 38.2C, headroom to critical 3.8C, Warning. Log: inlet temperature above upper non-critical threshold. Checks: cooling unit operation, airflow obstructions. Guardrails passed. Action in this recording: none.",
 "All six fans near 15000 RPM, which is unusually high. Maintenance note says a fan diagnostic job is running and high fan speeds are expected. Event log confirms a scheduled fan diagnostic. Temperatures OK: CPU 57.6C. Conclusion: scheduled maintenance. Action: none.",
 "BMC readings all OK: CPU 58.5C, GPU 61.3C, inlet 24.4C, fans near 9000 RPM, both PSUs 705 W. Host core 0 is at 100.0% utilization with scheduler delay 1716.4 microseconds against a baseline of 9.1% utilization and 43.1 microseconds. No maintenance scheduled. Conclusion: host CPU starvation, invisible to the BMC. The model's own answer was flagged for human review because it quoted a ratio that was not in the data. Action: none.",
 "Scorecard: on 60 fresh test cases fixed rules got 20 right and Ultra got all 60. Limits: the BMC is an emulator, the cases come from our own simulator, no GPU server was involved. What is real: the processor measurement on the host and the guardrails.",
]


def main():
    model = os.environ.get("NEBIUS_MODEL")
    out = []
    for i, facts in enumerate(FACTS):
        text = None
        for attempt in range(3):
            t, _ = llm.ask(SYSTEM, facts, model=model, max_tokens=4000, temperature=0.4)
            t = (t or "").split("</think>")[-1]
            t = " ".join(t.split()).strip('"')
            ok, bad = grounded([t], facts)
            if t and ok and len(t) < 700:
                text = t
                break
            print(f"part {i + 1}: attempt {attempt + 1} rejected {bad or 'empty or too long'}")
        out.append(text)
        print(f"part {i + 1}:", "accepted" if text else "fallback to our written script")
    os.makedirs("docs", exist_ok=True)
    json.dump({"model": model, "steps": out}, open("docs/narration.json", "w"), indent=1)
    print("saved docs/narration.json")


if __name__ == "__main__":
    main()
