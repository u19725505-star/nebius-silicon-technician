# Silicon Technician — AI Data-Center Technician Co-Pilot

**A human-supervised technician co-pilot that combines NVIDIA Nemotron diagnosis, deterministic checks, host scheduler measurements, and a browser-based Physical AI Mission Lab.**

Built for the **Nebius × NVIDIA Global AI Hackathon**, using NVIDIA Nemotron through Nebius Token Factory.

- **Live Mission Lab:** https://silicon-technician-judge-api.onrender.com/
- **Source code:** https://github.com/u19725505-star/nebius-silicon-technician

> **Judge quick start:** Open the live demo, choose **Fan failure**, **Hot inlet**, or **PSU anomaly**, inspect the three racks, then open the evidence report and export JSON or TXT. **Show recorded advisory** is local and does not call the model. **Run live Nemotron** sends a request to the configured backend and may consume Token Factory credits; it is optional.

## The problem

Data-center faults are not always obvious from one threshold. Sensor glitches can create false alarms, scheduled maintenance can resemble a fault, and a busy host can suffer CPU scheduling contention that a baseboard management controller (BMC) does not reveal. Technicians need a diagnosis tied to evidence, a safe next step, and a clear boundary between model suggestions and actions.

## What we built

Silicon Technician brings together two related experiences:

### 1. AI technician co-pilot

The Python co-pilot supplies structured telemetry and an emulated Redfish-style BMC to NVIDIA Nemotron through Nebius Token Factory. It checks model output against deterministic code, constrains available actions, logs actions, and falls back to a rule-based diagnoser when inference fails.

A host collector reads processor scheduling data from `/proc/schedstat`. In recorded contention experiments, run-queue delay increased when two CPU-bound tasks competed on the same core. This illustrates a host-level signal that may not be visible in BMC telemetry.

### 2. Physical AI Mission Lab

The hosted browser demo presents a virtual data-center floor with three racks and four selectable conditions:

- **Fan failure** — simulated zero-RPM fan and elevated CPU temperature.
- **Hot inlet** — simulated elevated inlet temperature and secondary thermal stress.
- **PSU anomaly** — simulated low PSU output.
- **Healthy run** — baseline control condition.

Users can inspect all racks, hear browser-generated speech summaries, review a recorded example advisory, and open or export a mission evidence report. JSON and readable TXT exports include telemetry, scenario findings, model-status disclosure, and safety limitations.

**The Mission Lab is a simulation:** rack markers are SVG-rendered and telemetry is synthetic. It has no live camera, validated computer vision, physical robot, or connected server hardware. Its recorded advisories are examples, not model-generated diagnoses. The separate Python co-pilot contains the live inference integration.

## Evaluation

The repository contains reproducible simulator bake-off output. On our generated test sets, the recorded results were:

| Generated test set | Fixed-rule baseline | Nemotron Ultra |
|---|---:|---:|
| 20 cases, seed 777 | 6/20 | 20/20 (Super 19/20; Nano 16/20) |
| 60 cases, seed 2026, prompt v2 | 17/60 | 57/60 (Super 53/60) |
| 60 cases, seed 3031, prompt v3, fresh set | 20/60 | 60/60 |

The fresh 60-case set had a recorded median Ultra diagnosis time of 5.4 seconds. These are results from **our own simulator**, not an independent benchmark or proof of real-hardware accuracy. A perfect score on 60 generated cases does not establish certainty or production readiness. See `bakeoff_results.txt`, `bakeoff_n60.txt`, `bakeoff_n60_seed3031.txt`, and `bakeoff_ultra.txt` for the recorded outputs.

### Host scheduler experiment

The project reads per-core scheduling delay from `/proc/schedstat`. In one recorded experiment, core 0's average run-queue wait per scheduling slice rose from about 43 microseconds at idle to 1,716 microseconds while two CPU-bound tasks competed for the core; utilization reached 100%. A single busy loop on an otherwise empty core did not produce the same delay, so the contention setup used two competing tasks.

These host measurements are real measurements from the machine running the experiment; they are not measurements from a production GPU server. The experiment was run in a small container.

## Safety and trust boundaries

- **Evidence grounding:** code checks that quoted numbers are present in the supplied data; unsupported computed ratios can be rejected and flagged for human review.
- **Deterministic cross-checks:** model diagnoses are compared with checks computed by code.
- **Bounded actions:** actions are selected from a fixed list and recorded in an audit log. The only automatic action described in the implementation is lighting a locate LED in the emulated environment; a reset request requires human approval.
- **Fallback behavior:** if inference fails, a rule-based diagnoser can take over.
- **Human oversight:** model output is advisory; it is not a substitute for an authorized technician.
- **Cost awareness:** the Mission Lab separates local recorded advice from an explicitly confirmed live inference request. Live requests may consume Token Factory credits.

## Run locally

Requires Python and a Nebius Token Factory API key for live inference.

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY="your-key"
export NEBIUS_MODEL="nvidia/Nemotron-3-Ultra-550b-a55b"

# Run the co-pilot with emulated BMC data and host measurements
python -m silicon_tech.copilot

# Re-run a simulator bake-off (live model calls consume credits)
python -m silicon_tech.bakeoff --models "$NEBIUS_MODEL" --n 60 --seed 3031

# Full audit, including live inference tests (may consume credits)
python -m silicon_tech.audit --live
```

Do not run the live bake-off or `audit --live` unless you intend to use API credits. The hosted Mission Lab's recorded-advisory path can be explored without a live model request.

## Repository map

- `silicon_tech/` — simulator, rule baseline, agent, Redfish emulator, host collector, co-pilot, and audit code
- `judge_api.py` — hosted demo API
- `docs/` — browser Mission Lab and judge-facing pages
- `bakeoff_*.txt`, `copilot_run*.txt`, `results_hard*.txt` — recorded benchmark and experiment output
- `narration.json` — scripted narration examples

## Known limitations

- The BMC is emulated; no real BMC, GPU server, camera, or robot is connected to the Mission Lab.
- Mission Lab telemetry and visual markers are synthetic; its marker display is not validated computer vision.
- The benchmark cases are generated by our own simulator and may not represent the distribution of real incidents.
- The host contention test ran in a small container, not on a production GPU server.
- Model outputs can be wrong or raise false alarms. Human review and deterministic checks remain necessary.
- Live inference requires network access and may consume credits.
- The Mission Lab's recorded advisories and live-model output must not be presented as the same thing.

## Developer-tool feedback

The OpenAI-compatible Token Factory endpoint made integration straightforward. Model ID naming differences and low token limits for reasoning models caused integration friction during development. Usage visibility helped track inference spending.

## License

See [LICENSE](LICENSE).
