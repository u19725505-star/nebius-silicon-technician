<div align="center">

# SILICON TECHNICIAN
### Physical AI Mission Lab · AI-Assisted Data-Center Operations

**See the signal. Understand the system. Keep humans in control.**

[![Live Mission Lab](https://img.shields.io/badge/Live-Mission%20Lab-76B900?style=for-the-badge&logo=render&logoColor=white)](https://silicon-technician-judge-api.onrender.com/)
[![NVIDIA Nemotron](https://img.shields.io/badge/Model-NVIDIA%20Nemotron-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![Nebius Token Factory](https://img.shields.io/badge/Inference-Nebius%20Token%20Factory-273849?style=flat-square)](https://tokenfactory.nebius.com/)
[![Safety](https://img.shields.io/badge/Hardware%20Control-Disabled-success?style=flat-square)](#safety-by-design)

*A human-supervised technician co-pilot combining structured diagnosis, deterministic checks, host scheduler measurements, and a browser-based inspection simulation.*

</div>

---

## Mission at a glance

| Capability | What it demonstrates |
|---|---|
| **Live model inference** | An opt-in backend request to NVIDIA Nemotron through Nebius Token Factory |
| **Three-rack inspection** | A browser agent following a deterministic route through a virtual data-center floor |
| **Fault scenarios** | Fan failure, hot inlet, PSU anomaly, and a healthy control condition |
| **Evidence-first reporting** | Rack telemetry, scenario findings, diagnosis output, and local JSON/TXT export |
| **Safety boundaries** | Human review required; no physical robot or server controls exposed by the Mission Lab |

> **Judge quick start:** Open the [Live Mission Lab](https://silicon-technician-judge-api.onrender.com/), choose **Fan failure**, **Hot inlet**, or **PSU anomaly**, and run the three-rack inspection. Review the evidence and open or export the mission report. **Show recorded advisory** is local and makes no model call. **Run live Nemotron** sends a request to the configured backend and may consume Token Factory credits; it is optional.

## The problem

Data-center faults rarely explain themselves through a single threshold. Sensor anomalies can trigger false alarms, scheduled maintenance can resemble a fault, and CPU scheduling contention may not appear in baseboard management controller (BMC) telemetry. Technicians need evidence-linked reasoning, bounded next steps, and a clear separation between model recommendations and authorized action.

## What we built

### 01 — AI technician co-pilot

The Python co-pilot supplies structured telemetry and an emulated Redfish-style BMC to NVIDIA Nemotron through Nebius Token Factory. Deterministic checks cross-check model output, actions are constrained to a fixed set, and inference failures can fall back to a rule-based diagnoser.

A host collector reads processor scheduling data from `/proc/schedstat`. In recorded contention experiments, run-queue delay increased when two CPU-bound tasks competed on the same core. This illustrates a host-level signal that may not be visible in BMC telemetry.

### 02 — Physical AI Mission Lab

The hosted browser experience presents a virtual data-center floor, three racks, a simulated inspection agent, voice narration, evidence reporting, and four selectable conditions:

- **Fan failure** — synthetic zero-RPM fan reading and elevated CPU temperature.
- **Hot inlet** — synthetic elevated inlet temperature and thermal stress.
- **PSU anomaly** — synthetic low power-supply output.
- **Healthy run** — baseline control condition.

The robot follows a deterministic local route. Users can inspect rack findings, view a recorded example advisory, and export JSON or readable TXT reports. The export itself does not call a model.

### 03 — Live diagnosis and simulated recovery

A successful live test using `nvidia/Nemotron-3-Ultra-550b-a55b` returned a schema-validated **CRITICAL** diagnosis for the simulated Rack B fan failure. The model cited a 0 RPM fan reading, 91°C CPU temperature, and 9°C thermal headroom. The recorded request latency was approximately 9.8 seconds.

The browser then ran a separate **simulated** recovery: displayed fan speed changed from 0 to 7,200 RPM and CPU temperature from 91°C to 64°C, with the simulation's verification result set to `PASS`.

**Interpretation:** the model inference was live; the scenario telemetry and before/after recovery values were simulated. No physical fan was repaired and no hardware command was executed.

## System flow

```text
Synthetic scenario + rack telemetry
                |
                v
   Deterministic inspection route
                |
                v
 Evidence report + optional live inference
                |
                v
    Schema and policy validation
                |
                v
 Human review + browser-only simulation
                |
                v
       Local JSON / TXT export
```

## Evaluation

The repository contains recorded simulator bake-off outputs. Results on generated test cases:

| Generated test set | Fixed-rule baseline | Nemotron Ultra |
|---|---:|---:|
| 20 cases, seed 777 | 6/20 | 20/20 (Super 19/20; Nano 16/20) |
| 60 cases, seed 2026, prompt v2 | 17/60 | 57/60 (Super 53/60) |
| 60 cases, seed 3031, prompt v3, fresh set | 20/60 | 60/60 |

The fresh 60-case set recorded a median Ultra diagnosis time of 5.4 seconds. These are results from **our own simulator**, not an independent benchmark or evidence of real-hardware diagnostic accuracy. A perfect score on 60 generated cases does not establish certainty or production readiness. See `bakeoff_results.txt`, `bakeoff_n60.txt`, `bakeoff_n60_seed3031.txt`, and `bakeoff_ultra.txt`.

### Host scheduler experiment

In one recorded experiment, core 0's average run-queue wait per scheduling slice rose from about 43 microseconds at idle to 1,716 microseconds while two CPU-bound tasks competed for the core; utilization reached 100%. A single busy loop on an otherwise empty core did not produce the same delay, so the contention setup used two competing tasks.

These are real measurements from the machine running the experiment, not measurements from a production GPU server. The experiment ran in a small container.

## Safety by design

- **No physical hardware connection in the Mission Lab.** The browser experience has no connected robot, production server, BMC, or live camera.
- **Synthetic perception.** Rack markers are SVG-rendered scenario indicators. They are not validated computer vision, and marker state must not be described as camera-confirmed detection.
- **Synthetic Mission Lab telemetry.** Values are scenario data, not live equipment readings.
- **Human review required.** Model output is advisory, not an eligibility, maintenance, or execution authority.
- **Simulation-only recovery.** A displayed recovery changes simulated values only; `hardware_action_executed` remains false.
- **Separate local and live modes.** Recorded advisory and report export make no model call. Live inference is opt-in and may consume credits.
- **Bounded co-pilot actions.** The separate Python co-pilot uses an emulated BMC and a fixed action list; a reset request requires human approval.

## Run locally

Requires Python. Live inference additionally requires a Nebius Token Factory API key.

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY="your-key"
export NEBIUS_MODEL="nvidia/Nemotron-3-Ultra-550b-a55b"

# Run the co-pilot with emulated BMC data and host measurements
python -m silicon_tech.copilot

# Re-run a simulator bake-off (live model calls consume credits)
python -m silicon_tech.bakeoff --models "$NEBIUS_MODEL" --n 60 --seed 3031

# Full audit including live inference tests (may consume credits)
python -m silicon_tech.audit --live
```

**Cost note:** do not run the live bake-off or `audit --live` unless you intend to use API credits. The hosted Mission Lab's recorded-advisory path can be explored without a live request.

## Repository map

- `silicon_tech/` — simulator, rule baseline, agent, Redfish emulator, host collector, co-pilot, and audit code
- `judge_api.py` — hosted demo API
- `docs/` — browser Mission Lab and judge-facing pages
- `bakeoff_*.txt`, `copilot_run*.txt`, `results_hard*.txt` — recorded benchmark and experiment outputs
- `narration.json` — scripted narration examples

## Known limitations

- The BMC is emulated; the Mission Lab is not connected to physical infrastructure.
- The browser route is deterministic, and visual markers are synthetic SVG state—not validated computer vision.
- Mission Lab telemetry and recovery values are synthetic.
- Generated benchmark cases may not represent the distribution of real incidents.
- The host contention experiment ran in a small container.
- Model outputs can be wrong or raise false alarms. Deterministic checks and human review remain necessary.
- Live inference requires network access and may consume credits.
- Recorded advisories, live model responses, and simulated recovery results are distinct evidence types.

## Developer-tool feedback

The OpenAI-compatible Token Factory endpoint made integration straightforward. Model ID naming differences and low token limits for reasoning models caused development friction; usage visibility helped track inference spending.

## License

See [LICENSE](LICENSE).
