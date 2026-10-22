# Silicon Technician

An AI data center technician for processors. It reads CPU telemetry the way a floor technician would, decides whether a node is healthy, and writes a diagnosis with a recommended action. Built for the Nebius x NVIDIA Global AI Hackathon, using NVIDIA Nemotron served from Nebius Token Factory.

## The problem

Fixed-threshold monitoring has three blind spots: it misses mild faults, it raises false alarms on sensor glitches, and it cannot tell a runaway process from a scheduled batch job. A technician handles all three by comparing cores with their peers, ignoring impossible readings, and checking the maintenance schedule. Silicon Technician gives Nemotron that same context.

## What it does

- **Simulated fleet** of 8-core processors with injectable faults: thermal throttle, noisy neighbor, runaway process, IRQ storm, memory stall. Faults can be mild, sensors can glitch, and benign scheduled jobs look like faults.
- **Rule-based baseline**: peer comparison with fixed thresholds.
- **Nemotron agent**: reads the latest tick, 8-tick medians, and maintenance context, then returns a JSON diagnosis with one fault per core, a summary, and an action.
- **Live-host collector**: reads this machine's real per-core utilization, context switches, interrupts, and scheduler run-queue delay from `/proc/schedstat`.
- **Zero-token gate**: the model is called only when the host differs from its healthy baseline.
- **Graceful fallback**: if the model call fails, the rule-based diagnoser takes over.

## Results

| Test | Rules | Nemotron |
|---|---|---|
| Held-out audit: 20 random cases (seed 777), not used for prompt tuning | 6/20 (30%) | **17/20 (85%)** |
| Hand-written hard set: 10 cases, prompt tuned on these | 1/10 | 8/10 and 9/10 (two runs) |

Held-out run: median 9.9 s per diagnosis (max 21.6 s), 28,388 tokens in and 39,671 out. Misses: two memory stalls where only one of the two affected cores was found, and one IRQ storm labelled as a noisy neighbor. Run-to-run variation exists even at temperature 0.

### Live processor

On the host running this repo, we start a real CPU hog on core 0. The average run-queue wait per scheduling slice on that core rose from 25.3 us to 121.9 us in the audit (18.6 us to 153.6 us in an earlier demo run), with utilization at 100%. An idle window sends nothing to the model and costs 0 tokens.

## Run it

    pip install -r requirements.txt
    export NEBIUS_API_KEY="your-key"
    export NEBIUS_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    python -m silicon_tech.simulator          # simulated fleet
    python -m silicon_tech.hard --live        # hard cases, rules vs Nemotron
    python -m silicon_tech.live_demo          # real host, real CPU hog
    python -m silicon_tech.audit --live       # full self-test

## Models and Nebius usage

Diagnoses use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` through Nebius Token Factory's OpenAI-compatible endpoint (`https://api.tokenfactory.nebius.com/v1/`). The code uses the standard `openai` Python client.

## Limitations

- The live demo runs on a small 2-core container: no temperature, IPC, or cache-miss counters, and the hog shares the CPU with the demo itself, so a second core also showed higher delay.
- The held-out cases come from the same simulator as the training-era cases. "Unseen" means unseen random draws, not different physics.
- The first prompt was tuned on the hand-written hard set, so that score is optimistic; the held-out number is the fairer one.
- Diagnosis only: the system recommends actions but does not take them.
- Latency per diagnosis is about 10 seconds with this reasoning model, which is too slow for per-tick use. The gate keeps calls rare.
- This has not been run on a GPU node or a production data center processor.

## Feedback on Nebius and NVIDIA tools

- The OpenAI-compatible endpoint made integration quick; the model listing call made it easy to find exact IDs.
- Model ID naming is inconsistent across NVIDIA models (mixed case, underscores, and dots), which caused guessing.
- With a reasoning model, a low `max_tokens` produced empty replies; a documented minimum or a clearer finish reason in the response would help.
- The usage page made cost easy to track: our full testing stayed within a few cents.

## Repository layout

- `silicon_tech/simulator.py`: processor simulator and fault injection
- `silicon_tech/diagnoser.py`, `scoreboard.py`: rule baseline and scoring
- `silicon_tech/hard.py`: hard-mode cases
- `silicon_tech/agent.py`, `llm.py`: Nemotron agent and Nebius client
- `silicon_tech/collector.py`, `live_agent.py`, `live_demo.py`: real-host path
- `silicon_tech/audit.py`: judge-readiness self-test

## License

See `LICENSE`.
