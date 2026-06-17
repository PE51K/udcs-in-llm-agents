# Verification Scripts

Scripts for running the UQ methods with fixed hyperparameters on the standard and
clarification benchmarks.

## Structure

```
verification/
├── run_parallel.sh                # Parallel sweep across benchmarks/methods
├── webshop/                       # Standard benchmarks
│   ├── react/
│   ├── uam/
│   └── proposed_v4/               # UAM confidence + u_request
├── alfworld/                      # (same method structure)
├── real/                          # (same method structure; + reaggregate_runs.py, hackable variants)
├── webshop_clarification/         # Clarification benchmarks (50% underspecified)
│   ├── react/
│   ├── uam/
│   └── proposed_v4/
└── alfworld_clarification/        # (same as webshop_clarification)
```

Each method directory contains a self-contained `<method>.py` script with its own agent
loop and UQ logic.

## Setup

Copy `.env.example` to `.env` at the project root and set your API credentials:

```bash
cp .env.example .env
```

Required `.env` fields:
- `BASE_URL` — API base URL (e.g. `https://openrouter.ai/api/v1`)
- `OPENAI_API_KEY` — API key for the provider
- `MODEL` — default model name (e.g. `openrouter/openai/gpt-5.1`)

Optional:
- `PROVIDER` — default value for `--provider` (see below); unset = no override

## Running

From the repository root:

```bash
uv run --group scripts python -m scripts.verification.<benchmark>.<method>.<script> [args]
```

Example:

```bash
uv run --group scripts python -m \
    scripts.verification.webshop.uam.uam \
    --model openrouter/openai/gpt-4.1 \
    --n_tasks 50 \
    --log_level INFO \
    --random_seed 42
```

### Pinning an OpenRouter provider

Every script accepts `--provider <slug>` (default none → OpenRouter auto-routes).
When set, requests pin that upstream provider via OpenRouter's `provider.order`
and disable fallbacks (`allow_fallbacks: false`), so a run uses exactly one
provider or errors instead of silently routing elsewhere. Defaults to the
`PROVIDER` env var.

```bash
uv run --group scripts python -m \
    scripts.verification.webshop.uam.uam \
    --model openrouter/openai/gpt-5.1 \
    --provider openai
```

To apply it across a whole parallel sweep, set `PROVIDER` for `run_parallel.sh`:

```bash
PROVIDER=openai scripts/verification/run_parallel.sh
```

## Output format

Each verification experiment writes results to `verification_results/` with the structure:

```
verification_results/
└── <benchmark>/<method>/<model>/
    ├── test_results.json   # Aggregated results (per-task scores and success flags)
    └── tasks/              # Individual task trajectories
        └── task_<id>.json  # Full trajectory for a single task
```

## Benchmark setup

The benchmark environments are included as git submodules under `benchmarks/`. After
`git submodule update --init --recursive`, see:

- **WebShop:** [`docs/env_setup/WEBSHOP_SETUP.md`](../../docs/env_setup/WEBSHOP_SETUP.md)
- **ALFWorld:** [`docs/env_setup/ALFWORLD_SETUP.md`](../../docs/env_setup/ALFWORLD_SETUP.md)
- **REAL:** AGISDK submodule at `benchmarks/agisdk` (installed via `uv sync`)
