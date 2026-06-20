# Uncertainty Decomposition for Clarification Seeking in LLM Agents

Public code release accompanying the paper
**"Uncertainty Decomposition for Clarification Seeking in LLM Agents."**

**Preprint:** [arXiv:2606.19559](https://arxiv.org/abs/2606.19559)

This repository contains the **experiment (verification) scripts** used to evaluate
prompt-based uncertainty quantification (UQ) for LLM agents — both fault detection on
standard benchmarks and clarification seeking on underspecified tasks.

## Verification results and trajectories

The full verification results and agent trajectories from the paper's experiments are
available here:

- **[Verification results and trajectories (Google Drive)](https://drive.google.com/file/d/1QUJUwy-rA00x3U2uo9v-o6KIwPhvMc0d/view?usp=sharing)**

## Methods

| Method | Directory | Description |
|--------|-----------|-------------|
| **ReAct+UE** | `react/` | ReAct agent with an inline confidence score |
| **UAM** | `uam/` | Uncertainty-Aware Memory with `<confidence>` tags |
| **Proposed (v4)** | `proposed_v4/` | Minimal decomposition: UAM confidence + user-request uncertainty (`u_request`) for clarification seeking |

## Where everything lies

```
.
├── pyproject.toml                     # Dependencies (uv) + tooling config
├── .env.example                       # Template for API credentials (copy to .env)
└── scripts/
    └── verification/                  # All experiment scripts
        ├── README.md                  # How to run a single experiment + flags
        ├── run_parallel.sh            # Run a parallel sweep across benchmarks/methods
        │
        ├── webshop/                   # Standard benchmarks
        ├── alfworld/                  #   each contains: react/ uam/ proposed_v4/
        ├── real/                      #   (real/ also has reaggregate_runs.py + hackable variants)
        │
        ├── webshop_clarification/     # Clarification benchmarks (50% underspecified tasks)
        └── alfworld_clarification/    #   each contains: react/ uam/ proposed_v4/
```

Each `<benchmark>/<method>/<method>.py` is **self-contained**: it implements its own
agent loop and UQ logic and does not import shared internal packages.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/). Python 3.11 (see `.python-version`).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Fetch the benchmark submodules (WebShop, ALFWorld, AGISDK/REAL)
git submodule update --init --recursive

# 3. Provide API credentials
cp .env.example .env
# then edit .env:
#   BASE_URL        — API base URL (e.g. https://openrouter.ai/api/v1)
#   OPENAI_API_KEY  — API key for the provider
#   MODEL           — default model (e.g. openrouter/openai/gpt-5.1)

# 4. Install dependencies
uv sync
```

> If you cloned without `--recursive`, run step 2 to populate `benchmarks/`.

## Running experiments

From the repository root:

```bash
uv run --group scripts python -m scripts.verification.<benchmark>.<method>.<script> [args]
```

Example:

```bash
uv run --group scripts python -m \
    scripts.verification.webshop.uam.uam \
    --model openrouter/openai/gpt-5.1 \
    --n_tasks 50 \
    --log_level INFO \
    --random_seed 42
```

Every script accepts `--help` for its full argument list. See
[`scripts/verification/README.md`](scripts/verification/README.md) for details, including
how to pin an OpenRouter provider and how to launch a parallel sweep with `run_parallel.sh`.

## Benchmark environments

The benchmark environments (WebShop, ALFWorld, AGISDK/REAL) are included as git submodules
under `benchmarks/`, pinned to the exact commits used in the paper. After
`git submodule update --init --recursive`, follow the per-benchmark setup guides to download
data and start the required services:

- **WebShop:** [`docs/env_setup/WEBSHOP_SETUP.md`](docs/env_setup/WEBSHOP_SETUP.md)
- **ALFWorld:** [`docs/env_setup/ALFWORLD_SETUP.md`](docs/env_setup/ALFWORLD_SETUP.md)
- **REAL:** the AGISDK submodule at `benchmarks/agisdk` (installed via `uv sync` as a
  workspace member)
