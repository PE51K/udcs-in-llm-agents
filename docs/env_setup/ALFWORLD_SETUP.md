# ALfWorld Environment Setup

Setup guide for the ALfWorld benchmark used in verification scripts.

## Prerequisites

ALfWorld is included as a git submodule and declared as a dependency in `pyproject.toml`. No system-level prerequisites are needed beyond Python 3.11+.

---

## Setup Steps

Run all commands from the **project root** (`uncertainty-aware-web-agents/`).

### 1. Fetch the submodule

```bash
git submodule update --init benchmarks/alfworld
```

### 2. Install dependencies

```bash
uv sync
```
or (if cmake issue occures)
```bash
UV_LINK_MODE=copy uv sync
```

This installs `alfworld` (and its dependency `textworld`) automatically.

### 3. Download game data

```bash
uv run alfworld-download
```

This downloads PDDL files, game files, and MaskRCNN detector to `~/.cache/alfworld/` (~250 MB total):

```
~/.cache/alfworld/
├── json_2.1.1/              # Task JSON files
│   ├── train/
│   ├── valid_seen/
│   └── valid_unseen/
├── json_2.1.3_tw-pddl/      # TextWorld game files (.tw-pddl)
│   ├── train/
│   ├── valid_seen/
│   └── valid_unseen/
├── logic/
│   ├── alfred.pddl           # PDDL domain
│   └── alfred.twl2           # TextWorld grammar
└── detectors/
    └── mrcnn_alfred_objects_sep13_004.pth  # MaskRCNN (visual mode only)
```

> `ALFWORLD_DATA` defaults to `~/.cache/alfworld/` and is auto-set on import. Override with `export ALFWORLD_DATA=<path>` if needed.

---

## Verify

Configure `.env` at the project root first (see [verification/README.md](../../scripts/verification/README.md) — `BASE_URL`, `OPENAI_API_KEY`, `MODEL`), then:

```bash
uv run --group scripts python -m \
    scripts.verification.alfworld.react.react \
    --n_tasks 1 --log_level DEBUG --random_seed 42
```

A successful run loads the environment, runs one task, and prints step confidences and metrics.

---

## Task Types

ALfWorld has 6 task types (filter with `--task_types`):

| ID | Name | Description |
|----|------|-------------|
| 1 | `pick_and_place_simple` | Pick up object, place in receptacle |
| 2 | `look_at_obj_in_light` | Examine object under a lamp |
| 3 | `pick_clean_then_place_in_recep` | Clean object (sinkbasin), then place |
| 4 | `pick_heat_then_place_in_recep` | Heat object (microwave), then place |
| 5 | `pick_cool_then_place_in_recep` | Cool object (fridge), then place |
| 6 | `pick_two_obj_and_place` | Pick two objects, place together |

Example — run only Pick & Place and Clean & Place tasks:

```bash
uv run --group scripts python -m \
    scripts.verification.alfworld.react.react \
    --n_tasks 10 --task_types 1 3 --random_seed 42
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: alfworld` | Package not installed | Run `uv sync` |
| `FileNotFoundError` on game files | Data not downloaded | Run `uv run alfworld-download` |
| `ALFWORLD_DATA` path errors | Custom data path not set | `export ALFWORLD_DATA=<path>` or use default `~/.cache/alfworld/` |
| `No game files found` | Submodule not fetched or wrong config path | Run `git submodule update --init benchmarks/alfworld` |

---

## Notes

- The script uses the `eval_out_of_distribution` split (`valid_unseen/`) by default.
- Game data files are downloaded to `~/.cache/alfworld/` and are not tracked in git.
- The MaskRCNN detector (~178 MB) is only needed for visual (THOR) mode, not for our text-based scripts.
