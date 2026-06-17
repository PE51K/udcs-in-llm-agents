# WebShop Environment Setup

Setup guide for the WebShop benchmark used in verification scripts. Covers Arch Linux and Ubuntu.

## Prerequisites

### Java (required by the Lucene search engine)

**Arch Linux:**
```bash
sudo pacman -S jdk-openjdk
```

**Ubuntu:**
```bash
sudo apt install default-jdk
```

Verify: `java -version` (11+ required)

### spaCy model (required by WebShop engine)

```bash
uv run --group scripts python -m spacy download en_core_web_sm
uv run --group scripts python -m spacy download en_core_web_lg
```

> All other dependencies (`pyserini`, `gdown`, `gym`, `flask`, etc.) are declared in `pyproject.toml` under the `scripts` group and installed automatically via `uv`.

---

## Setup Steps

Run all commands from the **project root** (`uncertainty-aware-web-agents/`).

### 1. Download data files

```bash
mkdir -p benchmarks/WebShop/data
cd benchmarks/WebShop/data

uv run --group scripts gdown "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"  # items_shuffle_1000.json
uv run --group scripts gdown "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"  # items_ins_v2_1000.json
uv run --group scripts gdown "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"  # items_human_ins.json

cd ../../..
```

Expected result:
```
benchmarks/WebShop/data/
├── items_shuffle_1000.json   (~4.5 MB) — product info
├── items_ins_v2_1000.json    (~147 KB) — product attributes
└── items_human_ins.json      (~5.1 MB) — human instructions
```

### 2. Build the search engine index

```bash
cd benchmarks/WebShop/search_engine
mkdir -p resources resources_100 resources_1k resources_100k

uv run --group scripts python convert_product_file_format.py
mkdir -p indexes
uv run --group scripts bash run_indexing.sh

cd ../../..
```

Expected output: `Indexing Complete! 1,000 documents indexed` (×4, no errors).

---

## Verify

Configure `.env` at the project root first (see [verification/README.md](../../scripts/verification/README.md) — `BASE_URL`, `OPENAI_API_KEY`, `MODEL`), then:

```bash
uv run --group scripts python -m \
    scripts.verification.webshop.react.react \
    --n_tasks 1 --log_level DEBUG --random_seed 42
```

A successful run loads the environment, runs one task, and prints step confidences and a ROC-AUC score.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: items_shuffle_1000.json` | Data not downloaded | Run Step 1 |
| `java: command not found` | Java not installed | Install JDK (see Prerequisites) |
| `OSError` in pyserini indexing | Missing index directories | Run `mkdir -p indexes` before `run_indexing.sh` |
| `[E050] Can't find model 'en_core_web_sm'` | spaCy model missing | Run `uv run --group scripts python -m spacy download en_core_web_sm` |
| `[E050] Can't find model 'en_core_web_lg'` | spaCy model missing | Run `uv run --group scripts python -m spacy download en_core_web_lg` |

---

## Notes

- This guide sets up the **small (1,000-product) dataset** used by default (`items_shuffle_1000.json`). The full 1.18M-product dataset is not needed for running scripts in this project.
- Data files are gitignored and must be downloaded locally on each machine.
- The `run_indexing.sh` script builds four indexes (`indexes`, `indexes_100`, `indexes_1k`, `indexes_100k`) — all four are needed.
