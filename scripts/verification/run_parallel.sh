#!/usr/bin/env bash
#
# Launch verification runs (non-_ya / OpenRouter scripts) in parallel,
# each in its own detached `screen` session.
#
# Every combination of BENCHMARKS x METHODS x BACKBONES is started as one
# screen session. For proposed_v4 on *_clarification benchmarks, one session
# per threshold in THRESHOLDS is started instead (the t-ablation sweep).
#
# Usage:
#   scripts/verification/run_parallel.sh                 # use defaults below
#   BENCHMARKS="webshop alfworld" METHODS="react uam" \
#     BACKBONES="z-ai/glm-4.7" scripts/verification/run_parallel.sh
#   DRY_RUN=1 scripts/verification/run_parallel.sh       # print, don't launch
#
# By default every session runs at DEBUG log level. Each verification script
# does its own logging to stdout; nothing is captured to disk here, so attach
# to a live session to watch it.
#
# Then:
#   screen -ls                  # list running sessions
#   screen -r <session-name>    # attach to one
#
# ---------------------------------------------------------------------------
# Configuration (override any of these via environment variables)
# ---------------------------------------------------------------------------

# Which benchmarks to run. Valid:
#   alfworld webshop real alfworld_clarification webshop_clarification
BENCHMARKS="${BENCHMARKS:-alfworld webshop real alfworld_clarification webshop_clarification}"

# Which methods/agents. Valid: react uam proposed_v4
# (clarification benchmarks only support these three; proposed_v4 there also
#  runs the threshold sweep.)
METHODS="${METHODS:-react uam proposed_v4}"

# Which backbones (OpenRouter model ids, WITHOUT the leading "openrouter/").
BACKBONES="${BACKBONES:-z-ai/glm-4.7 deepseek/deepseek-v3.2-exp qwen/qwen3.5-35b-a3b openai/gpt-oss-120b openai/gpt-5.1}"

# Thresholds for proposed_v4 on *_clarification benchmarks.
THRESHOLDS="${THRESHOLDS:-0.25 0.5 0.75}"

# Optional OpenRouter provider(s) to pin for EVERY run, overriding the per-backbone
# defaults below. Comma-separated for an ordered allowlist (e.g. "DeepInfra,Parasail").
# Empty (default) uses the per-backbone defaults from default_providers(); a backbone
# with no default and no override auto-routes. Pinning disables provider fallbacks.
PROVIDER="${PROVIDER:-}"

# Only launch a (benchmark, method, backbone[, threshold]) combination when its
# current count of error-free completed tasks is BELOW this number; skip the rest.
# Use to top up under-completed runs without touching finished ones. Empty = run all.
ONLY_BELOW="${ONLY_BELOW:-}"

# Space-list of backbones for which to DISABLE the OpenRouter reasoning channel.
# Some thinking models (e.g. qwen) route their pre-action reasoning into a
# separate `reasoning` field and then OMIT required output tags (<u_request>,
# <confidence>) from `content`. Disabling the channel makes them emit the full
# tagged sequence inline. Matching runs get OPENROUTER_DISABLE_REASONING=1, which
# the agents read in _query_model. Empty = never disable.
DISABLE_REASONING_BACKBONES="${DISABLE_REASONING_BACKBONES:-qwen/qwen3.5-35b-a3b}"

# Common args passed to every script.
N_TASKS="${N_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-25}"
RANDOM_SEED="${RANDOM_SEED:-42}"
# Default to DEBUG so each session's log captures full per-step agent detail.
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Extra args appended verbatim to every invocation (e.g. EXTRA_ARGS="--no-shuffle_tasks").
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Global screenshot override for the `real` benchmark only (the only one that
# sends an image; the flag exists solely on the real runners). Values: on | off
# | empty. When set, it forces that value for EVERY real run, overriding the
# per-backbone auto behavior below. Empty (default) = auto (see NON_MULTIMODAL_BACKBONES).
REAL_SCREENSHOT="${REAL_SCREENSHOT:-}"

# Backbones that are NOT multimodal and 404 on image input. On the `real`
# benchmark these auto-run axtree-only (--no-use_screenshot); multimodal
# backbones (e.g. gpt-5.1) keep screenshots. This per-model automation lets a
# single run mix multimodal and non-multimodal backbones. Ignored when
# REAL_SCREENSHOT is set (that global override wins). Space-list of model ids.
NON_MULTIMODAL_BACKBONES="${NON_MULTIMODAL_BACKBONES:-deepseek/deepseek-v3.2-exp openai/gpt-oss-120b z-ai/glm-4.7}"

# How to invoke python. Run from the repo root.
RUNNER="${RUNNER:-uv run --group scripts python -m}"

# If set to 1, print the commands instead of launching screen sessions.
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------
set -euo pipefail

usage() {
    cat <<EOF
Launch verification runs (non-_ya / OpenRouter scripts) in parallel detached
screen sessions: one per BENCHMARK x METHOD x BACKBONE. proposed_v4 on a
*_clarification benchmark expands into one session per THRESHOLD.

Usage:
  scripts/verification/run_parallel.sh [-h|--help|--list]
  VAR=value [VAR=value ...] scripts/verification/run_parallel.sh

Configurable parameters (set as environment variables):

  Parameter      Valid values / format                       Current value
  -------------  ------------------------------------------  -------------
  BENCHMARKS     space-list: alfworld webshop real
                 alfworld_clarification webshop_clarification
                                                             ${BENCHMARKS}
  METHODS        space-list: react uam proposed_v4           ${METHODS}
  BACKBONES      space-list of OpenRouter model ids
                 (without the leading "openrouter/")         ${BACKBONES}
  THRESHOLDS     space-list of floats; only used by
                 proposed_v4 on *_clarification benchmarks   ${THRESHOLDS}
  PROVIDER       comma-list of OpenRouter provider slugs to
                 pin for EVERY run, overriding per-backbone
                 defaults; empty = use default_providers()   ${PROVIDER:-(per-backbone defaults)}
  ONLY_BELOW     integer N — only launch combinations whose
                 error-free task count is < N; empty = all   ${ONLY_BELOW:-(none)}
  DISABLE_REASONING_BACKBONES
                 space-list of backbones to run with the
                 OpenRouter reasoning channel disabled       ${DISABLE_REASONING_BACKBONES:-(none)}
  N_TASKS        integer                                     ${N_TASKS}
  MAX_STEPS      integer                                     ${MAX_STEPS}
  RANDOM_SEED    integer                                     ${RANDOM_SEED}
  LOG_LEVEL      DEBUG|INFO|WARNING|ERROR|CRITICAL           ${LOG_LEVEL}
  EXTRA_ARGS     extra CLI args appended to every run        ${EXTRA_ARGS:-(none)}
  REAL_SCREENSHOT on|off — global screenshot override for
                 `real`; empty = per-backbone auto           ${REAL_SCREENSHOT:-(auto)}
  NON_MULTIMODAL_BACKBONES
                 space-list of backbones auto-run axtree-only
                 on `real` (no screenshot)                   ${NON_MULTIMODAL_BACKBONES:-(none)}
  RUNNER         python launcher prefix                      ${RUNNER}
  DRY_RUN        0|1 — 1 prints commands, launches nothing   ${DRY_RUN}

Examples:
  DRY_RUN=1 scripts/verification/run_parallel.sh
  BENCHMARKS="webshop alfworld" METHODS="react uam" \\
    BACKBONES="z-ai/glm-4.7" scripts/verification/run_parallel.sh
  ONLY_BELOW=70 scripts/verification/run_parallel.sh   # top up runs with <70 done
  PROVIDER="DeepInfra,Parasail" scripts/verification/run_parallel.sh  # override pins

Manage sessions:
  screen -ls                  # list running sessions
  screen -r <session-name>    # attach to one
EOF
}

case "${1:-}" in
    -h|--help|--list)
        usage
        exit 0
        ;;
    "")
        ;;
    *)
        echo "ERROR: unknown argument: $1" >&2
        echo "Run with --help to see configurable parameters." >&2
        exit 2
        ;;
esac

# cd to repo root (two levels up from this script: scripts/verification/ -> repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v screen >/dev/null 2>&1; then
    echo "ERROR: 'screen' is not installed. Install it (e.g. 'brew install screen')." >&2
    exit 1
fi

# Validate the global screenshot override up front (the per-model logic in the
# loop assumes REAL_SCREENSHOT is one of these or empty).
case "${REAL_SCREENSHOT}" in
    ""|on|true|1|off|false|0) ;;
    *) echo "ERROR: REAL_SCREENSHOT must be on|off (got '${REAL_SCREENSHOT}')." >&2; exit 1 ;;
esac

# model id -> filesystem-safe tag (mirrors the scripts' own model_dir scheme)
model_tag() {
    echo "$1" | sed 's#^openrouter/##; s#^openai/##; s#/#--#g'
}

# Default allowed OpenRouter provider(s) per backbone (comma-separated, in order),
# mirroring docs/allowed_openrouter_provider.md. Slugs are the OpenRouter routing
# names (e.g. "Novita", not the "NovitaAI" display name). Overridden by PROVIDER.
default_providers() {
    case "$1" in
        openai/gpt-5.1)              echo "OpenAI" ;;
        deepseek/deepseek-v3.2-exp)  echo "Novita,SiliconFlow,AtlasCloud" ;;
        z-ai/glm-4.7)                echo "Parasail,SiliconFlow" ;;
        openai/gpt-oss-120b)         echo "DeepInfra" ;;
        qwen/qwen3.5-35b-a3b)        echo "DeepInfra,Parasail" ;;
        *)                           echo "" ;;
    esac
}

# Count error-free completed tasks in a results_dir (mirrors count_completed.py):
# tasks/ benches -> number of task_*.json; real -> runs/*/summary_info.json with
# no err_msg / stack_trace. Missing dir -> 0.
done_count() {
    local rdir="$1"
    if [[ -d "${rdir}/tasks" ]]; then
        find "${rdir}/tasks" -maxdepth 1 -name 'task_*.json' 2>/dev/null | wc -l | tr -d ' '
    elif [[ -d "${rdir}/runs" ]]; then
        python3 - "${rdir}" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/runs/*/summary_info.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("err_msg") is None and d.get("stack_trace") is None:
        n += 1
print(n)
PY
    else
        echo 0
    fi
}

launched=0
skipped=0

launch() {
    local session="$1"; shift
    local cmd="$*"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] ${session}"
        echo "          ${cmd}"
        return
    fi

    if screen -ls 2>/dev/null | grep -q "[.]${session}[[:space:]]"; then
        echo "[skip] session already exists: ${session}"
        skipped=$((skipped + 1))
        return
    fi

    # -dmS: detached, named session. Wrap in bash -lc so the session stays
    # until the command finishes. Output stays inside the screen session.
    screen -dmS "${session}" bash -lc "cd '${REPO_ROOT}' && ${cmd}"
    echo "[start] ${session}"
    launched=$((launched + 1))
}

# Skip a run when ONLY_BELOW is set and its results_dir already has >= ONLY_BELOW
# error-free completed tasks; otherwise launch it.
launch_or_skip() {
    local session="$1" rdir="$2" cmd="$3"
    if [[ -n "${ONLY_BELOW}" ]]; then
        local d; d="$(done_count "${rdir}")"
        if (( d >= ONLY_BELOW )); then
            echo "[skip-complete] ${session} (done=${d} >= ${ONLY_BELOW})"
            skipped=$((skipped + 1))
            return
        fi
        echo "[below] ${session} (done=${d} < ${ONLY_BELOW})"
    fi
    launch "${session}" "${cmd}"
}

for bench in ${BENCHMARKS}; do
    for method in ${METHODS}; do
        module="scripts.verification.${bench}.${method}.${method}"
        is_clar=0
        [[ "${bench}" == *_clarification ]] && is_clar=1

        for model in ${BACKBONES}; do
            tag="$(model_tag "${model}")"

            # PROVIDER overrides everything; otherwise use the per-backbone default.
            providers="${PROVIDER:-$(default_providers "${model}")}"
            provider_arg=""
            [[ -n "${providers}" ]] && provider_arg="--provider ${providers}"

            # Screenshots exist only on the `real` runners. REAL_SCREENSHOT (if set)
            # is a global override; otherwise auto-disable screenshots for
            # non-multimodal backbones so they run axtree-only (avoids the
            # "No endpoint ... image input" 404s). Multimodal backbones keep the
            # runner default (screenshots on).
            screenshot_arg=""
            if [[ "${bench}" == "real" ]]; then
                if [[ -n "${REAL_SCREENSHOT}" ]]; then
                    case "${REAL_SCREENSHOT}" in
                        on|true|1)   screenshot_arg="--use_screenshot" ;;
                        off|false|0) screenshot_arg="--no-use_screenshot" ;;
                    esac
                else
                    for nmb in ${NON_MULTIMODAL_BACKBONES}; do
                        [[ "${model}" == "${nmb}" ]] && screenshot_arg="--no-use_screenshot"
                    done
                fi
            fi

            base_args="--model openrouter/${model} --n_tasks ${N_TASKS} --max_steps ${MAX_STEPS} --random_seed ${RANDOM_SEED} --log_level ${LOG_LEVEL} ${provider_arg} ${screenshot_arg} ${EXTRA_ARGS}"

            # Disable the reasoning channel for configured backbones (see above).
            reasoning_env=""
            for drb in ${DISABLE_REASONING_BACKBONES}; do
                [[ "${model}" == "${drb}" ]] && reasoning_env="OPENROUTER_DISABLE_REASONING=1 "
            done

            if [[ "${is_clar}" == "1" && "${method}" == "proposed_v4" ]]; then
                # threshold sweep — one session per threshold, isolated results_dir
                for t in ${THRESHOLDS}; do
                    session="${bench}-${method}-${tag}-t${t}"
                    rdir="verification_results/${bench}/${method}/${tag}/t${t}"
                    cmd="${reasoning_env}${RUNNER} ${module} ${base_args} --clarification_threshold ${t} --results_dir ${rdir}"
                    launch_or_skip "${session}" "${rdir}" "${cmd}"
                done
            else
                session="${bench}-${method}-${tag}"
                rdir="verification_results/${bench}/${method}/${tag}"
                cmd="${reasoning_env}${RUNNER} ${module} ${base_args} --results_dir ${rdir}"
                launch_or_skip "${session}" "${rdir}" "${cmd}"
            fi
        done
    done
done

echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete (no sessions launched)."
else
    echo "Launched ${launched} session(s), skipped ${skipped}."
    echo "List:   screen -ls"
    echo "Attach: screen -r <session-name>"
fi
