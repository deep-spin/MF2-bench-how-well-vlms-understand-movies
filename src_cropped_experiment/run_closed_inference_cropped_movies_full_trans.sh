#!/bin/bash
set -euo pipefail

# Closed-source inference on cropped clips + FULL (untrimmed) per-claim transcripts.
# Run from the private repo root (or anywhere — this script cds there).
# Expects OPENROUTER_API_KEY in .env or the environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src_cropped_experiment:${PYTHONPATH:-}"

VENV_PATH="${VENV_PATH:-/mnt/scratch-hades/manos/venvs/mf2-env}" # set your venv path here
source "$VENV_PATH/bin/activate"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi
API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in .env or the environment}"
API_BASE="${API_BASE:-https://openrouter.ai/api/v1}"

module load ffmpeg 2>/dev/null || true

MODEL_NAME="gemini-2.5-pro" # registry key in models/closed_source/registry.py
MODEL_PROVIDER="openrouter/google/gemini-2.5-pro"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results_closed_source_cropped_movies}"

MODALITY="video_only"  # video_only | transcripts_only | video_and_transcripts | statement_only
USER_PROMPT_TEMPLATE_NAME="explanation_free"
SYSTEM_PROMPT_TEMPLATE_NAME="default"
MODE="independent"
NUM_FRAMES=120

CROPPED_ROOT="${REPO_ROOT}/data/cropped_experiment_data"
VIDEOS_ROOT="${CROPPED_ROOT}/cropped_movies"
TRANSCRIPTS_ROOT="${CROPPED_ROOT}/transcripts_full"
SYNOPSIS_ROOT="${REPO_ROOT}/data/synopsis"
DATA_PATH="${CROPPED_ROOT}/cropped_movies_experiment_claims.csv"

MOVIE_IDS=(5 9 13 15 16 17 29 30 34 37 38 39 47 54)

echo "Running $MODEL_NAME on movies: ${MOVIE_IDS[*]} (modality=$MODALITY, transcripts=full)"

for MOVIE_ID in "${MOVIE_IDS[@]}"; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}"

  python "${REPO_ROOT}/src_cropped_experiment/run_closed_source_cropped_movies.py" \
    --model_name "$MODEL_NAME" \
    --mode "$MODE" \
    --modality "$MODALITY" \
    --movie_id "$MOVIE_ID" \
    --data_path "$DATA_PATH" \
    --videos_root "$VIDEOS_ROOT" \
    --transcripts_root "$TRANSCRIPTS_ROOT" \
    --synopsis_root "$SYNOPSIS_ROOT" \
    --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
    --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --API_KEY "$API_KEY" \
    --api_base "$API_BASE" \
    --model_provider "$MODEL_PROVIDER" \
    --num_frames "$NUM_FRAMES"
done
