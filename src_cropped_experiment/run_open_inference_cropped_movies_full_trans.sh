#!/bin/bash
set -euo pipefail

# Inference on cropped movies + FULL transcript (See Paper Table 2)
# Run from the repo root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src_cropped_experiment:${PYTHONPATH:-}"



VENV_PATH="/mnt/scratch-hades/manos/venvs/mf2-env" # set your venv path here
source "$VENV_PATH/bin/activate"

module load ffmpeg 2>/dev/null || true

MODELS=(
  "google/gemma-3-27b-it"
)

BASE_OUTPUT_DIR="./results_open_source_cropped_movies" # set your output directory here

MODALITY="video_and_transcripts"  # video_only | transcripts_only | video_and_transcripts | statement_only
USER_PROMPT_TEMPLATE_NAME="direct_free"
SYSTEM_PROMPT_TEMPLATE_NAME="default"
MODE="independent"
MAX_FRAMES=180  # leave empty to use each adapter's default

# Cropped assets (per-claim files under {root}/{movie_id}/{movie_id}_claim_{claim_id}.ext)
CROPPED_ROOT="${REPO_ROOT}/data/cropped_experiment_data" # root of cropped data
VIDEOS_ROOT="${CROPPED_ROOT}/cropped_movies" # directory with cropped movie files.
TRANSCRIPTS_ROOT="${CROPPED_ROOT}/transcripts_full"  # full transcript copied per claim
SYNOPSIS_ROOT="None"
DATA_PATH="${CROPPED_ROOT}/cropped_movies_experiment_claims.csv" # csv file with the claims to be processed.

MOVIE_IDS=(5 9 13 15 16 17 29 30 34 37 38 39 47 54)

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_FRAMES}" ]]; then
  MAX_FRAMES_ARGS=(--max_frames "$MAX_FRAMES")
fi

for MODEL in "${MODELS[@]}"; do
  echo "Running $MODEL on movies: ${MOVIE_IDS[*]} (modality=$MODALITY, transcripts=full)"

  for MOVIE_ID in "${MOVIE_IDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}"

    python "${REPO_ROOT}/src_cropped_experiment/run_open_source_cropped_movies.py" \
      --model "$MODEL" \
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
      "${MAX_FRAMES_ARGS[@]}"
      # --host "hades" \
      # --port "8003"
  done
done
