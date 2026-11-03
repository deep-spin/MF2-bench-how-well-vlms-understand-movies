#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Set the venv path as a variable
VENV_PATH="/mnt/scratch-hades/manos/venvs/mf2-env" # set your venv path here
source "$VENV_PATH/bin/activate"

# Load API key from .env
export $(grep -v '^#' .env | xargs)
API_KEY="..." # set your openrouter api key here
API_BASE=https://openrouter.ai/api/v1 # open-router base url

module load ffmpeg

# Registry key (models/closed_source/registry.py)
# MODEL=claude-3.7-sonnet
# MODEL_PROVIDER=openrouter/anthropic/claude-3.7-sonnet
MODEL=gemini-2.5-pro
MODEL_PROVIDER=openrouter/google/gemini-2.5-pro-preview
# MODEL=gpt-5
# MODEL_PROVIDER=openrouter/openai/gpt-5

BASE_OUTPUT_DIR=./results # set your output directory here
MODALITY=video_only
USER_PROMPT_TEMPLATE_NAME=explanation_free
MODE=independent
NUM_FRAMES=120
CACHE_VIDEO_FRAMES=True # whether to cache the video frames or not

MOVIE_IDS=(1 2 )
# MOVIE_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 42 43 44 45 46 47 48 49 50 51 52 53 54)

SYSTEM_PROMPT_TEMPLATE_NAME="default"
DATA_PATH="${REPO_ROOT}/data/MF2_test.csv"
VIDEO_BASE_PATH="${REPO_ROOT}/data/MF2/test" # directory with <movie_id>.mp4 files from the HF dataset
TRANSCRIPTS_BASE_PATH="${REPO_ROOT}/data/transcripts"
SYNOPSIS_BASE_PATH="${REPO_ROOT}/data/synopsis"

CACHE_VIDEO_FRAMES_FLAG=""
if [ "$CACHE_VIDEO_FRAMES" = True ]; then
    CACHE_VIDEO_FRAMES_FLAG="--cache_video_frames"
fi

echo "Running inference in movies: ${MOVIE_IDS[@]}"
echo "Running inference for model: $MODEL (provider: $MODEL_PROVIDER) with modality: $MODALITY"

for MOVIE_ID in "${MOVIE_IDS[@]}"; do

    OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}

    VIDEO_PATH="${VIDEO_BASE_PATH}/${MOVIE_ID}.mp4"
    TRANSCRIPTS_PATH="${TRANSCRIPTS_BASE_PATH}/${MOVIE_ID}.srt"
    SYNOPSIS_PATH="${SYNOPSIS_BASE_PATH}/${MOVIE_ID}.synopsis"

    python ./run_closed_source.py \
        --model "$MODEL" \
        --mode "$MODE" \
        --movie_id "$MOVIE_ID" \
        --data_path "$DATA_PATH" \
        --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
        --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
        --modality "$MODALITY" \
        --output_dir "$OUTPUT_DIR" \
        --video_path "$VIDEO_PATH" \
        --transcripts_path "$TRANSCRIPTS_PATH" \
        --synopsis_path "$SYNOPSIS_PATH" \
        --model_provider "$MODEL_PROVIDER" \
        --API_KEY "$API_KEY" \
        --api_base "$API_BASE" \
        --num_frames "$NUM_FRAMES" \
        $CACHE_VIDEO_FRAMES_FLAG
done
