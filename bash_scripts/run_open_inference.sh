#!/bin/bash

# VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-transformers" # set your venv path here
VENV_PATH="/mnt/scratch-hades/manos/venvs/mf2-env" # set your venv path here
source "$VENV_PATH/bin/activate"

module load ffmpeg

MODELS=( google/gemma-3-27b-it ) # Qwen/Qwen2.5-VL-7B-Instruct 
BASE_OUTPUT_DIR=./results # set your output directory here

MODALITY=video_and_transcripts # video_only, video_and_transcripts, statement_only, synopsis_only, transcripts_only, ...
USER_PROMPT_TEMPLATE_NAME=direct_free
MODE=independent
MAX_FRAMES=64 # set frames to use for inference

MOVIE_IDS=(1 ) # set from 1-54 (53 movies in total, 41 movie id is not included in the dataset)

SYSTEM_PROMPT_TEMPLATE_NAME="default"
DATA_PATH="data/MF2_test.csv"
VIDEO_BASE_PATH="data/MF2/test" # set your video base path here (where the full-length movies are stored, after downloading the HF dataset)
TRANSCRIPTS_BASE_PATH="data/transcripts"
SYNOPSIS_BASE_PATH="data/synopsis"


echo "Running inference in movies: ${MOVIE_IDS[@]}"

for MODEL in "${MODELS[@]}"; do
    for MOVIE_ID in "${MOVIE_IDS[@]}"; do

        OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODALITY}/${USER_PROMPT_TEMPLATE_NAME}

        VIDEO_PATH="${VIDEO_BASE_PATH}/${MOVIE_ID}.mp4"
        TRANSCRIPTS_PATH="${TRANSCRIPTS_BASE_PATH}/${MOVIE_ID}.srt"
        SYNOPSIS_PATH="${SYNOPSIS_BASE_PATH}/${MOVIE_ID}.synopsis"

        python ./run_open_vlm.py \
            --model "$MODEL" \
            --movie_id "$MOVIE_ID" \
            --mode "$MODE" \
            --data_path "$DATA_PATH" \
            --user_prompt_template_name "$USER_PROMPT_TEMPLATE_NAME" \
            --system_prompt_template_name "$SYSTEM_PROMPT_TEMPLATE_NAME" \
            --modality "$MODALITY" \
            --output_dir "$OUTPUT_DIR" \
            --video_path "$VIDEO_PATH" \
            --transcripts_path "$TRANSCRIPTS_PATH" \
            --synopsis_path "$SYNOPSIS_PATH" \
            --max_frames "$MAX_FRAMES"
    done
done
