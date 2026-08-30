#!/usr/bin/env python3
import argparse
import os
import json
import base64
import cv2
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from loguru import logger
import re

from models.closed_source.registry import MODELS_MAPPING as MODELS_MAPPING_CLOSED_SOURCE
from templates.prompt_utils import build_prompts, build_prompts_multiple_choice

from utils import validate_paths_for_mode, load_transcripts, load_synopsis


def parse_args():
    p = argparse.ArgumentParser(description="Run LiteLLM on cropped per-claim segments")
    p.add_argument("--model",required=True, help="Model to be used. Check models.closed_source.registry.py for available models.")
    p.add_argument("--mode",default="independent", choices=["independent", "multiple_choice"], help="Mode to be used.")
    p.add_argument("--video_path", default=None, help="Path to the video file.")
    p.add_argument("--transcripts_path", default=None, help="Path to the transcripts file.")
    p.add_argument("--synopsis_path", default=None, help="Path to the synopsis file.")
    p.add_argument("--movie_id", type=int, required=True, help="Movie ID to filter.")
    p.add_argument("--data_path", required=True, help="Path to the dataset CSV file.")
    p.add_argument("--movie_id_column", default="movie_id", help="Column name for movie ID.")
    p.add_argument("--true_claim_column", default="true_claim", help="Column name for true claims.")
    p.add_argument("--false_claim_column", default="false_claim", help="Column name for false claims.")
    p.add_argument("--user_prompt_template_name", default="default", help="Template to be used for the prompt.")
    p.add_argument("--system_prompt_template_name", default="default", help="Template to be used for system prompt.")
    p.add_argument("--modality", required=True, choices=["video_only", "transcripts_only", "video_and_transcripts", "video_and_synopsis", "video_transcripts_and_synopsis", "statement_only" , "synopsis_only", "video_and_transcripts_and_audio"], help="Inference mode to be used.")
    p.add_argument("--output_dir", default="./results", help="Output directory.")
    p.add_argument("--cache_video_frames", action="store_true",default=False, help="Use cache for video frames.")

    p.add_argument("--model_provider", required=True, help="Model provider.")
    p.add_argument("--API_KEY", default="None", help="API key for LiteLLM. If 'None', environment variables are used.")
    p.add_argument("--api_base", default=None, help="Optional base URL (e.g., OpenRouter/local proxy)")
    
    p.add_argument("--num_frames", type=int, default=None, help="Frames to sample per video")

    return p.parse_args()


def model_inference(model, api_key, api_base, model_provider, input_prompts, video_path=None, system_prompt=None, num_frames=None, cache_video_frames=True):
    # model mapping to model_registry name
    if model in MODELS_MAPPING_CLOSED_SOURCE:
        inference_fn = MODELS_MAPPING_CLOSED_SOURCE[model]
    else:
        logger.error(f"Unknown model: {model}. Available models are: {MODELS_MAPPING_CLOSED_SOURCE.keys()}. Check models.closed_source.registry.py for available models.")
        raise ValueError(f"Unknown model: {model}. Available models are: {MODELS_MAPPING_CLOSED_SOURCE.keys()}. Check models.closed_source.registry.py for available models.")
    logger.info(f"Running inference for {model}.")

    # --------------------------------------------------
    # 👇 Optional caching block (ONLY when enabled)
    # --------------------------------------------------
    cached_image_parts = None
    if cache_video_frames and video_path:
        from models.closed_source.inference_new import sample_frames_to_b64, MAX_FRAMES
        if not num_frames:
            num_frames = MAX_FRAMES

        frames_b64 = sample_frames_to_b64(video_path, num_frames)

        if 'gemini' in model_provider:
            cached_image_parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    "cache_control": {"type": "ephemeral"},
                }
                for b64 in frames_b64
            ]
        else:
            cached_image_parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
                for b64 in frames_b64
            ]

    responses = []
    for input_prompt_text in tqdm(input_prompts, desc="Running inference for claims", total=len(input_prompts)):
        response = inference_fn(model_provider, api_key, api_base, input_prompt_text,
            video_path=video_path, 
            cached_image_parts=cached_image_parts,
            system_prompt=system_prompt,
            num_frames=num_frames,
        )
        responses.append(response)
    return responses




if __name__ == '__main__':
    args = parse_args()

    claims = pd.read_csv(args.data_path)

    output_dir = os.path.join(args.output_dir, args.model_provider.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True) 
    output_file = os.path.join(output_dir, f'{args.movie_id}-results.json')
    if os.path.exists(output_file):
        logger.info(f"Results file {output_file} already exists, skipping...")
        exit()

    logger.info(f"Loading claims for movie {args.movie_id}.")
    true_claims = claims[claims[args.movie_id_column] == args.movie_id][args.true_claim_column].dropna().tolist()
    false_claims = claims[claims[args.movie_id_column] == args.movie_id][args.false_claim_column].dropna().tolist()


    # Validate paths based on inference mode
    video_path, transcripts_path, synopsis_path = validate_paths_for_mode(args.modality, args.video_path, args.transcripts_path, args.synopsis_path)

    # load transcripts, synopsis if needed
    transcripts = load_transcripts(transcripts_path)
    synopsis = load_synopsis(synopsis_path)

    logger.info(f"Video path: {video_path}")
    logger.info(f"Transcripts path: {transcripts_path}")
    logger.info(f"Synopsis path: {synopsis_path}")

    # Build (text) prompts for true and false claims
    logger.info(f"Selected mode: {args.mode}")
    if args.mode == "independent":
        logger.info(f"Building prompts...")
        input_prompts_true,system_prompt_true = build_prompts(true_claims, args.user_prompt_template_name, args.system_prompt_template_name,video_path, transcripts, synopsis, movie_id=args.movie_id)
        input_prompts_false,system_prompt_false = build_prompts(false_claims, args.user_prompt_template_name, args.system_prompt_template_name,video_path, transcripts, synopsis, movie_id=args.movie_id) 
    
        # Show example prompt
        logger.info(f"Example user prompt used: {input_prompts_true[0]}")
        if system_prompt_true:
            logger.info(f"System prompt used: {system_prompt_true}")
        else:
            logger.info("Model's default system prompt used.")

         # Run inference
        responses = model_inference(args.model, args.API_KEY, args.api_base, args.model_provider, input_prompts_true+input_prompts_false, video_path, system_prompt=system_prompt_true, num_frames=args.num_frames, cache_video_frames=args.cache_video_frames)
        responses_true = responses[:len(input_prompts_true)]
        responses_false = responses[len(input_prompts_true):]
        
        result_data = {
            idx: {
                args.true_claim_column: responses_true[idx] if idx < len(responses_true) else "",
                args.false_claim_column: responses_false[idx] if idx < len(responses_false) else ""
            }
            for idx in range(max(len(responses_true), len(responses_false)))
        }

    elif args.mode == "multiple_choice":
        logger.info(f"Building prompts...")
        input_prompts,system_prompt,true_claim_letter = build_prompts_multiple_choice(true_claims,false_claims, args.user_prompt_template_name, args.system_prompt_template_name,video_path, transcripts, synopsis, movie_id=args.movie_id)
        responses = model_inference(args.model, args.API_KEY, args.api_base, args.model_provider, input_prompts, video_path, system_prompt=system_prompt, num_frames=args.num_frames, cache_video_frames=args.cache_video_frames)
        result_data = {
            idx: {
                "response_true_claim_target": true_claim_letter[idx] if idx < len(true_claim_letter) else "",
                "response": responses[idx] if idx < len(responses) else "",
            }
            for idx in range(len(responses))
        }
    
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Results saved to {output_file}.")