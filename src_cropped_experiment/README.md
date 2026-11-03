# Cropped-movie evaluation (full transcripts)

Evaluate single-scene claims on time-cropped video clips together with the **full, untrimmed** transcript of that movie. Run the commands below from the repository root. Each launcher changes to the repo root itself.

## Data

Claims and transcripts come with a normal clone. Only `data/cropped_experiment_data/cropped_movies/` is stored with Git LFS. Pull that folder after cloning:

```bash
git lfs install
git lfs pull --include="data/cropped_experiment_data/cropped_movies/**"
```

```
data/cropped_experiment_data/
  cropped_movies_experiment_claims.csv
  cropped_movies/{movie_id}/{movie_id}_claim_{claim_id}.mp4
  transcripts_full/{movie_id}/{movie_id}_claim_{claim_id}.srt
```

The launchers already point at these paths:

- claims: `data/cropped_experiment_data/cropped_movies_experiment_claims.csv`
- clips: `data/cropped_experiment_data/cropped_movies`
- transcripts: `data/cropped_experiment_data/transcripts_full`

Movies in this split: 5, 9, 13, 15, 16, 17, 29, 30, 34, 37, 38, 39, 47, 54.

## Open-weight models

Set `VENV_PATH` and `MODELS` at the top of the script. The default run is `video_and_transcripts` with the `direct_free` prompt and 180 frames.

```bash
bash src_cropped_experiment/run_open_inference_cropped_movies_full_trans.sh
```

Results are written to:

```
<output_dir>/<model_name>/
```

`<model_name>` is the Hugging Face id with `/` replaced by `__`. For the default Gemma model that is `google__gemma-3-27b-it`.

## Closed-source models

Set `OPENROUTER_API_KEY` in `.env` or the environment. It uses the same claims, clips, and `transcripts_full` paths. `MODALITY` in the script is `video_only`; set it to `video_and_transcripts` to send the full transcript with the clip.

```bash
bash src_cropped_experiment/run_closed_inference_cropped_movies_full_trans.sh
```

Results are written to:

```
<output_dir>/<modality>/<prompt>/<model_name>/
```

## Parse metrics

Point `--output_dir_cropped` at the model directory that contains the `*-results.json` files.

```bash
python src_cropped_experiment/parse_model_outputs_cropped_movies.py \
  --output_dir_cropped <output_dir> \
  --strategy last-occurrence
```
