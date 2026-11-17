
# 🎬 Movie Facts and Fibs (MF²) — Github Repo

Guidelines and inference code for reproducing results.

## 🔗 Important Links

- 📄 **Paper:** [How Well Do Vision-Language Models Understand Movies? (MF²)](https://aclanthology.org/2026.emnlp-main.1702/)
- 🤗 **Dataset (Hugging Face):** [sardinelab/MF2-movie-facts-and-fibs](https://huggingface.co/datasets/sardinelab/MF2-movie-facts-and-fibs)
- 🌐 **Website:** [mf2.io](https://manzar96.github.io/mf2.io/)
- 📚 **Cite us:** [Citation](#citation)



## 📦 Clone this repo

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:deep-spin/MF2-bench-how-well-vlms-understand-movies.git
cd MF2-bench-how-well-vlms-understand-movies
```

## 📥 Download the MF2 dataset

Download the full-length movies from [sardinelab/MF2-movie-facts-and-fibs](https://huggingface.co/datasets/sardinelab/MF2-movie-facts-and-fibs) into `data/MF2`:

```bash
git lfs install
(cd data && git clone https://huggingface.co/datasets/sardinelab/MF2-movie-facts-and-fibs MF2)
```

Each inference script sets `VIDEO_BASE_PATH` itself to the directory that contains `<movie_id>.mp4`. Claims (`data/MF2_test.csv`), transcripts (`data/transcripts/`), and synopsis files (`data/synopsis/`) are already in this repo.

> 💡 **Cropped-movie experiment.** Optional single-scene clips with the full transcript. See [`src_cropped_experiment/README.md`](src_cropped_experiment/README.md).


## ⚙️ Setup Instructions



### 1. 🐍 Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv mf2-env
source mf2-env/bin/activate
```



### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

This is the default (transformers) stack used for most open-weight models (Gemma, InternVL, Qwen2.5-VL, LLaVA-Video, etc.).

> ⚠️ **vLLM models.** For in-process vLLM inference (e.g. `Qwen/Qwen3-VL-30B-A3B-Instruct`), use a **separate** environment and install `requirements-vllm.txt` instead. That stack pins a different PyTorch version and will conflict with `requirements.txt`.

> 💡 **Per-model extras.** Some adapters need extra packages (for example `llava` for LLaVA-Video, `flash-attn` for Phi-4 / Qwen3-VL). You may need to install additional packages for a specific model, or use a dedicated venv if two models have conflicting dependencies.

Install`flash-attn`:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```



### 3. 🎞️ FFmpeg

Make sure ffmpeg is installed and the module is loaded.
If your system does not use modules, you can install FFmpeg via a package manager.

```bash
sudo apt install ffmpeg
module load ffmpeg
```



## 🚀 Running an Open-Weight Model

Edit the variables in `bash_scripts/run_open_inference_gemma.sh` (venv path, movie directory, output dir, movie IDs, modality, frames), then:

```bash
bash bash_scripts/run_open_inference_gemma.sh
```

That script loops over movie IDs and calls `run_open_vlm.py`. The important knobs are:

- **model**: Hugging Face id as listed in `models/open_source/registry.py` (example: `google/gemma-3-27b-it`)
- **output_dir**: where per-movie JSON results are written (`<output_dir>/<model>/<movie_id>-results.json`)
- **modality**:
  - `video_only`
  - `transcripts_only`
  - `video_and_transcripts`
  - `statement_only`
  - `synopsis_only`
- **user_prompt_template_name**: `explanation_free`(recommended for closed-source models), `direct_free` (*recommended for open-weight models)*
- **max_frames**: number of frames sampled from each movie

Equivalent one-movie call:

```bash
python ./run_open_vlm.py \
    --model google/gemma-3-27b-it \
    --movie_id 1 \
    --mode independent \
    --data_path data/MF2_test.csv \
    --user_prompt_template_name direct_free \
    --system_prompt_template_name default \
    --modality video_and_transcripts \
    --output_dir <output_dir> \
    --video_path data/MF2/test/1.mp4 \
    --transcripts_path data/transcripts/1.srt \
    --synopsis_path data/synopsis/1.synopsis \
    --max_frames 180
```

Other example launchers live under `bash_scripts/`.


## 📊 Evaluation

After inference, `run_open_vlm.py` writes one JSON file per movie under a model-specific subdirectory:

```
<output_dir>/<model_with_slashes_replaced>/<movie_id>-results.json
```

For example, with `google/gemma-3-27b-it` the files land in:

```
<output_dir>/google__gemma-3-27b-it/
```

Parse those raw outputs and print aggregate metrics with:

```bash
python parse_model_outputs.py \
    --output_dir <output_dir> \
    --strategy last-occurrence
```



##### Arguments

- `<output_dir>`: directory containing `*-results.json` for a single model run (the inner model folder, not the top-level results root).

The script logs **overall metrics** to stdout, including:

- `pairwise_accuracy` — both claims correct (TRUE on true claim, FALSE on false claim) - primary metric
- `true_positive_rate` / `true_negative_rate`
- `accuracy` — per-side accuracy across all claim outputs
- `none_rate` — fraction of outputs that could not be parsed



## 📧 Contact

For questions or support, contact [emmanouil.zaranis@tecnico.ulisboa.pt](mailto:emmanouil.zaranis@tecnico.ulisboa.pt) or [manoszar96@gmail.com](mailto:manoszar96@gmail.com).

## 📜 License

CC-BY 4.0 license.

## Citation

```bibtex
@inproceedings{zaranis-etal-2026-well,
    title = "How Well Do Vision-Language Models Understand Movies?",
    author = "Zaranis, Emmanouil  and
      Farinhas, Ant{\'o}nio  and
      Santos, Saul  and
      Canaverde, Beatriz  and
      Ramos, Miguel Moura  and
      Mohammed, Wafaa  and
      Attanasio, Giuseppe  and
      Zerva, Chrysoula  and
      Sivakumaran, Nithin  and
      Yu, Shoubin  and
      Bueno-Benito, Elena  and
      Surikuchi, Aditya K  and
      Peters, Ben  and
      Sanchez Villegas, Danae  and
      Viveiros, Andre G.  and
      Vasylenko, Pavlo  and
      Liao, Baohao  and
      Sannigrahi, Sonal  and
      Yoon, Jaehong  and
      Stengel-Eskin, Elias  and
      Dimiccoli, Mariella  and
      Lanz, Oswald  and
      Suglia, Alessandro  and
      Bansal, Mohit  and
      Pezzelle, Sandro  and
      Frank, Stella  and
      Niculae, Vlad  and
      Elliott, Desmond  and
      Bernardi, Raffaella  and
      Fern{\'a}ndez, Raquel  and
      Martins, Andre",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1702/",
    doi = "10.18653/2026.emnlp-main.1702",
    pages = "38762--38801",
    ISBN = "979-8-89176-459-0",
    abstract = "Vision-language models (VLMs) can increasingly handle extended contexts and answer questions about long videos{---}but can they understand the narrative structure of movies as humans do? To investigate this question, we introduce ${MF}^2$ ($\textit{Movie Facts and Fibs}$), a fully manually curated benchmark designed to evaluate whether models can identify, recall, and comprehend information central to the movie{'}s storyline, using both the visual and textual modalities. ${MF}^2$ includes over 50 $\textit{open-licensed}$ movies, each paired with manually constructed sets of claim pairs: one true claim ($\textit{fact}$) and one plausible but false ($\textit{fib}$), totaling over 850 pairs. Our analysis across 20 open-source and 2 proprietary models reveals a substantial gap relative to human performance. While subtitles provide a strong signal for narrative understanding, other textual sources (e.g., synopses) are less informative. In contrast, when visual information is distributed across the entire movie and jointly embedded with subtitles, models struggle to identify and effectively use the relevant content. However, when the visual input is restricted to claim-relevant visual content, models improve significantly, indicating substantial room for improvement in processing large multimodal contexts and distinguishing relevant from irrelevant information. Overall, our findings highlight persistent challenges in multimodal narrative understanding for current VLMs. We release our data and code to support further research in this direction."
}
```