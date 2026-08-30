from transformers import AutoModel
from tqdm import tqdm
from loguru import logger

MAX_FRAMES = 180
DEFAULT_FPS = 1.0

_MODEL = None


def load_longvila_model(args, max_frames=None, fps=None):
    global _MODEL
    if _MODEL is None:
        _MODEL = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
    if max_frames is not None:
        _MODEL.config.num_video_frames = max_frames
        _MODEL.config.fps = 0
    elif fps is not None:
        _MODEL.config.fps = fps
    return _MODEL


def longvila_inference(
    args,
    input_prompts,
    video_path=None,
    system_prompt=None,
    shuffle_frames=False,
    max_frames=None,
):
    if max_frames is None:
        max_frames = MAX_FRAMES
    if shuffle_frames:
        logger.warning("LongVILA does not support shuffle_frames; frames are sampled by the model.")

    model = load_longvila_model(args, max_frames=max_frames if video_path else None)

    responses = []
    for input_prompt in tqdm(input_prompts):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{input_prompt}"
        else:
            prompt = input_prompt
        if video_path:
            content = [prompt, {"path": video_path}]
        else:
            content = [prompt]
        response = model.generate_content(content)
        if isinstance(response, str):
            responses.append(response.strip())
        else:
            responses.append(str(response).strip())
    return responses
