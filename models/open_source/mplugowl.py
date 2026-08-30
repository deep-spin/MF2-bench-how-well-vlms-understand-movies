import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm
import numpy as np

MAX_NUM_FRAMES = 16  # default max frames per video

_MODEL = None
_TOKENIZER = None
_PROCESSOR = None


def encode_video(video_path, max_frames=None, shuffle_frames=False):
    """
    Extract frames from video and return as list of PIL Images.
    """
    from decord import VideoReader, cpu

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]

    if shuffle_frames:
        np.random.shuffle(frames)
    return frames


def load_mplug_owl3_model(args):
    global _MODEL, _TOKENIZER, _PROCESSOR
    if _MODEL is None:
        _MODEL = AutoModel.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            torch_dtype=torch.half,
            trust_remote_code=True,
        ).eval().cuda()
        _TOKENIZER = AutoTokenizer.from_pretrained(args.model)
        _PROCESSOR = _MODEL.init_processor(_TOKENIZER)
    return _MODEL, _TOKENIZER, _PROCESSOR


def mplug_owl3_inference(
    args,
    input_prompts,
    video_path=None,
    system_prompt=None,
    shuffle_frames=False,
    max_frames=None
):
    """
    Inference function for mPLUG-Owl3 (video-text model) following GEMMA3 pattern.

    Args:
        args: Object with args.model (model path or ID)
        input_prompts: List of user prompts (text)
        video_paths: Optional list of video paths
        system_prompt: Optional system prompt text
        shuffle_frames: Whether to shuffle video frames
    Returns:
        List of generated text responses
    """
    if max_frames is None:
        max_frames = MAX_NUM_FRAMES

    model, tokenizer, processor = load_mplug_owl3_model(args)
    device = "cuda"

    video_frames_list = None
    if video_path:
        video_frames_list = encode_video(video_path, max_frames=max_frames, shuffle_frames=shuffle_frames)

    responses = []
    for input_prompt in tqdm(input_prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": ""}
        if video_frames_list:
            user_message["content"] = "<|image|>" * len(video_frames_list) + input_prompt
        else:
            user_message["content"] = input_prompt
        messages.append(user_message)

        if video_frames_list:
            inputs = processor(messages, images=video_frames_list, videos=None)
        else:
            inputs = processor(messages, images=None, videos=None)

        inputs.to(device)
        inputs.update({
            "tokenizer": tokenizer,
            "max_new_tokens": 1024,
            "decode_text": True
        })

        with torch.inference_mode():
            output = model.generate(**inputs)
        responses.append(output[0])

    return responses
