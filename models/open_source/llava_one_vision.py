from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import torch

MAX_FRAMES = 32

_MODEL = None
_PROCESSOR = None


def load_llavaonevision_model(args):
    global _MODEL, _PROCESSOR

    if _MODEL is None:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        _PROCESSOR = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True
        )

    return _MODEL, _PROCESSOR


def _move_to_device(inputs, model):
    """
    Safe move for device_map='auto' models.
    We do NOT use model.device.
    """
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda" if torch.cuda.is_available() else "cpu")
    return inputs


def llavaonevision_inference(
    args,
    input_prompts,
    video_path=None,
    system_prompt=None,
    shuffle_frames=False,
    max_frames=None
):
    if max_frames is None:
        max_frames = MAX_FRAMES

    model, processor = load_llavaonevision_model(args)

    responses = []

    for input_prompt in tqdm(input_prompts):

        # -------------------------
        # Build conversation
        # -------------------------
        conversation = []

        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            })

        user_content = []

        if video_path is not None:
            user_content.append({"type": "video", "path": video_path})

        user_content.append({"type": "text", "text": input_prompt})

        conversation.append({
            "role": "user",
            "content": user_content
        })

        # -------------------------
        # Processor step
        # -------------------------
        if video_path is not None:
            inputs = processor.apply_chat_template(
                conversation,
                num_frames=max_frames,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

        # -------------------------
        # Move tensors safely
        # -------------------------
        inputs = _move_to_device(inputs, model)

        # -------------------------
        # Generate
        # -------------------------
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        # -------------------------
        # Decode
        # -------------------------
        response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        responses.append(response)

    return responses