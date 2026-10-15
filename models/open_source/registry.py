from models.open_source.vllm_served_model import inference_vllm_served_model
from models.open_source.cogvlm2 import cogvlm2_inference
from models.open_source.kangaroo import kangaroo_inference
from models.open_source.minicpm import minicpm_inference
from models.open_source.phi4 import phi4_inference_fast
from models.open_source.llava_one_vision import llavaonevision_inference
from models.open_source.videollama3 import videollama3_inference
from models.open_source.aria import aria_inference
from models.open_source.molmo2 import molmo2_inference_fast
from models.open_source.mplugowl import mplug_owl3_inference
from models.open_source.mimovl import mimovl_inference
from models.open_source.longvila import longvila_inference
from models.open_source.llavavideo import llavavideo_inference
from models.open_source.ovis import ovis_inference
from models.open_source.internvl3_global import internvl3_inference
from models.open_source.gemma3 import gemma3_inference
from models.open_source.qwenvl import qwenvl_inference
from models.open_source.qwenvl3 import qwenvl3_inference_vllm


MODELS_MAPPING = {
    # Qwen models
    "Qwen/Qwen2.5-VL-7B-Instruct": qwenvl_inference,
    "Qwen/Qwen2.5-VL-72B-Instruct": qwenvl_inference,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": qwenvl3_inference_vllm,

    # InternVL models
    "OpenGVLab/InternVL3-8B": internvl3_inference,
    "OpenGVLab/InternVL3-78B": internvl3_inference,
    "OpenGVLab/InternVL3_5-38B-Instruct": internvl3_inference,

    # Gemma3 models
    "google/gemma-3-27b-it": gemma3_inference,

    # Molmo2 models
    "allenai/Molmo2-8B": molmo2_inference_fast,

    # Phi4 models
    "microsoft/Phi-4-multimodal-instruct": phi4_inference_fast,

    # Ovis2 models
    "AIDC-AI/Ovis2-34B": ovis_inference,

    # LLaVA models
    "lmms-lab/LLaVA-Video-7B-Qwen2": llavavideo_inference,
    "lmms-lab/LLaVA-Video-72B-Qwen2": llavavideo_inference,
    "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct": llavaonevision_inference,
    "Efficient-Large-Model/LongVILA-R1-7B": longvila_inference,
    "DAMO-NLP-SG/VideoLLaMA3-7B": videollama3_inference,

    # Others
    "zai-org/cogvlm2-video-llama3-chat": cogvlm2_inference,
    "KangarooGroup/kangaroo": kangaroo_inference,
    "ernie-4.5-vl": inference_vllm_served_model,
    "glm-4.5v": inference_vllm_served_model,
    "rhymes-ai/Aria": aria_inference,
    "XiaomiMiMo/MiMo-VL-7B-RL": mimovl_inference,
    "openbmb/MiniCPM-V-4_5": minicpm_inference,
    "mPLUG/mPLUG-Owl3-7B-240728": mplug_owl3_inference,
}
