"""Shared model loading for train and eval.

Both paths MUST build the base model identically. The adapter's stored
target_modules are matched by suffix, and the audio/vision towers contain
attention blocks whose submodules share the text stack's naming
(`…self_attn.q_proj.linear`). If the towers are present at adapter-load time,
PEFT matches them too and dies on Gemma4ClippableLinear. Training strips them,
so eval must strip them as well — hence one function used by both.
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TOWERS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def bnb_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base(model_path, attn="sdpa", verbose=True):
    """Load the 4-bit base model with the multimodal towers removed.

    Returns (model, tokenizer). Falls back to the intact model if removal or the
    text-only sanity forward fails.
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    kwargs = dict(quantization_config=bnb_4bit(), device_map={"": 0},
                  attn_implementation=attn, dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    try:
        removed = [n for n in TOWERS if model.model._modules.pop(n, None) is not None]
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model(**tok("ping", return_tensors="pt").to(model.device))
        if verbose:
            print(f"removed towers {removed}; text-only forward OK; "
                  f"vram={torch.cuda.memory_allocated() // 2**20}MiB")
    except Exception as e:
        print(f"tower removal failed ({e}); reloading intact model")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tok
