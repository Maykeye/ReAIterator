from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
import torch.nn as nn
import random
from backends.utils import get_model_layers_list, get_model_finalizer
from backends.utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT
from backends.utils import CMD_FINETUNE_STEP, CMD_FINETUNE_RESET
from backends.utils import MODEL_NAME_OR_PATH
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS
from backends.utils import G_FINETUNE_STEP
from typing import Optional

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
gen_config = {}
finetune: Optional[nn.Module] = None
finetune_zero_state: Optional[dict] = None


def randomize_parms():
    def sample_setting(value):
        delta = value * 0.1 * (random.random() - 0.5)
        new_value = value + delta
        if isinstance(value, int):
            new_value = int(new_value)
        return new_value

    return {k: sample_setting(gen_config[k])
            for k in [G_TOP_K, G_TOP_P, G_REPETITION_PENALTY, G_TEMPERATURE]}


def backend_transformers(cmd, prompt=None, cfg={}):
    global model
    global tokenizer
    global finetune, finetune_zero_state

    if cmd == CMD_INIT:
        model_directory = str(cfg[MODEL_NAME_OR_PATH])
        model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        gen_config[N_TOKENS] = cfg[N_TOKENS]
        gen_config[G_REPETITION_PENALTY] = cfg[G_REPETITION_PENALTY] or 1.20
        gen_config[G_TEMPERATURE] = cfg[G_TEMPERATURE] or 0.95
        gen_config[G_TOP_P] = cfg[G_TOP_P] or 0.75
        gen_config[G_TOP_K] = cfg[G_TOP_K] or 140
        gen_config[G_FINETUNE_STEP] = cfg[G_FINETUNE_STEP] or 75
        model.requires_grad_(False)

        return

    assert tokenizer is not None

    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return len(tokenizer.encode(prompt))

    assert model is not None

    if cmd == CMD_GENERATE:
        inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        current_config = randomize_parms()
        current_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=gen_config[N_TOKENS],
            top_p=current_config[G_TOP_P],
            top_k=current_config[G_TOP_K],
            temperature=current_config[G_TEMPERATURE],
            repetition_penalty=current_config[G_REPETITION_PENALTY],
            pad_token_id=tokenizer.pad_token_id
        )
        out = model.generate(**inputs, generation_config=current_config)
        return tokenizer.decode(out[0])

    if cmd == CMD_FINETUNE_RESET:
        return

    if cmd == CMD_FINETUNE_STEP:
        return

    raise ValueError(f"Unknown command {cmd}")
