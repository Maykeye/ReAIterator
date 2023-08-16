from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os
import glob
import random
from backends.utils import CMD_FINETUNE_RESET, CMD_FINETUNE_STEP, CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT
from backends.utils import MODEL_BASENAME, MODEL_NAME_OR_PATH
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS
from typing import Optional

model: Optional[ExLlama] = None
tokenizer: Optional[ExLlamaTokenizer] = None
generator: Optional[ExLlamaGenerator] = None
gen_config = {}


def randomize_parms():
    def sample_setting(key):
        value = gen_config[key]
        delta = value * 0.1 * (random.random() - 0.5)
        return value + delta
    assert generator is not None
    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = sample_setting(G_TEMPERATURE)
    generator.settings.top_p = sample_setting(G_TOP_P)
    generator.settings.top_k = int(sample_setting(G_TOP_K))


def backend_exllama(cmd, prompt=None, cfg={}):
    global model
    global tokenizer
    global generator

    if cmd == CMD_INIT:
        model_directory = cfg[MODEL_NAME_OR_PATH]
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        if MODEL_BASENAME in cfg:
            model_path = os.path.join(model_directory, cfg[MODEL_BASENAME]+".safetensors")
        else:
            st_pattern = os.path.join(model_directory, "*.safetensors")
            model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path
        config.max_seq_len = 4096

        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(tokenizer_path)

        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        generator.disallow_tokens([tokenizer.eos_token_id])
        generator.settings.typical = 0.5
        gen_config[N_TOKENS] = cfg[N_TOKENS]
        gen_config[G_REPETITION_PENALTY] = cfg[G_REPETITION_PENALTY] or 1.20
        gen_config[G_TEMPERATURE] = cfg[G_TEMPERATURE] or 0.95
        gen_config[G_TOP_P] = cfg[G_TOP_P] or 0.75
        gen_config[G_TOP_K] = cfg[G_TOP_K] or 140
        return

    assert tokenizer is not None

    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return tokenizer.encode(prompt).shape[1]

    assert model is not None

    if cmd == CMD_GENERATE:
        randomize_parms()
        assert generator is not None
        output = generator.generate_simple(prompt, max_new_tokens=gen_config[N_TOKENS])
        return output

    if cmd == CMD_FINETUNE_RESET:
        return
    if cmd == CMD_FINETUNE_STEP:
        return

    raise ValueError(f"Unknown command {cmd}")
