from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os
import glob
import random
from backends.utils import CMD_FINETUNE_RESET, CMD_FINETUNE_STEP, CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT
from backends.utils import MODEL_BASENAME, MODEL_NAME_OR_PATH
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS, G_MINIGEN_STEP, G_MINIGEN_STEP_MIN
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
    generator.settings.token_repetition_penalty_decay = 300
    generator.settings.token_repetition_penalty_sustain = 400
    generator.settings.token_repetition_penalty_max = 1.0001
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
        gen_config[G_MINIGEN_STEP] = cfg[G_MINIGEN_STEP] or 50
        gen_config[G_MINIGEN_STEP_MIN] = cfg[G_MINIGEN_STEP_MIN] or 3
        return

    assert tokenizer is not None

    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return tokenizer.encode(prompt).shape[1]

    assert model is not None

    if cmd == CMD_GENERATE:
        assert generator is not None
        text = prompt
        i = 0
        while i < gen_config[N_TOKENS]:
            print(i, gen_config[N_TOKENS])
            n = random.randint(gen_config[G_MINIGEN_STEP_MIN], gen_config[G_MINIGEN_STEP])
            to_gen = min(n, gen_config[N_TOKENS]-i)
            i += to_gen
            randomize_parms()
            text = generator.generate_simple(text, max_new_tokens=to_gen)
        return text

    if cmd == CMD_FINETUNE_RESET:
        return
    if cmd == CMD_FINETUNE_STEP:
        return

    raise ValueError(f"Unknown command {cmd}")
