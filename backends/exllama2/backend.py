from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Cache_8bit, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
import random
from backends.utils import CMD_FINETUNE_RESET, CMD_FINETUNE_STEP, CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT
from backends.utils import MODEL_NAME_OR_PATH
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS, G_MINIGEN_STEP, G_MINIGEN_STEP_MIN, G_MAX_LEN
from typing import Optional

model: Optional[ExLlamaV2] = None
tokenizer: Optional[ExLlamaV2Tokenizer] = None
generator: Optional[ExLlamaV2BaseGenerator] = None
gen_config = {}


def randomize_parms():
    def sample_setting(key):
        value = gen_config[key]
        delta = value * 0.1 * (random.random() - 0.5)
        return value + delta
    assert generator is not None
    ExLlamaV2Sampler.Settings()
    settings = ExLlamaV2Sampler.Settings()
    settings.token_repetition_range = 400
    settings.token_repetition_decay = 300
    settings.token_repetition_penalty = 1.0001
    settings.temperature = sample_setting(G_TEMPERATURE)
    settings.top_p = sample_setting(G_TOP_P)
    settings.top_k = int(sample_setting(G_TOP_K))
    if tokenizer:
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    return settings


def backend_exllamav2(cmd, prompt=None, cfg={}):
    global model
    global tokenizer
    global generator

    if cmd == CMD_INIT:
        model_directory = cfg[MODEL_NAME_OR_PATH]
        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()
        config.max_input_len = 4096
        CACHE_CLASS = ExLlamaV2Cache_8bit if cfg[G_MAX_LEN] > 8192 else ExLlamaV2Cache
        config.max_seq_len = max(8192, cfg[G_MAX_LEN])

        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(config)

        cache = CACHE_CLASS(model, lazy=True)
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        model.load_autosplit(cache)

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
            settings = randomize_parms()
            assert isinstance(text, str)
            text = generator.generate_simple(text, settings, num_tokens=to_gen)
        return text

    if cmd == CMD_FINETUNE_RESET:
        return
    if cmd == CMD_FINETUNE_STEP:
        return

    raise ValueError(f"Unknown command {cmd}")
