from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os
import glob
from backend_utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT, N_TOKENS


MODEL_NAME_OR_PATH = "model_directory"
MODEL_BASENAME = "model_basename"
model = None
tokenizer = None
generator = None


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
        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(tokenizer_path)

        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        generator.disallow_tokens([tokenizer.eos_token_id])

        # TODO: configure
        generator.settings.token_repetition_penalty_max = 1.2
        generator.settings.temperature = 0.95
        generator.settings.top_p = 0.75
        generator.settings.top_k = 140
        generator.settings.typical = 0.5
        return

    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return tokenizer.encode(prompt).shape[1]

    if cmd == CMD_GENERATE:
        n_tokens = cfg[N_TOKENS]
        output = generator.generate_simple(prompt, max_new_tokens=n_tokens)
        return output

    raise ValueError(f"Unknown command {cmd}")
