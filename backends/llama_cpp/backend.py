import random
from llama_cpp import Llama, LlamaRAMCache
from backends.utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT
from backends.utils import CMD_FINETUNE_STEP, CMD_FINETUNE_RESET
from backends.utils import MODEL_NAME_OR_PATH
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS
from backends.utils import G_FINETUNE_STEP
from typing import Optional

model: Optional[Llama] = None
gen_config = {}


def randomize_parms():
    def sample_setting(value):
        delta = value * 0.1 * (random.random() - 0.5)
        new_value = value + delta
        if isinstance(value, int):
            new_value = int(new_value)
        return new_value

    return {k: sample_setting(gen_config[k])
            for k in [G_TOP_K, G_TOP_P, G_REPETITION_PENALTY, G_TEMPERATURE]}


def backend_llama_cpp(cmd, prompt:Optional[str]=None, cfg={}):
    global model
    global tokenizer

    if cmd == CMD_INIT:
        gguf_path = str(cfg[MODEL_NAME_OR_PATH])
        model = Llama(model_path=gguf_path, n_ctx=4096, n_gpu_layers=35)
        cache = LlamaRAMCache()
        model.set_cache(cache)
        gen_config[N_TOKENS] = cfg[N_TOKENS]
        gen_config[G_REPETITION_PENALTY] = cfg[G_REPETITION_PENALTY] or 1.20
        gen_config[G_TEMPERATURE] = cfg[G_TEMPERATURE] or 0.95
        gen_config[G_TOP_P] = cfg[G_TOP_P] or 0.75
        gen_config[G_TOP_K] = cfg[G_TOP_K] or 140
        gen_config[G_FINETUNE_STEP] = cfg[G_FINETUNE_STEP] or 75
        return


    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        assert model is not None
        tokens = model.tokenize(prompt.encode())
        return len(tokens)

    assert model is not None

    if cmd == CMD_GENERATE:
        assert prompt is not None
        current_config = randomize_parms()
        out = model(prompt=prompt, 
                    top_p=current_config[G_TOP_P],
                    top_k=current_config[G_TOP_K],
                    repeat_penalty=current_config[G_REPETITION_PENALTY],
                    temperature=current_config[G_TEMPERATURE])
        text = out["choices"][0]["text"]
        return prompt + text

    if cmd == CMD_FINETUNE_RESET:
        return

    if cmd == CMD_FINETUNE_STEP:
        return

    raise ValueError(f"Unknown command {cmd}")
