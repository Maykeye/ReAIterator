from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
from backend_utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT, N_TOKENS

MODEL_NAME_OR_PATH = "model_name_or_path"
MODEL_BASENAME = "model_basename"
USE_TRITON = "use_triton"
USE_SAFETENSORS = "use_safetensors"
TRUST_REMOTE_CODE = "trust_remote_code"
DEVICE = "cuda:0"
TEMPERATURE = 0.7

model = None
tokenizer = None


def backend_gptq(cmd, prompt=None, state={}):
    global model
    global tokenizer
    if cmd == CMD_INIT:
        assert prompt is None
        tokenizer = AutoTokenizer.from_pretrained(state[MODEL_NAME_OR_PATH])
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path=state[MODEL_NAME_OR_PATH],
            model_basename=state[MODEL_BASENAME],
            use_safetensors=state.get(USE_SAFETENSORS, True),
            trust_remote_code=state.get(TRUST_REMOTE_CODE, False),
            device=state.get(DEVICE, "cuda:0"),
            use_triton=state.get(USE_TRITON, False),
            quantize_config=None)
        return

    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return len(tokenizer(prompt).input_ids)

    if cmd == CMD_GENERATE:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        outs = model.generate(
            inputs=input_ids, do_sample=True,
            temperature=TEMPERATURE, max_new_tokens=N_TOKENS)
        return tokenizer.decode(outs[0], skip_special_tokens=True)

    raise ValueError(f"Unknown command {cmd}")