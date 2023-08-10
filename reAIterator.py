from tqdm.auto import tqdm
import os
from pathlib import Path

STORY = "/tmp/prompt.ptxt"
ACTUAL_PROMPT = f"{STORY}.act"
MODEL_ID_PATH = os.path.expanduser("~/models/Nous-Hermes-Llama2-GPTQ")
USE_TRITON = False
MODEL_BASENAME = "gptq_model-4bit-128g"
PROMPT_LEN_TO_SPLIT = 2000
N_GENS = 4
N_TOKENS = 128
TEMPERATURE = 0.7
MARKER = ";;;"
MARKER_SKIP = f"{MARKER}-"
MARKER_QUIT = f"{MARKER}---"

# Back-end commands
CMD_INIT = "init"
CMD_TOKEN_COUNT = "token-count"
CMD_GENERATE = "generate"


def export_prompt():
    global prompt
    global reconstruct_prompt
    Path(ACTUAL_PROMPT).write_text(reconstructed_prompt)
    Path(STORY).write_text(prompt)
    editor = os.environ.get("EDITOR", "vim")
    os.system(f"{editor} {STORY}")
    prompt = Path(f"{STORY}").read_text()


def gptq(cmd, prompt=None, state={}):
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer
    if cmd == CMD_INIT:
        assert prompt is None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_PATH)
        model = AutoGPTQForCausalLM.from_quantized(
            MODEL_ID_PATH,
            model_basename=MODEL_BASENAME,
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=USE_TRITON,
            quantize_config=None)
        state["tokenizer"] = tokenizer
        state["model"] = model
        return

    tokenizer, model = state["tokenizer"], state["model"]
    if cmd == CMD_TOKEN_COUNT:
        assert prompt is not None
        return len(tokenizer(prompt).input_ids)
    if cmd == CMD_GENERATE:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outs = model.generate(
            inputs=input_ids, do_sample=True,
            temperature=TEMPERATURE, max_new_tokens=N_TOKENS)
        return tokenizer.decode(outs[0], skip_special_tokens=True)
    raise ValueError(f"Unknown command {cmd}")


if Path(STORY).exists():
    prompt = Path(STORY).read_text()
else:
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    prompt += "### Instruction: In a galaxy far far away\n\n"
    prompt += "### Response:\n"
reconstructed_prompt = prompt


def reconstruct_prompt(whole_prompt):
    parts = whole_prompt.split(MARKER)
    res = ""
    for part in parts:
        part = part.removeprefix("\n")
        if part.startswith("---"):
            break
        if part.startswith("-^"):
            res = res.removesuffix("\n")
            continue
        if part.startswith("-"):
            continue
        if part.startswith('^'):
            part = part[1:]
            res = res.removesuffix("\n")
        res += part
    return res


backend = gptq
backend(CMD_INIT)

export_prompt()
while True:
    reconstructed_prompt = reconstruct_prompt(prompt)
    n_tokens = backend(CMD_TOKEN_COUNT, prompt=reconstructed_prompt)
    if n_tokens > PROMPT_LEN_TO_SPLIT:
        input(f"vvv SPLIT vvv [promptlen: {n_tokens}/{PROMPT_LEN_TO_SPLIT}")
        export_prompt()
        continue

    print(f"Prompt length: {n_tokens}")
    start = len(reconstructed_prompt)
    outs = [""]
    for _ in tqdm(range(N_GENS), desc="Generations"):
        generated = backend(CMD_GENERATE, prompt=reconstructed_prompt)
        outs.append(generated[start:])

    outs = "\n~~~v~~~\n".join(outs)
    if MARKER_QUIT in prompt:
        prompt = prompt.replace(MARKER_QUIT, f'{outs}\n{MARKER_QUIT}', 1)
    else:
        prompt += outs
    export_prompt()
