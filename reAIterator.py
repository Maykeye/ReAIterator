from tqdm.auto import tqdm
import os
from pathlib import Path
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

STORY = "/tmp/prompt.txt"
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


def export_prompt():
    global prompt
    global reconstruct_prompt
    Path(ACTUAL_PROMPT).write_text(reconstructed_prompt)
    Path(STORY).write_text(prompt)
    editor = os.environ.get("EDITOR", "vim")
    os.system(f"{editor} {STORY}")
    prompt = Path(f"{STORY}").read_text()


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_PATH)
if Path(STORY).exists():
    prompt = Path(STORY).read_text()
else:
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    prompt += "### Instruction: In a galaxy far far away\n\n"
    prompt += "### Response:\n"
reconstructed_prompt = prompt

model = AutoGPTQForCausalLM.from_quantized(
    MODEL_ID_PATH,
    model_basename=MODEL_BASENAME,
    use_safetensors=True,
    trust_remote_code=False,
    device="cuda:0",
    use_triton=USE_TRITON,
    quantize_config=None)


def reconstruct_prompt(whole_prompt):
    parts = whole_prompt.split(";;;")
    res = ""
    for part in parts:
        part = part.removeprefix("\n")
        if part.startswith("---"):
            break
        if part.startswith("-"):
            continue

        if part.startswith('^'):
            part = part[1:]
            res = res.removesuffix("\n")
        res += part
    return res


export_prompt()
while True:
    reconstructed_prompt = reconstruct_prompt(prompt)
    initial_len = len(reconstructed_prompt)
    input_ids = tokenizer(reconstructed_prompt, return_tensors='pt').input_ids.cuda()
    if input_ids.shape[1] > PROMPT_LEN_TO_SPLIT:
        input(f"vvv SPLIT vvv [promptlen: {input_ids.shape[1]}/{PROMPT_LEN_TO_SPLIT}")
        export_prompt()
        continue

    print(f"Prompt length: {input_ids.shape[1]}")
    start = input_ids.shape[1]
    outs = [""]
    for _ in tqdm(range(N_GENS), desc="Generations"):
        output = model.generate(inputs=input_ids,
                                do_sample=True,
                                temperature=TEMPERATURE,
                                max_new_tokens=N_TOKENS)
        outs.append(tokenizer.decode(output[0][start:], skip_special_tokens=True))

    outs = "\n~~~v~~~\n".join(outs)
    if MARKER_QUIT in prompt:
        prompt = prompt.replace(MARKER_QUIT, f'{outs}\n{MARKER_QUIT}', 1)
    else:
        prompt += outs
    export_prompt()
