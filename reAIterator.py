from tqdm.auto import tqdm
import os
from pathlib import Path
from backend_utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT, N_TOKENS
# import backend_gptq
import backend_exllama
from optparse import OptionParser

opt_parser = OptionParser()
opt_parser.add_option(
    "-m", "--model",
    action="store", dest="model", type="string",
    help="full path to the model")

options, args = opt_parser.parse_args()
assert options.model is not None, "Use --model /path/to/the/model.safetensors"
model_path = Path(options.model)
if model_path.suffix == ".safetensors":
    model_basename = model_path.stem
    model_path = model_path.parent
else:
    model_basename = None


STORY = "/tmp/prompt.ptxt"
ACTUAL_PROMPT = f"{STORY}.act"
PROMPT_LEN_TO_SPLIT = 2000
N_GENS = 4
MARKER = ";;;"
MARKER_SKIP_SUFFIX = f"-"
MARKER_QUIT_SUFFIX = f"---"
MARKER_GLUE_SUFFIX = f"^"
MARKER_QUIT = f"{MARKER}{MARKER_QUIT_SUFFIX}"


def export_prompt():
    global prompt
    global reconstruct_prompt
    Path(ACTUAL_PROMPT).write_text(reconstructed_prompt)
    Path(STORY).write_text(prompt)
    editor = os.environ.get("EDITOR", "vim")
    os.system(f"{editor} {STORY}")
    prompt = Path(f"{STORY}").read_text()


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
        if part.startswith(MARKER_QUIT_SUFFIX):
            if part[len(MARKER_QUIT_SUFFIX):len(MARKER_QUIT_SUFFIX)+1] == MARKER_GLUE_SUFFIX:
                res = res.removesuffix("\n")
            break
        if part.startswith(MARKER_SKIP_SUFFIX):
            if part[len(MARKER_SKIP_SUFFIX):len(MARKER_SKIP_SUFFIX)+1] == MARKER_GLUE_SUFFIX:
                res = res.removesuffix("\n")
            continue
        if part.startswith(MARKER_GLUE_SUFFIX):
            part = part[1:]
            res = res.removesuffix("\n")
        res += part
    return res


backend = backend_exllama.backend_exllama
backend(CMD_INIT, None, {
    backend_exllama.MODEL_NAME_OR_PATH: model_path,
    backend_exllama.MODEL_BASENAME: model_basename
})

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
