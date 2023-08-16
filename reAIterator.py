from tqdm.auto import tqdm
import os
from pathlib import Path
from backends.utils import MODEL_NAME_OR_PATH, MODEL_BASENAME
from backends.utils import CMD_INIT, CMD_GENERATE, CMD_TOKEN_COUNT, CMD_FINETUNE_RESET, CMD_FINETUNE_STEP
from backends.utils import G_TEMPERATURE, G_REPETITION_PENALTY, G_TOP_P, G_TOP_K, N_TOKENS, G_FINETUNE_STEP
from backends.exllama.backend import backend_exllama as backend
# from backends.transformers.backend import backend_transformers as backend
from optparse import OptionParser

opt_parser = OptionParser()
opt_parser.add_option(
    "-m", "--model",
    action="store", dest="model", type="string",
    help="full path to the model (e.g. /path/model.safetensors)")
opt_parser.add_option(
    "-p", "--prompt",
    action="store", dest="prompt", type="string",
    default="/tmp/prompt.ptxt",
    help="full path to the text file with prompt (e.g. /tmp/prompt.ptxt)")
opt_parser.add_option(
    "-g", "--n_gens",
    action="store", dest="n_gens", type="int",
    default=4,
    help="number of responses to generate (default: 4)")
opt_parser.add_option(
    "-t", "--n_tokens",
    action="store", dest="n_tokens", type="int",
    default=128,
    help="number of tokens to generate (default: 128)")
opt_parser.add_option(
    "-x", "--max_len",
    action="store", dest="max_len", type="int",
    default=2000,
    help="if prompt has more tokens than this value, require user to rewrite the prompt (default: 2000)")
opt_parser.add_option(
    "--g_temperature", dest="g_temperature", type="float",
    help="Generator parameter: temperature")
opt_parser.add_option(
    "--g_repetition_penalty", dest="g_repetition_penalty", type="float",
    help="Generation parameter: repetition penalty")
opt_parser.add_option(
    "--g_top_p", dest="g_top_p", type="float",
    help="Generator parameter: top P")
opt_parser.add_option(
    "--g_top_k", dest="g_top_k", type="int",
    help="Generator parameter: top K")

options, args = opt_parser.parse_args()

assert options.model is not None, "Use --model /path/to/the/model.safetensors"
model_path = Path(options.model)
if model_path.suffix == ".safetensors":
    model_basename = model_path.stem
    model_path = model_path.parent
else:
    model_basename = None

prompt_path = options.prompt
reconstructed_prompt_path = f"{prompt_path}.act"
n_gens = options.n_gens
assert n_gens > 0
n_tokens = options.n_tokens
assert n_tokens > 0
max_len = options.max_len
assert max_len > 0

MARKER = ";;;"
MARKER_SKIP_SUFFIX = f"-"
MARKER_QUIT_SUFFIX = f"---"
MARKER_GLUE_SUFFIX = f"^"
MARKER_QUIT = f"{MARKER}{MARKER_QUIT_SUFFIX}"


def export_prompt():
    global prompt
    global reconstruct_prompt
    Path(reconstructed_prompt_path).write_text(reconstructed_prompt)
    Path(prompt_path).write_text(prompt)
    editor = os.environ.get("EDITOR", "vim")
    os.system(f"{editor} {prompt_path}")
    prompt = Path(f"{prompt_path}").read_text()


if Path(prompt_path).exists():
    prompt = Path(prompt_path).read_text()
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


backend(CMD_INIT, None, {
    MODEL_NAME_OR_PATH: model_path,
    MODEL_BASENAME: model_basename,
    G_TEMPERATURE: options.g_temperature,
    G_REPETITION_PENALTY: options.g_repetition_penalty,
    G_TOP_P: options.g_top_p,
    G_TOP_K: options.g_top_k,
    G_FINETUNE_STEP: None,  # NYI
    N_TOKENS: n_tokens
})

export_prompt()
while True:
    reconstructed_prompt = reconstruct_prompt(prompt)
    current_len = backend(CMD_TOKEN_COUNT, prompt=reconstructed_prompt)
    if current_len > max_len:
        input(f"vvv SPLIT vvv [promptlen: {current_len}/{max_len}")
        export_prompt()
        continue

    print(f"Prompt length: {current_len}")
    start = len(reconstructed_prompt)
    outs = [""]
    backend(CMD_FINETUNE_RESET)
    backend(CMD_FINETUNE_STEP, reconstructed_prompt)
    for _ in tqdm(range(n_gens), desc="Generations"):
        backend(CMD_FINETUNE_STEP, reconstructed_prompt)
        generated = backend(CMD_GENERATE, prompt=reconstructed_prompt)
        outs.append(generated[start:])

    outs = "\n~~~v~~~\n".join(outs)
    if MARKER_QUIT in prompt:
        prompt = prompt.replace(MARKER_QUIT, f'{outs}\n{MARKER_QUIT}', 1)
    else:
        prompt += outs
    export_prompt()
