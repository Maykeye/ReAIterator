import tempfile
from pathlib import Path
import torch
import torch.nn as nn
# Back-end commands
CMD_INIT = "init"
CMD_TOKEN_COUNT = "token-count"
CMD_GENERATE = "generate"
CMD_FINETUNE_STEP = "finetune-on-the-go"
CMD_FINETUNE_RESET = "finetune-reset"

MODEL_NAME_OR_PATH = "model_path_id"
MODEL_BASENAME = "model_basename"
N_TOKENS = "n_tokens"

# TODO: class instead of dict
G_TEMPERATURE = "g_temperature"
G_REPETITION_PENALTY = "g_repetition_penalty"
G_TOP_P = "g_top_p"
G_TOP_K = "g_top_k"
G_FINETUNE_STEP = "g_finetune_step"
G_MINIGEN_STEP_MIN = "g_minigen_step_min"
G_MINIGEN_STEP = "g_minigen_step"
G_MAX_LEN = "g_max_len"

# TODO: maybe generalize to sum([m(x) for m in module_list[1:]], module_list[0](x))


class Add(nn.Module):
    def __init__(self, lhs: nn.Module, rhs: nn.Module):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def forward(self, x: torch.FloatTensor):
        lhs = self.lhs(x)
        rhs = self.rhs(x)
        y = lhs + rhs
        return y


def move_model_according_to_tensor_data(model: nn.Module, tensor: torch.Tensor):
    return model.to(device=tensor.device, dtype=tensor.dtype)


def get_model_layers_list(model: nn.Module) -> nn.ModuleList:
    """Return Modulelist of hidden layers"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Can't find layer list")


def get_model_finalizer(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise ValueError("Can't find layer list")


def prepare_last_layer_cache(folder, loader):
    # TODO: parametrize (parametrize HF model location as well)
    # TODO: currently only llama is supported, there will be fail on GPT-named models
    # TODO: windows-style-path not supported
    cached_last_layer_path = tempfile.gettempdir()+"/" + folder.replace('/', '%')+".cache"
    if not Path(cached_last_layer_path).exists():
        print("Can't find the cached last layer, loading whole model")
        model = loader()
        last_layer = get_model_layers_list(model)[-1]
        state = {
            'config': model.config.to_dict(),
            'data': last_layer.state_dict()
        }
        print(f"Saving {cached_last_layer_path}")
        torch.save(state, cached_last_layer_path)
        del model
        torch.cuda.empty_cache()
    return cached_last_layer_path


def load_last_layer_from_cache(model, path) -> nn.Module:
    assert model is not None
    print("Loading last layer data")
    state = torch.load(path)
    config = model.config.__class__.from_dict(state["config"])
    last_layer = get_model_layers_list(model)[-1]
    last_layer = last_layer.__class__(config)
    last_layer.load_state_dict(state["data"])
    last_layer.cuda().bfloat16()
    return last_layer
