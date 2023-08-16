import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaConfig, LlamaForCausalLM

#
# PLAYGROUND
#


class Extra(nn.Module):
    def __init__(self, base, config: LlamaConfig):
        super().__init__()
        self._base = [base]
        self.mlp_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        self.mlp = LlamaMLP(self.mlp_config)
        torch.nn.init.xavier_uniform_(self.mlp.up_proj.weight.data)
        torch.nn.init.xavier_uniform_(self.mlp.down_proj.weight.data)
        torch.nn.init.xavier_uniform_(self.mlp.gate_proj.weight.data)

    def forward(self, x, *args, **kwargs):
        y = self.base(x).to(dtype=x.dtype)
        y = self.mlp(y)
        print(self.mlp.gate_proj.weight.data.std())
        # y = y.to(x.dtype)
        return y

    @property
    def base(self) -> nn.Module:
        return self._base[0]


tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/MythoMix-L2-13B-GPTQ/"))
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path=os.path.expanduser("~/models/MythoMix-L2-13B-GPTQ/"),
    model_basename="gptq_model-4bit-128g",
    use_safetensors=True,
    trust_remote_code=False,
    device="cuda:0",
    use_triton=False,
    quantize_config=None,
)

model.bfloat16()
model.model.lm_head.float()


print("hello")
print(model)
for p in model.parameters():
    p.requires_grad_(False)


base: LlamaForCausalLM = model.model
for p in base.model.norm.parameters():
    p.requires_grad_(True)

for p in base.model.layers[-1].parameters():
    if p.dtype.is_floating_point:
        p.requires_grad_(True)


extra = Extra(base.model.layers[-1].mlp, model.config)
base.model.layers[-1].mlp = extra
extra.to(device="cuda")


for _ in (bar := tqdm(range(3))):
    opt_fn = torch.optim.AdamW(base.parameters())
    x = torch.arange(2000)[None].cuda()
    for i in range(0, 2000, 500):
        loss = model(x[:, i:i+500], labels=x[:, i:i+500]).loss
        print(loss.item())
        loss.backward()
        opt_fn.step()
        opt_fn.zero_grad()

# bar.set_description(f'L:{loss:.4f}')
