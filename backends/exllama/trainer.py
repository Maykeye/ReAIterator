from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from einops import rearrange

#
# PLAYGROUND
#
# POSSIBLE TODO:
# * replace lm_head nn.Linear(hidden->vocab) with CAPTURE_HIDDEN(x'=hidden)
# * after model.forward ignore model() logits
# * use captured x' on newly created layer, use nn.Linear which was initialized with lm_head


class Mixer(nn.Module):
    def __init__(self, input_dim, interm_dim) -> None:
        super().__init__()
        raise NotImplementedError()


model_directory = os.path.expanduser("~/models/MythoMix-L2-13B-GPTQ/")
model_config_path = os.path.join(model_directory, "config.json")
config = ExLlamaConfig(model_config_path)
config.model_path = model_directory + "gptq_model-4bit-128g.safetensors"
mixer = Mixer(config.vocab_size, config.hidden_size).cuda()

opt = torch.optim.AdamW(mixer.parameters())
for _ in (bar := tqdm(range(1024))):
    x = torch.randn(1, 100, config.vocab_size).cuda()
    y = mixer(x)
    loss = F.cross_entropy(y[:, :-1].ravel(), x[:, 1:].ravel())
    loss.backward()
    opt.step()
    opt.zero_grad()
    bar.set_description(f'{loss.item():.4f}')


opt = None

model: ExLlama = ExLlama(config)


tokenizer_path = os.path.join(model_directory, "tokenizer.model")
tokenizer: ExLlamaTokenizer = ExLlamaTokenizer(tokenizer_path)
cache = ExLlamaCache(model)

inputs = tokenizer.encode("A kitten c on the")
logits = model.forward(inputs, cache, False)
print("done")
print(logits)
y_true = inputs[0].cuda()
print(y_true)


y_pred = mixer(logits.cuda())
y_pred = y_pred[:, :-1, :]
print("^^^")
print(y_true)
y_pred = rearrange(y_pred, 'b t c -> (b t) c')
y_true = y_true[1:]
loss = F.cross_entropy(y_pred, y_true)
