ReAIterator: reiterate text file through AI

Simple script to call the text editor to edit prompt for the model.
Editor is configured by $EDITOR env var(vim is default). 

Options:
--model, -m (mandatory)
    Path to the model. E.g. /home/user/models/llama2.gptq/gptq/model-4bit-128g.safetensors
--prompt, -p (default: /tmp/prompt.ptxt)
    Path to write the story to
--n_gens, -g (default: 4)
    Number of responses to generate
--n_tokens, -t (default: 128)
    Number of tokens to generate
--max_len, -x
    Maximum number of tokens in prompt
--g_temperature, --g_repetition_penalty, --g_top_p, --g_top_k
    Generator parameters

Example:

```console 
$ python reAIterator.py --model ~/models/MythoMix-L2-13B-GPTQ/gptq_model-4bit-128g.safetensors --prompt /tmp/a.ptxt -g 4 -t 120 --g_temperature=0.69 --g_repetition_penalty=1.13 --g_top_p=0.95 -x 4000
```

Features are
* It allows to mark blocks of text to be excluded from generation.
* It generates 4 responses one by one to preserve precious VRAM
* Responses are generated with randomized setting (each setting is +/- 5%)
* It has token count threshold. After threshold is reached, script will commplain and
ask to edit the prompt.
* Goes with even simpler script roll.py to roll dice in another session
* (Currently exLlama backend only): randomizes parameters every N tokens during single generation to shake text generation

Caveats:
* vim adds EOL. Use :set noeol / add ;;;- at the end to remove it

Blocks of text are separated by marker ;;;
;;;--- means exclude all further blocks
;;;- means exclude block until the next ;;;

For example you had

```txt
You are an AI. Write cool story.
### Instruction:
Write touhou story.

### Response:
(Cool adventures)
Remilia: Sakuya, lets start danmaku
[The fight starts]
```

And now you want to know what characters are in this setting.
So you change the story to 
```txt
You are an expert TTRPG game-master. Describe NPC characters.
### Instruction:
Write character description for Koakuma
(Copy paste background information for Koakuma from wiki)

### Response:
{"Koakuma":
  {"STR":;;;---
### Instruction:
Write touhou story.

### Response:
(Cool adventures)
Remilia: Sakuya, lets start danmaku
[The fight starts]
```

Old instruction after ;;;--- will be discarded for the next generation.
After that you can use the result to describe the fight scene.

Possible TODO:
[ ] parametrize backend via cmdline
[ ] parametrize finetune step and explore it so it works rather than hinder experience
[ ] figure out is it possible/feasable to finetune on the go for gptq/exllama
[ ] Add --remove-last-eol-because-i-am-too-lazy-to-configure-vim option    
[ ] use input() instead of system, so VS code in different window can be used    
[ ] nested structure.    
[ ] better markers for "remove N whitespaces from the last section"    
