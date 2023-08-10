ReAIterator: reiterate text file through AI

Simple script to call the text editor to edit prompt for the model.
Editor is configured by $EDITOR env var(vim is default). 

Options:
--model, -m
    Path to the model. E.g. /home/user/models/llama2.gptq/gptq/model-4bit-128g.safetensors
--prompt, -p
    Path to write the story to E.g. /tmp/prompt.ptxt
--n_gens, -g
    Number of responses to generate (default: 4)
--n_tokens, -t
    Number of tokens to generate (default: 128)

Features are
* It allows to mark blocks of text to be excluded from generation.
* It generates 4 responses one by one to preserve precious VRAM
* It has token count threshold. After threshold is reached, script will commplain and
ask to edit the prompt.
* Goes with even simpler script roll.py to roll dice in another session

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
Remilia: Sakuya, lets start danmakue
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
Remilia: Sakuya, lets start danmakue
[The fight starts]
```

Old instruction after ;;;--- will be discarded for the next generation.
After that you can use the result to describe the fight scene.

TODO:
[ ] nested structure.
[ ] better markers for "remove N whitespaces from the last section"
[ ] use input() instead of system, so VS code in different window can be used
