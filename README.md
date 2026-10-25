# Kikamba GPT
Built by Joseph (Botswana), Sept–Oct 2026.

## What it is
A small GPT trained from scratch on Kikamba text. It completes sentences. It can't answer questions or translate.

## Data
Trained on the Kikamba corpus at github.com/kithyadan/kikamba-corpus (not mine), about 2.6 million tokens.
I kept the diacritics (ũ, ĩ). I tested a no-diacritics version against it: my dad scored 20 samples in total (10 of each) from 1 to 5, and I made a rule beforehand that I'd switch only if plain won by more than 1 point. It won by 0.8, so I kept the diacritics.

## Model
BPE tokenizer (8,000 vocab) → 4-layer transformer, 4 heads, 256-dim embeddings, 256-token context, dropout 0.2. About 5.27M parameters.
Written in PyTorch, following Karpathy's Zero to Hero series and Sebastian Raschka's *Build a Large Language Model (From Scratch)*, plus other videos. Trained for 4,750 steps on a free Kaggle GPU.

## Results
Held-out bits per character (lower is better):
- bigram (counting): 3.298
- MLP (3-char context): 2.794
- GPT: 1.521

Blind test: I mixed 10 real corpus sentences with 10 model samples and gave them to my dad, a native speaker.
- For 9 of 20 he couldn't tell which was which.
- Of the 11 he called "model", 8 were model and 3 were real.
- His 1–5 "sounds like Kikamba" scores: real 2.7, model 2.5.

What I take from this: the model gets the words and sound of Kikamba right, even though the sentences don't always make sense.

## Demo
Try it: [huggingface.co/spaces/j-osep-h/kikamba-gpt](https://huggingface.co/spaces/j-osep-h/kikamba-gpt)
The model runs fully in the browser. I wrote the model and training code myself; the demo wrappers (app.py, export_web.py and kikamba.js) were generated with an AI assistant around my trained weights.

Files: `notebooks/` (00 data → 06 evaluate, in order), `models/gpt.py`, `tokenizer/`, `results/` (bpc numbers, loss curve, blind-test sheets).
