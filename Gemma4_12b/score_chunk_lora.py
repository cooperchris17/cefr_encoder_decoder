"""Score one chunk with the fine-tuned LoRA adapter, save a .npy, exit.

Run as a subprocess so the OS reclaims all VRAM on exit. Deliberately a plain
script rather than cell-based: it exists to be invoked repeatedly from the
driver notebook, and the process boundary is the whole point.

    python score_chunk_lora.py <start> <end> [split]
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch

from prompts import LEVELS, build_messages

START, END = int(sys.argv[1]), int(sys.argv[2])
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "test"

ADAPTER_PATH = "runs/gemma4_lora/lora_adapter"
DATA_DIR = "data"
RUN_DIR = "runs/gemma4_lora"
MAX_SEQ_LENGTH = 3072
CHUNK = 2
NUM_CLASSES = len(LEVELS)

os.makedirs(RUN_DIR, exist_ok=True)
out_path = f"{RUN_DIR}/{SPLIT}_probs_{START}_{END}.npy"
if os.path.exists(out_path):
    print(f"exists, skipping: {out_path}")
    sys.exit(0)

df = pd.read_json(f"{DATA_DIR}/{SPLIT}.jsonl", lines=True)
df = df[df["label"].isin(LEVELS)].reset_index(drop=True)

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
tok = getattr(tokenizer, "tokenizer", tokenizer)

EOT_ID = tok.convert_tokens_to_ids("<turn|>")
assert EOT_ID is not None and EOT_ID >= 0, "check the eot token string"

LAB_IDS = [tok(lvl, add_special_tokens=False).input_ids + [EOT_ID] for lvl in LEVELS]
MAX_LAB = max(len(l) for l in LAB_IDS)


@torch.no_grad()
def score_labels(text):
    prompt_ids = tok.apply_chat_template(
        build_messages(text), tokenize=True, add_generation_prompt=True)
    start = len(prompt_ids)
    pad_id = tok.pad_token_id
    ll = np.zeros(NUM_CLASSES)

    for lo in range(0, NUM_CLASSES, CHUNK):
        idx = list(range(lo, min(lo + CHUNK, NUM_CLASSES)))
        seqs = [prompt_ids + LAB_IDS[i] for i in idx]
        maxlen = max(len(s) for s in seqs)

        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s)
            attn[r, :len(s)] = 1

        input_ids, attn = input_ids.to(model.device), attn.to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        sliced = logits[:, start - 1: start - 1 + MAX_LAB, :]
        logprobs = torch.log_softmax(sliced.float(), dim=-1)

        for r, i in enumerate(idx):
            ll[i] = sum(logprobs[r, j, input_ids[r, start + j]].item()
                        for j in range(len(LAB_IDS[i])))
        del logits, sliced, logprobs

    ll -= ll.max()
    e = np.exp(ll)
    return e / e.sum()


from tqdm import tqdm

probs = []
for i, text in enumerate(tqdm(df["text"].iloc[START:END],
                              desc=f"lora {START}-{END}")):
    probs.append(score_labels(text))
    if i % 10 == 0:
        torch.cuda.empty_cache()

np.save(out_path, np.vstack(probs))
print(f"saved {out_path}  reserved {torch.cuda.memory_reserved()/1e9:.1f} GB")
