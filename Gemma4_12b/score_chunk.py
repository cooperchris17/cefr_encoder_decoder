"""Score one chunk of one condition, save a .npy, exit.

Run as a subprocess so the OS reclaims all VRAM on exit. This is deliberately a
plain script rather than cell-based: it exists to be invoked repeatedly from the
driver notebook, and the process boundary is the whole point.

    python score_chunk.py <condition> <start> <end> [split]
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random

import numpy as np
import pandas as pd
import torch

from prompts import LEVELS, SYSTEM_PROMPT, CLASSIFICATION_REQUEST

CONDITION = sys.argv[1]
START, END = int(sys.argv[2]), int(sys.argv[3])
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "val"

MODEL_ID = "unsloth/gemma-4-12b-it"
DATA_DIR = "data"
RUN_DIR = "runs/gemma4_12b_prompting"
MAX_SEQ_LENGTH = 8192
CHUNK = 1
NUM_CLASSES = len(LEVELS)

os.makedirs(RUN_DIR, exist_ok=True)
out_path = f"{RUN_DIR}/{SPLIT}_{CONDITION}_{START}_{END}.npy"
if os.path.exists(out_path):
    print(f"exists, skipping: {out_path}")
    sys.exit(0)

# --- data ---------------------------------------------------------------
df_train = pd.read_json(f"{DATA_DIR}/train.jsonl", lines=True)
df = pd.read_json(f"{DATA_DIR}/{SPLIT}.jsonl", lines=True)
df_train = df_train[df_train["label"].isin(LEVELS)].reset_index(drop=True)
df = df[df["label"].isin(LEVELS)].reset_index(drop=True)


# --- few-shot pools (identical sampling to the notebook) ----------------
def sample_few_shot_examples(df, n_per_class, levels=LEVELS, random_state=42,
                             topic_col="topic"):
    examples = []
    rng = random.Random(random_state)
    for level in levels:
        subset = df[df["label"] == level]
        n = min(n_per_class, len(subset))
        if topic_col in subset.columns and subset[topic_col].nunique() >= n:
            topics = subset[topic_col].unique().tolist()
            rng.shuffle(topics)
            sampled = pd.concat([
                subset[subset[topic_col] == t].sample(n=1, random_state=random_state)
                for t in topics[:n]
            ])
        else:
            sampled = subset.sample(n=n, random_state=random_state)
        for _, row in sampled.iterrows():
            examples.append({"text": row["text"], "label": level,
                             "topic": row.get(topic_col, None)})
    return examples


def take_first_n_per_level(grouped, n_per_class, levels=LEVELS):
    out = []
    for level in levels:
        out.extend([ex for ex in grouped if ex["label"] == level][:n_per_class])
    return out


grouped = sample_few_shot_examples(df_train, n_per_class=3)
rng = random.Random(42)
EXAMPLE_SETS = {}
for n_shots, per_level in [(8, 1), (16, 2), (24, 3)]:
    s = take_first_n_per_level(grouped, per_level)
    rng.shuffle(s)
    EXAMPLE_SETS[n_shots] = s

shots = () if CONDITION == "zero_shot" else EXAMPLE_SETS[int(CONDITION.split("_")[-1])]

# --- model --------------------------------------------------------------
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
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


def build_prompt_messages(text, shots=()):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in shots:
        msgs.append({"role": "user",
                     "content": CLASSIFICATION_REQUEST + str(ex["text"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": CLASSIFICATION_REQUEST + str(text)})
    return msgs


@torch.no_grad()
def score_labels(text, shots=()):
    prompt_ids = tok.apply_chat_template(
        build_prompt_messages(text, shots), tokenize=True,
        add_generation_prompt=True)
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


# --- run ----------------------------------------------------------------
from tqdm import tqdm

probs = []
for i, text in enumerate(tqdm(df["text"].iloc[START:END],
                              desc=f"{CONDITION} {START}-{END}")):
    probs.append(score_labels(text, shots))
    if i % 10 == 0:
        torch.cuda.empty_cache()

np.save(out_path, np.vstack(probs))
print(f"saved {out_path}  reserved {torch.cuda.memory_reserved()/1e9:.1f} GB")
