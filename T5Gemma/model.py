"""
T5Gemma 2 classifier for CEFR level prediction (encoder-only conditions). 

One backbone (the T5Gemma 2 *encoder* only -- the decoder is dropped to save memory),
mean-pooled, with a swappable linear head:

    mode="nominal"  -> head emits K logits,   trained with cross-entropy   (baseline)
    mode="corn"     -> head emits K-1 logits,  trained with corn_loss        (ordinal)

Keeping the backbone and pooling identical across modes means any difference in QWK/MAE
is attributable to the head + loss, not to representation changes.

T5Gemma 2 (google/t5gemma-2-{270m-270m,1b-1b,4b-4b}) is pretrained-only and multimodal;
for text-only classification we pass input_ids/attention_mask and route through the text
encoder. The encoder is obtained via the standard HF `get_encoder()` with defensive
fallbacks, since this model family is new and the internal attribute path may shift
across transformers versions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


def _get_encoder(base) -> nn.Module:
    """Return the encoder submodule across plausible transformers layouts."""
    if hasattr(base, "get_encoder"):
        try:
            enc = base.get_encoder()
            if enc is not None:
                return enc
        except Exception:
            pass
    for path in ("encoder", "model.encoder", "model.model.encoder"):
        obj = base
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate the encoder on this model. Inspect `dir(base)` / "
        "`base.config` and set the encoder path explicitly. On the version this was "
        "written against, `base.get_encoder()` works."
    )


def _hidden_size(base, encoder):
    # T5Gemma2 ties all word embeddings, so the input-embedding dim is the model hidden size.
    emb = base.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.shape[-1]
    # fallback: T5Gemma2 nests it under config.encoder.text_config.hidden_size
    cfg = getattr(base, "config", None)
    for path in (("encoder", "text_config", "hidden_size"),
                 ("text_config", "hidden_size"),
                 ("hidden_size",)):
        node = cfg
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, int):
            return node
    raise ValueError("Could not infer hidden size from the model config.")


class T5Gemma2OrdinalClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int = 4,
        mode: str = "corn",                # "corn" or "nominal"
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
        drop_decoder: bool = True,
    ):
        super().__init__()
        assert mode in {"corn", "nominal"}
        self.mode = mode
        self.num_classes = num_classes

        base = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.encoder = _get_encoder(base)
        hidden = _hidden_size(base, self.encoder)

        # Free the decoder; classification only needs the (bidirectional) encoder.
        if drop_decoder:
            for attr in ("decoder",):
                if hasattr(base, attr):
                    try:
                        delattr(base, attr)
                    except Exception:
                        pass

        out_dim = num_classes if mode == "nominal" else num_classes - 1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)
        # Keep the small head in fp32 for stable logits/loss even with a bf16 backbone.
        self.head.to(torch.float32)

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B,T,1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = enc_out.last_hidden_state          # (B, T, H)
        pooled = self._mean_pool(last_hidden, attention_mask)
        logits = self.head(self.dropout(pooled).to(torch.float32))
        # Loss is computed in the Trainer (see train.py) so the head/loss pairing
        # stays explicit; we just surface logits here.
        return SequenceClassifierOutput(logits=logits)
