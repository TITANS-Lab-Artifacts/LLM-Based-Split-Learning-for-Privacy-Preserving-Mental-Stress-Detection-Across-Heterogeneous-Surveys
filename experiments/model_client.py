"""
model_client.py
===============
Lightweight CLIENT side of the split-learning model.

Pipeline (per the spec, the client is as light as possible):
    token ids
      -> LoRA-adapted embedding layer            (embed_tokens + LoRA)
      -> projection head (2-layer MLP)           (the cut activation)

The cut activation has the model's hidden size so the server trunk can consume
it directly as `inputs_embeds`. The heavy transformer body is NOT here — it is
entirely server-side.

Only the LoRA deltas on the embedding and the small projection head are
trainable; the base embedding weights are frozen.

------------------------------------------------------------------------------
WHY THE EARLIER VERSION PRODUCED EMPTY OUTPUT (ANLS = 0.0)
------------------------------------------------------------------------------
The frozen LLaMA trunk consumes the cut as `inputs_embeds`. It was TRAINED on
the raw output of `embed_tokens` (the per-layer RMSNorm lives INSIDE each
decoder block, not at the model's input). The old ProjectionHead returned
    LayerNorm(x + h)
which feeds the trunk a re-centred, re-scaled distribution it has never seen at
its embedding entry point. With that out-of-distribution input, neither the
24M-param trunk LoRA nor the tiny client head could fit the targets in a short
run, the loss sat at ~4.4 (random), and generation emitted EOS immediately ->
empty strings -> ANLS 0.

Fixes here:
  * No LayerNorm at the cut. The head is a near-identity residual so the cut
    starts ≈ the raw embedding and the trunk sees an in-distribution input from
    step 0. The residual branch is zero-initialised (up-proj weights/bias = 0)
    so `cut == embedding` at init and the head only *adds* learned signal.
  * A `DIAGNOSTIC_IDENTITY` mode (config flag) bypasses the head AND the
    embedding-LoRA delta entirely, so the cut IS the raw frozen embedding. Use
    it to isolate "architecture (head/bottleneck/norm)" from "plumbing
    (gradient transport / inputs_embeds path)": if the loss drops with an
    identity cut, the bug was in the head; if it still sits at ~4.4, the bug is
    in the gradient transport.
"""

import torch
import torch.nn as nn

from config import DIAGNOSTIC_IDENTITY


class ProjectionHead(nn.Module):
    """
    2-layer MLP mapping hidden_size -> bottleneck -> hidden_size, added as a
    RESIDUAL with NO output normalization, so the cut activation stays in the
    same distribution as the raw embedding the frozen trunk expects.

    The up-projection is zero-initialised so the head is an exact identity at
    init (cut == embedding); it only learns to *add* a correction. This makes
    early training stable without forcing a LayerNorm that breaks the trunk's
    input distribution.
    """
    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, hidden_size)
        # zero-init the residual branch -> head starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):                 # x: [B, T, H]
        h = self.up(self.act(self.down(x)))
        return x + h                      # residual, NO norm


class ClientModel(nn.Module):
    """
    Holds the (LoRA-adapted) embedding layer and the projection head.

    The embedding is copied from the base LLaMA model so its token<->vector
    mapping matches the server trunk exactly, then frozen.
    """
    def __init__(self, embed_tokens, hidden_size, vocab_size,
                 lora_r, lora_alpha, bottleneck, device, dtype=torch.float16):
        super().__init__()
        self.device_ = device
        self.hidden_size = hidden_size
        self.identity = bool(DIAGNOSTIC_IDENTITY)

        # Copy the embedding weights onto the CLIENT device, frozen.
        import torch.nn as _nn
        with torch.no_grad():
            w = embed_tokens.weight.detach().to(device=device, dtype=dtype).clone()
        self.embed = _nn.Embedding(w.shape[0], w.shape[1],
                                   padding_idx=getattr(embed_tokens, "padding_idx", None))
        self.embed.weight = _nn.Parameter(w, requires_grad=False)
        self.embed.to(device)
        for p in self.embed.parameters():
            p.requires_grad = False

        # LoRA on the embedding: delta = (one_hot @ A^T) @ B^T, scaled.
        # A: [r, vocab]  (per-token low-rank code)   B: [hidden, r]
        self.lora_A = nn.Parameter(torch.zeros(lora_r, vocab_size))
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, lora_r))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)               # delta starts at 0
        self.lora_scale = lora_alpha / lora_r

        self.proj = ProjectionHead(hidden_size, bottleneck)

        self.to(device)
        # keep LoRA + head in fp32 for stable optimization; base embed stays fp16
        self.lora_A.data = self.lora_A.data.to(device).float()
        self.lora_B.data = self.lora_B.data.to(device).float()
        self.proj.to(device).float()

    def forward(self, input_ids):
        """
        input_ids: [B, T] (long); moved to the client device here.
        Returns the cut activation [B, T, H] (fp32) on the client device.

        In DIAGNOSTIC_IDENTITY mode the cut is the raw frozen embedding (no LoRA
        delta, no projection head) so any failure to learn must lie in the
        server / gradient-transport plumbing, not the client head.
        """
        input_ids = input_ids.to(self.device_)
        with torch.no_grad():
            base = self.embed(input_ids)           # [B,T,H] fp16, frozen
        base = base.float()

        if self.identity:
            # cut == raw embedding. requires_grad so the (now trivial) client
            # backward still runs cleanly; there are no trainable client params
            # contributing, which is exactly the point of the probe.
            return base

        # LoRA embedding delta: gather columns of A for each token, then up-proj
        a_sel = self.lora_A[:, input_ids]          # [r, B, T]
        delta = torch.einsum("hr,rbt->bth", self.lora_B, a_sel) * self.lora_scale
        emb = base + delta
        cut = self.proj(emb)                       # [B,T,H] fp32
        return cut

    def trainable_parameters(self):
        if self.identity:
            return []                              # nothing client-side to train
        return [self.lora_A, self.lora_B, *self.proj.parameters()]

    def param_count(self):
        return sum(p.numel() for p in self.trainable_parameters())
