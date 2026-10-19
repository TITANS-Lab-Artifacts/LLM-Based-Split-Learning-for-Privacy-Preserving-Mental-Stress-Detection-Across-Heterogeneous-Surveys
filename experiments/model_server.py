"""
model_server.py
===============
Heavy SERVER side of the split-learning model.

It loads the full LLaMA-3.2-3B, applies LoRA to the transformer trunk, and
exposes a forward that consumes the CLIENT's cut activation (as inputs_embeds)
rather than token ids. The cut is right after the client projection head, so
the server runs the entire decoder stack + final norm + LM head.

  cut activation [B,T,H]  ->  decoder layers (LoRA)  ->  norm  ->  lm_head
                          ->  next-token logits  ->  causal-LM loss

Because the server takes `inputs_embeds`, the client embedding gradient flows
back out as the gradient w.r.t. that input tensor — exactly the cut-layer
gradient that split learning transports to the client.

A factory also builds the shared base model and hands the client its embedding
layer, so both sides agree on the token<->vector mapping and hidden size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from config import (MODEL_ID, LORA_R, LORA_ALPHA, LORA_DROPOUT,
                    LORA_TARGETS_SERVER, PROJ_HIDDEN, SYSTEM_PROMPT)

LLAMA3_TEMPLATE = (
    "{%- set loop_messages = messages -%}"
    "{%- for message in loop_messages -%}"
    "{%- set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' -%}"
    "{%- if loop.index0 == 0 -%}{%- set content = bos_token + content -%}{%- endif -%}"
    "{{ content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif -%}")


def load_tokenizer():
    # clean_up_tokenization_spaces=False: the cleanup step is for WordPiece and
    # is destructive for BPE (strips spaces before punctuation); turning it off
    # both silences the warning and keeps decoded text correct for LLaMA's BPE.
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, clean_up_tokenization_spaces=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_TEMPLATE
    return tok


def _attn_impl():
    try:
        import flash_attn  # noqa
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class ServerModel(nn.Module):
    """
    Wraps the LoRA-adapted LLaMA. Trains/evaluates from `inputs_embeds`
    (the cut activation), not token ids.
    """
    def __init__(self, server_device, dtype=torch.float16):
        super().__init__()
        self.device_ = server_device
        self.dtype_  = dtype                         # the model's compute dtype

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, device_map={"": server_device},
            trust_remote_code=True, attn_implementation=_attn_impl())
        base.enable_input_require_grads()
        self.peft = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS_SERVER,
            bias="none"))

        self.hidden_size = base.config.hidden_size
        self.vocab_size  = base.config.vocab_size

    # the underlying HF model (peft-wrapped)
    def hf(self):
        return self.peft

    def get_embed_tokens(self):
        """Embedding layer for the client (shared token<->vector mapping)."""
        return self.peft.get_input_embeddings()

    def forward(self, cut_activation, attention_mask, labels=None):
        """
        cut_activation : [B,T,H] on the SERVER device (already transferred),
                         requires_grad=True so we can read its gradient.
        attention_mask : [B,T] on the server device.
        labels         : [B,T] on the server device or None.
        Returns (loss, logits). When labels is None, loss is None.

        The cut activation arrives in fp32 (client keeps its small params in
        fp32 for stable optimization); it MUST be cast to the model's compute
        dtype (fp16) here, otherwise an fp32 input into the fp16 trunk raises a
        dtype mismatch deep in the stack (surfacing at lm_head). The cast is
        differentiable, so gradients still flow back to the fp32 client params.
        """
        out = self.peft(
            inputs_embeds=cut_activation.to(self.device_, dtype=self.dtype_),
            attention_mask=attention_mask.to(self.device_),
            labels=labels.to(self.device_) if labels is not None else None,
            use_cache=False,
        )
        return out.loss, out.logits

    def trainable_parameters(self):
        return [p for p in self.peft.parameters() if p.requires_grad]
