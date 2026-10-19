"""
run_experiment.py
=================
Run ONE split-learning setup end-to-end:
  partition train data across 10 clients  ->  build client+server  ->
  split-train  ->  evaluate on the full test set  ->  save results JSON.

CLI:
  python run_experiment.py --volume symmetric --schema homogeneous --frac 0.1
  python run_experiment.py --volume high --schema moderate --frac 0.05 --epochs 1

Results are written to result_split/<volume>__<schema>.json
With --save_model, the trained adapters are also saved under
  result_split/saved_models/<train_stem>__<test_stem>/
"""

import argparse
import json
import os
from datetime import datetime

import torch

from config import (VOLUME_LEVELS, SCHEMA_LEVELS, FRAC_DEFAULT, MAX_SAMPLE, EPOCHS,
                    BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LEN, MAX_NEW_TOKENS,
                    EVAL_BATCH, LORA_R, LORA_ALPHA, PROJ_HIDDEN, N_CLIENTS,
                    MODEL_ID, RANDOM_SEED, RESULT_DIR,
                    ensure_result_dir, get_devices)
from data_partition import partition_clients, load_test


def model_dir_name(train_file, test_file):
    """Directory name built from the train/test data file stems, e.g.
       train_mh_tech_dataset__test_mh_tech_dataset."""
    tr = os.path.splitext(os.path.basename(train_file))[0]
    te = os.path.splitext(os.path.basename(test_file))[0]
    return f"{tr}__{te}"


def save_model(client, server, tok, train_file, test_file, meta, mean_anls,
               base_dir=None):
    """
    Save the trained split model under a directory named by the train/test data.

    Only the trainable / reconstructable pieces are stored (the frozen base
    LLaMA is reloaded from the hub via MODEL_ID):
      <base_dir>/<train_stem>__<test_stem>/
        server_lora/            PEFT adapter for the LLaMA trunk
        client_state.pt         embedding LoRA (A,B) + projection head weights
        tokenizer/              the tokenizer (chat template, pad token, etc.)
        manifest.json           shapes / hyperparams needed to rebuild the client

    Returns the path to the saved-model directory.
    """
    base_dir = base_dir or os.path.join(RESULT_DIR, "saved_models")
    out_dir = os.path.join(base_dir, model_dir_name(train_file, test_file))
    os.makedirs(out_dir, exist_ok=True)

    # ── server trunk LoRA (PEFT) ─────────────────────────────────────────────
    server.hf().save_pretrained(os.path.join(out_dir, "server_lora"))

    # ── client adapters: embedding LoRA + projection head ────────────────────
    client_state = {
        "lora_A":     client.lora_A.detach().cpu(),
        "lora_B":     client.lora_B.detach().cpu(),
        "lora_scale": float(client.lora_scale),
        "proj":       {k: v.detach().cpu() for k, v in client.proj.state_dict().items()},
        "identity":   bool(getattr(client, "identity", False)),
    }
    torch.save(client_state, os.path.join(out_dir, "client_state.pt"))

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))

    # ── manifest: everything needed to rebuild the client + interpret the save ─
    manifest = {
        "base_model":        MODEL_ID,
        "train_file":        train_file,
        "test_file":         test_file,
        "hidden_size":       client.hidden_size,
        "vocab_size":        client.lora_A.shape[1],
        "lora_r":            LORA_R,
        "lora_alpha":        LORA_ALPHA,
        "proj_bottleneck":   PROJ_HIDDEN,
        "max_seq_len":       MAX_SEQ_LEN,
        "client_trainable_params": client.param_count(),
        "anls":              round(float(mean_anls), 6),
        "partition":         meta,
        "saved_at":          datetime.now().isoformat(),
        "layout": {
            "server_lora": "PEFT adapter dir for the LLaMA trunk "
                           "(load base from base_model, then PeftModel.from_pretrained)",
            "client_state.pt": "torch.load -> dict(lora_A, lora_B, lora_scale, proj, identity)",
            "tokenizer": "AutoTokenizer.from_pretrained on this dir",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  saved model -> {out_dir}")
    return out_dir



def run_one(volume, schema, frac=FRAC_DEFAULT, max_sample=MAX_SAMPLE, epochs=EPOCHS,
            batch_size=BATCH_SIZE, lr=LEARNING_RATE, max_eval=None,
            eval_batch=EVAL_BATCH, seed=RANDOM_SEED, save_model_flag=False):
    import random
    random.seed(seed); torch.manual_seed(seed)

    server_device, client_device, n_gpu = get_devices()
    result_dir = ensure_result_dir()
    name = f"{volume}__{schema}"
    print(f"\n{'='*70}\n  SETUP: {name}\n{'='*70}")
    print(f"  devices: server={server_device} client={client_device} (GPUs={n_gpu})")

    # ── 1. Partition ──────────────────────────────────────────────────────────
    clients_rows, meta = partition_clients(volume, schema, frac=frac,
                                           max_sample=max_sample, seed=seed)
    test_rows, test_file = load_test(schema)
    print(f"  test set: {test_file} ({len(test_rows)} rows)")

    # ── 2. Build models (import here so partition tests don't need torch) ──────
    from model_server import ServerModel, load_tokenizer
    from model_client import ClientModel
    from split_train import train_split
    from evaluate import evaluate_split

    tok = load_tokenizer()
    print("  building server (LoRA LLaMA trunk) ...")
    server = ServerModel(server_device)
    print("  building client (LoRA embedding + projection head) ...")
    client = ClientModel(
        embed_tokens=server.get_embed_tokens(),
        hidden_size=server.hidden_size, vocab_size=server.vocab_size,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, bottleneck=PROJ_HIDDEN,
        device=client_device)
    print(f"  client trainable params: {client.param_count():,}")
    print(f"  server trainable params: "
          f"{sum(p.numel() for p in server.trainable_parameters()):,}")

    # ── 3. Train (split learning) ─────────────────────────────────────────────
    train_stats = train_split(
        client, server, tok, clients_rows, server_device, client_device,
        epochs=epochs, batch_size=batch_size, lr=lr, max_len=MAX_SEQ_LEN)

    # ── 4. Evaluate on the full test set ──────────────────────────────────────
    print(f"  evaluating on test set (batch={eval_batch}) ...")
    mean_anls, sample_results = evaluate_split(
        client, server, tok, test_rows, server_device, client_device,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=eval_batch, max_eval=max_eval)
    print(f"  >>> {name}  ANLS = {mean_anls:.4f} ({mean_anls*100:.2f}%)")

    # ── 5. Save ────────────────────────────────────────────────────────────────
    out = {
        "setup": name, "volume": volume, "schema": schema,
        "model": MODEL_ID, "n_clients": N_CLIENTS,
        "frac": frac, "max_sample": max_sample, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "eval_batch": eval_batch,
        "max_seq_len": MAX_SEQ_LEN, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "proj_bottleneck": PROJ_HIDDEN,
        "client_trainable_params": client.param_count(),
        "partition": meta, "test_file": test_file, "test_rows": len(test_rows),
        "train_stats": train_stats,
        "metrics": {"anls": round(mean_anls, 6)},
        "saved_at": datetime.now().isoformat(),
        "samples": sample_results[:200],     # cap stored samples to keep file small
    }
    path = os.path.join(result_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {path}")

    # ── 5b. Optionally save the trained model (named by train/test data) ───────
    if save_model_flag:
        save_model(client, server, tok, meta["train_file"], test_file,
                   meta, mean_anls)

    # ── 6. Free GPU memory between setups ──────────────────────────────────────
    del server, client
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name, mean_anls


def main():
    p = argparse.ArgumentParser(description="Run one split-learning setup")
    p.add_argument("--volume", required=True, choices=VOLUME_LEVELS)
    p.add_argument("--schema", required=True, choices=SCHEMA_LEVELS)
    p.add_argument("--frac",   type=float, default=FRAC_DEFAULT,
                   help="Fraction of TRAIN data to use (0<frac<=1). Test is full.")
    p.add_argument("--max_sample", type=int, default=MAX_SAMPLE,
                   help="Absolute cap on total TRAIN rows (split across clients). "
                        "Overrides --frac when set, e.g. 4000 -> ~400/client.")
    p.add_argument("--epochs", type=int,   default=EPOCHS)
    p.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    p.add_argument("--eval_batch", type=int, default=EVAL_BATCH,
                   help=f"Eval generation batch size (default {EVAL_BATCH}).")
    p.add_argument("--lr",     type=float, default=LEARNING_RATE)
    p.add_argument("--max_eval", type=int, default=None,
                   help="Cap number of test rows scored (default: all).")
    p.add_argument("--save_model", action="store_true",
                   help="Save trained adapters under result_split/saved_models/"
                        "<train_stem>__<test_stem>/.")
    a = p.parse_args()
    run_one(a.volume, a.schema, frac=a.frac, max_sample=a.max_sample, epochs=a.epochs,
            batch_size=a.batch, lr=a.lr, max_eval=a.max_eval, eval_batch=a.eval_batch,
            save_model_flag=a.save_model)


if __name__ == "__main__":
    main()
