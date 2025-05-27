#!/usr/bin/env python3
"""
analyze_with_watcher_api.py

Use the WeightWatcher API to analyze an AlphaDrop model checkpoint,
produce per-layer ESD plots, and emit CSV/JSON summaries.
"""
import os
import sys
import argparse
import json
import torch
import weightwatcher as ww
import pandas as pd

# ─── 1) Expose your BTD-Transformer code ─────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
LM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BTD_DIR       = os.path.join(LM_ROOT, 'BTD-Transformer')
BTD_UTILS_DIR = os.path.join(BTD_DIR, 'utils')

sys.path.insert(0, BTD_UTILS_DIR)  # for proj_adaptive_softmax, etc.
sys.path.insert(0, BTD_DIR)        # for models, data_utils, etc.

from models.transformer_upload_TempBal import TensorizedTransformerLM

def main():
    p = argparse.ArgumentParser(
        description="WeightWatcher API analysis for AlphaDrop model"
    )
    p.add_argument('--model-path',  type=str, required=True,
                   help="Path to your model.pt checkpoint")
    p.add_argument('--output-dir',  type=str, default='ww_api_output',
                   help="Where to save plots, CSV, and JSON")
    # these must match your training
    p.add_argument('--tgt-len',  type=int, default=32, help="tgt_len")
    p.add_argument('--ext-len',  type=int, default=0,  help="ext_len")
    p.add_argument('--mem-len',  type=int, default=0,  help="mem_len")
    p.add_argument('--n-layer',  type=int, default=3,    help="n_layer")
    p.add_argument('--n-head',   type=int, default=1,    help="n_head")
    p.add_argument('--d-model',  type=int, default=256,  help="d_model")
    p.add_argument('--d-head',   type=int, default=40,   help="d_head")
    p.add_argument('--d-inner',  type=int, default=2100, help="d_inner")
    p.add_argument('--dropout',  type=float, default=0.3,  help="dropout")
    p.add_argument('--dropatt',  type=float, default=0.0,  help="dropatt")
    p.add_argument('--randomize',action='store_true',
                   help="Compare to randomized ESDs (correlation traps)")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── 2) Load checkpoint & infer vocab size ─────────────────────────
    ckpt = torch.load(args.model_path, map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    emb_key = next(k for k in sd if k.endswith('emb_layers.0.weight'))
    n_token = sd[emb_key].size(0)
    print(f"ℹ️  Inferred n_token = {n_token}")

    # ─── 3) Build & load your model ────────────────────────────────────
    model = TensorizedTransformerLM(
        n_token=n_token,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        dropatt=args.dropatt,
        tgt_len=args.tgt_len,
        ext_len=args.ext_len,
        mem_len=args.mem_len,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ─── 4) Run WeightWatcher analyze + ESD plotting ────────────────────
    watcher = ww.WeightWatcher(model=model)
    # chdir so WW’s savefig goes into your output folder
    cwd = os.getcwd()
    os.chdir(args.output_dir)

    details = watcher.analyze(
        plot=True,
        randomize=args.randomize,
        mp_fit=True,
        pool=True,
        savefig=True
    )  # returns a pandas.DataFrame of layer-by-layer metrics :contentReference[oaicite:0]{index=0}

    # back to original cwd
    os.chdir(cwd)

    # ─── 5) Persist results ─────────────────────────────────────────────
    csv_out = os.path.join(args.output_dir, 'ww_details.csv')
    details.to_csv(csv_out, index=False)
    print(f"✅ Layer details CSV → {csv_out}")

    summary = watcher.get_summary(details)
    json_out = os.path.join(args.output_dir, 'ww_summary.json')
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Model summary JSON → {json_out}")

    # ─── 6) Quick console report ────────────────────────────────────────
    print("\n📊 Per-layer α exponents:")
    print(details[['layer_id','alpha']])
    print("\n📈 Overall summary metrics:")
    print(summary)

if __name__ == '__main__':
    main()
