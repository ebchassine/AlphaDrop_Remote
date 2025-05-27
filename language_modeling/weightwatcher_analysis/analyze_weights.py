#!/usr/bin/env python3
"""
python analyze_weights.py   --model-path /jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_13_lr_0.000125/model.pt   --output-dir seed13_ww   --tgt-len 32   --ext-len 0   --mem-len 0
"""
import os
import sys
import argparse
import json
import torch
import pandas as pd
from weightwatcher import WeightWatcher

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
LM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BTD_DIR       = os.path.join(LM_ROOT, 'BTD-Transformer')
BTD_UTILS_DIR = os.path.join(BTD_DIR,   'utils')
sys.path.insert(0, BTD_UTILS_DIR)
sys.path.insert(0, BTD_DIR)

from models.transformer_upload_TempBal import TensorizedTransformerLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--output-dir', default='weightwatcher_analysis')
    # hyper‐parameters should match your training run
    p.add_argument('--n-layer', type=int,   default=3)
    p.add_argument('--n-head',  type=int,   default=1)
    p.add_argument('--d-model', type=int,   default=256)
    p.add_argument('--d-head',  type=int,   default=40)
    p.add_argument('--d-inner', type=int,   default=2100)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--dropatt', type=float, default=0.0)
    p.add_argument('--tgt-len', type=int, default=32,
                    help='Target sequence length (same as training)')
    p.add_argument('--ext-len', type=int, default=0,
                    help='Extended context length (same as training)')
    p.add_argument('--mem-len', type=int, default=0,
                    help='Memory length (same as training)')

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load checkpoint & infer vocab size
    ckpt = torch.load(args.model_path, map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    emb_key = next(k for k in sd if k.endswith('emb_layers.0.weight'))
    n_token = sd[emb_key].size(0)

    # build & load model (ignore name‐mismatches)
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
    model.load_state_dict(sd, strict=False)   # <-- allow mismatches

    # run WeightWatcher
    ww = WeightWatcher(model=model)
    results = ww.analyze()

    # save outputs
    # with open(os.path.join(args.output_dir, 'ww_raw_results.json'), 'w') as f:

    raw_out = os.path.join(args.output_dir, 'ww_raw_results.json')

    # make sure results is a list of dicts
    entries = results if isinstance(results, list) else [results]

    clean_results = []
    for layer_res in entries:
        clean = {}
        for k, v in layer_res.items():
            if isinstance(v, pd.DataFrame):
                clean[k] = v.to_dict(orient='records')
            elif isinstance(v, pd.Series):
                clean[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                # For any numpy scalar types
                if hasattr(v, 'item') and not isinstance(v, str):
                    try:
                        clean[k] = v.item()
                        continue
                    except:
                        pass
                # Fallback: try serializing, else stringify
                try:
                    json.dumps(v)
                    clean[k] = v
                except TypeError:
                    clean[k] = str(v)
        clean_results.append(clean)

    with open(raw_out, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"✅ Raw results written to {raw_out}")

if __name__ == '__main__':
    main()
