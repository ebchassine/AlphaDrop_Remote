"""
Load your AlphaDrop model, run WeightWatcher analyses on each layer,
and dump per-layer metrics to JSON and CSV.
"""
import os
import json
import torch
import pandas as pd
from weightwatcher import WeightWatcher

# — Modify these paths to point to your model and code root —
"""
Baseline model paths:
13 seed
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_13_lr_0.000125/model.pt
37 seed
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_37_lr_0.000125/model.pt
43 seed
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_43_lr_0.000125/model.pt
51 seed 
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_51_lr_0.000125/model.pt
71 seed 
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_71_lr_0.000125/model.pt

AD model paths 
13 seed (undertrained?)
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/tb_stage_update/ptb-adam/bs120-remove_last-eigs_thresh_50/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200_esd_interval10/alpha_median_min1.0_slope1.0_xmin_pos2.0_assign_tb_linear_map/seed_13_lr_0.000125/model.pt
51 seed
/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/tb_stage_update/ptb-adam/bs120-remove_last-eigs_thresh_50/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200_esd_interval10/alpha_median_min1.0_slope1.0_xmin_pos2.0_assign_tb_linear_map/seed_51_lr_0.000125/model.pt

"""
MODEL_PATH = "/absolute/path/from/gitignore/model.pt"
MODEL_PATH = "/jumbo/yaoqingyang/ewongchassine/Projects/TempBalance/language_modeling/checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_13_lr_0.000125/model.pt"
OUTPUT_DIR = "weightwatcher_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) load your model architecture & weights
from BTD_Transformer.models.transformer_upload_TempBal import TensorizedTransformerLM
# …or whatever your AlphaDrop model class is…
model = TensorizedTransformerLM( # ← use the same args you trained with
    n_token=…, n_layer=3, n_head=1, d_model=256, d_head=40,
    d_inner=2100, dropout=0.3, dropatt=0.0, …
)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2) instantiate WW and run
ww = WeightWatcher(model=model)
print("🔍 Running WeightWatcher analysis (this may take a minute)…")
results = ww.analyze()

# 3) save results
#   • full raw JSON
with open(os.path.join(OUTPUT_DIR, "ww_raw_results.json"), "w") as f:
    json.dump(results, f, indent=2)

#   • tabular summary (one row per layer)
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "ww_layer_summary.csv"), index=False)

print(f"Analysis complete. Results in {OUTPUT_DIR}/")
