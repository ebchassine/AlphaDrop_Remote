#!/bin/bash
#
# run_ad_exp_hparam_fixed.sh
#
# Fixes:
#   – “cd” into the correct BTD-Transformer folder
#   – Uses `train_tb.py` (not a nonexistent file)
#   – Creates the output directory before writing into it
#   – Passes “--tb-update stage” so that ESD is recomputed each epoch
#   – Saves into a “checkpoints/…” parent folder (just like your baseline/AD experiments)
#

# (1) Move up to the BTD-Transformer folder (relative to scripts/tensorized/)
cd ../BTD-Transformer
export PYTHONPATH=..

# (2) Fixed random seed
SEED=51

# (3) Learning rates to sweep
LEARNING_RATES=(0.000125 0.00025 0.0005)

# (4) Which assign‐functions to try
ASSIGN_FUNCS=(tb_linear_map tb_sqrt tb_log2)

# (5) Loop over LR × assign‐func
for LR in "${LEARNING_RATES[@]}"; do
  for AFUNC in "${ASSIGN_FUNCS[@]}"; do
    TAG="lr${LR}_seed${SEED}_afunc${AFUNC}"

    #   • We will mirror your “checkpoints/tensorized/…” structure
    OUT_DIR="../../checkpoints/tensorized/ptb-adam/exp_${TAG}"
    INFO_TXT="${OUT_DIR}/experiment_info.txt"

    # (6) Make sure the folder actually exists before `train_tb.py` writes anything:
    mkdir -p "${OUT_DIR}"

    # (7) Launch the existing `train_tb.py` (exactly as the repo expects)
    CUDA_VISIBLE_DEVICES=1 python train_tb.py \
      --cuda \
      --data ../penn \
      --dataset ptb \
      --n_layer 3 \
      --d_model 256 \
      --n_head 1 \
      --d_head 40 \
      --d_inner 2100 \
      --dropout 0.3 \
      --dropatt 0.0 \
      --lr ${LR} \
      --max_step 40000 \
      --tgt_len 32 \
      --mem_len 0 \
      --eval_tgt_len 32 \
      --batch_size 120 \
      --gpu0_bsz 1 \
      --block_length 4 \
      --max_epoch 200 \
      --log-interval 200 \
      --esd-interval 200 \
      --pl-fitting median \
      --xmin-pos 2 \
      --remove-first-layer True \
      --remove-last-layer True \
      --metric alpha \
      --assign-func ${AFUNC} \
      --lr-min-ratio 0.7 \
      --lr-slope 0.6 \
      --tb-update stage \
      --layernorm False \
      --seed ${SEED} \
      --work_dir "${OUT_DIR}"

    # (8) Finally, write a short experiment_info.txt into that same folder
    echo "Experiment Info: ${TAG}" > "${INFO_TXT}"
    echo "Learning rate: ${LR}" >> "${INFO_TXT}"
    echo "Assign Func: ${AFUNC}" >> "${INFO_TXT}"
  done
done
