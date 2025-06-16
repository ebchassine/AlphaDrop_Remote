#!/usr/bin/env bash
# run_ad_exp_dropout.sh
# Sweep multiple dropout rates × seeds and write each to its own folder under checkpoints/

# Resolve your local absolute source path
src_path=$(pwd)
ckpt_base="${src_path}/checkpoints"

echo "Using source path: $src_path"
echo "Using checkpoint base: $ckpt_base"
echo ""

pwd
hostname
date

echo "Starting job..."
source ~/.bashrc
conda activate ww_train_lm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

cd ${src_path}/BTD-Transformer/

# ─────────────────────────────────────────────────────────────────────────────
# Define the dropout rates you want to sweep
DROPOUT_RATES=(0.10 0.20 0.30 0.40)
# ─────────────────────────────────────────────────────────────────────────────

# Loop over your SLURM array IDs (each line in ptb_tb.txt has a different seed & config)
for SLURM_ARRAY_TASK_ID in {7..11}; do

    # Read the config line for this task
    cfg=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
          ${src_path}/BTD-Transformer/scripts/txt/ptb_tb.txt)

    # Parse fields from that line
    lr=$(echo  $cfg | cut -f1  -d ' ')
    seed=$(echo  $cfg | cut -f2  -d ' ')
    pl_fitting=$(echo $cfg | cut -f3  -d ' ')
    remove_first=$(echo $cfg | cut -f4  -d ' ')
    remove_last=$(echo  $cfg | cut -f5  -d ' ')
    metric=$(echo    $cfg | cut -f6  -d ' ')
    assign_func=$(echo $cfg | cut -f7  -d ' ')
    layernorm=$(echo  $cfg | cut -f8  -d ' ')
    lr_min_ratio=$(echo $cfg | cut -f9  -d ' ')
    lr_slope=$(echo   $cfg | cut -f10 -d ' ')
    xmin_pos=$(echo   $cfg | cut -f11 -d ' ')
    max_epoch=$(echo  $cfg | cut -f12 -d ' ')
    esd_interval=$(echo $cfg | cut -f13 -d ' ')
    optim=$(echo     $cfg | cut -f14 -d ' ')
    n_layer=$(echo   $cfg | cut -f15 -d ' ')
    max_step=$(echo  $cfg | cut -f16 -d ' ')
    n_head=$(echo    $cfg | cut -f17 -d ' ')
    batch_size=$(echo $cfg | cut -f18 -d ' ')
    tb_update=$(echo $cfg  | cut -f19 -d ' ')

    # Inner loop: sweep dropout rates
    for DROPOUT in "${DROPOUT_RATES[@]}"; do

        # Build a unique run directory for this combo
        RUN_DIR=${ckpt_base}/dropout_${DROPOUT}/task${SLURM_ARRAY_TASK_ID}_seed${seed}_lr${lr}
        mkdir -p "${RUN_DIR}"

        echo "=== Task ${SLURM_ARRAY_TASK_ID} | seed=${seed} | lr=${lr} | dropout=${DROPOUT} ==="
        echo "→ Writing outputs to ${RUN_DIR}"

        CUDA_VISIBLE_DEVICES=3,7 python train_tb.py \
            --cuda \
            --data ../penn/ \
            --dataset ptb \
            --n_layer ${n_layer} \
            --seed ${seed} \
            --d_model 256 \
            --n_head ${n_head} \
            --d_head 40 \
            --d_inner 2100 \
            --dropout ${DROPOUT} \
            --work_dir ${RUN_DIR} \
            --dropatt 0.0 \
            --optim ${optim} \
            --lr ${lr} \
            --max_step ${max_step} \
            --tgt_len 32 \
            --mem_len 0 \
            --eval_tgt_len 32 \
            --batch_size ${batch_size} \
            --gpu0_bsz 1 \
            --pl-fitting ${pl_fitting} \
            --remove-first-layer ${remove_first} \
            --remove-last-layer ${remove_last} \
            --metric ${metric} \
            --assign-func ${assign_func} \
            --layernorm ${layernorm} \
            --lr-min-ratio ${lr_min_ratio} \
            --lr-slope ${lr_slope} \
            --xmin-pos ${xmin_pos} \
            --max_epoch ${max_epoch} \
            --esd-interval ${esd_interval} \
            --tb-update ${tb_update}

    done
done
