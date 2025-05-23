#!/usr/bin/env python3
"""
compare_stats.py

Analysis script to compare Baseline vs AlphaDrop runs.
Place this in BTD-Transformer/analysis/ and run:

    python compare_stats.py \
        --baseline_root ../checkpoints/tensorized/baseline/ptb-adam/seed_13_lr_0.000125 \
        --alphadrop_dir ../checkpoints/tensorized/alphadrop/ptb-adam \
        --output_dir .

Note: Ensure your training scripts for both runs are updated to dump ESD at every epoch into the `stats/` folder.
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_training_stats(dir_path):
    path = os.path.join(dir_path, 'training_stats.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return np.load(path, allow_pickle=True).item()

def load_esd_by_epoch(root):
    files = sorted(glob.glob(os.path.join(root, 'stats', 'esd_epoch_*.npy')))
    epochs, esds = [], {}
    for f in files:
        e = int(os.path.basename(f).split('_')[2].split('.')[0])
        data = np.load(f, allow_pickle=True).item()
        epochs.append(e)
        esds[e] = data
    return sorted(epochs), esds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_root', required=True,
                        help='Path to a single baseline seed folder')
    parser.add_argument('--alphadrop_dir', required=True,
                        help='Path to AlphaDrop ptb-adam folder')
    parser.add_argument('--output_dir', default='.', help='Where to save plots')
    args = parser.parse_args()

    # Load stats
    baseline = load_training_stats(args.baseline_root)
    ad = load_training_stats(args.alphadrop_dir)

    # Basic curves: perplexity & loss
    for key, ylabel, fname in [
        ('ppl', 'Perplexity', 'perplexity.png'),
        ('loss', 'Loss', 'loss.png')
    ]:
        plt.figure()
        plt.plot(baseline['step'], baseline[f'train_{key}'], label='Baseline Train')
        plt.plot(baseline['step'], baseline[f'val_{key}'],   label='Baseline Val')
        plt.plot(ad['step'],       ad[f'train_{key}'],       label='AlphaDrop Train')
        plt.plot(ad['step'],       ad[f'val_{key}'],         label='AlphaDrop Val')
        plt.xlabel('Step'); plt.ylabel(ylabel);
        plt.title(f'{ylabel} Curves'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname))

    # Generalization gap (val_ppl - train_ppl)
    gap_base = np.array(baseline['val_ppl']) - np.array(baseline['train_ppl'])
    gap_ad   = np.array(ad['val_ppl'])       - np.array(ad['train_ppl'])
    plt.figure()
    plt.plot(baseline['step'], gap_base, label='Baseline Gap')
    plt.plot(ad['step'],       gap_ad,   label='AlphaDrop Gap')
    plt.xlabel('Step'); plt.ylabel('Perplexity Gap'); plt.title('Generalization Gap');
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'gen_gap.png'))

    # === ESD analysis: AlphaDrop only ===
    epochs, esds = load_esd_by_epoch(args.alphadrop_dir)
    # Mean alphahat vs epoch
    alpha_means = [np.mean(esds[e]['alphahat']) for e in epochs]
    plt.figure()
    plt.plot(epochs, alpha_means, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Mean $\\hat\\alpha$'); plt.title('AlphaDrop ESD Mean $\\hat\\alpha$ vs Epoch');
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'esd_alpha_vs_epoch.png'))

    # Alphahat distribution at last epoch
    last = epochs[-1]
    plt.figure()
    plt.hist(esds[last]['alphahat'], bins=20)
    plt.xlabel('$\\hat\\alpha$'); plt.ylabel('Count');
    plt.title(f'AlphaDrop $\\hat\\alpha$ Distribution (Epoch {last})');
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'esd_alpha_hist.png'))

    # Spectral Norm per layer at last epoch
    layers = esds[last]['longname']
    plt.figure(figsize=(10,4))
    plt.plot(layers, esds[last]['spectral_norm'], marker='o')
    plt.xticks(rotation=90);
    plt.xlabel('Layer'); plt.ylabel('Spectral Norm');
    plt.title(f'AlphaDrop Spectral Norm per Layer (Epoch {last})');
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'esd_spectral_norm.png'))

    print(f"Plots saved to {args.output_dir}")

if __name__ == '__main__':
    main()
