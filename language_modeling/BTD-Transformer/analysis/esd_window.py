#!/usr/bin/env python3
"""
esd_window.py

Script to analyze per-epoch ESD metrics over fixed-size segments and plot the ESD evolution per segment.

Place this in BTD-Transformer/analysis/ and run:

    python esd_window.py \
        --esd_dir ../../checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_51_lr_0.000125/stats \
        --window_size 20 \
        --output_dir graphs

This will read all esd_epoch_{i}.npy files, compute mean \hatalpha per epoch, split into consecutive windows of size <window_size>, and for each window
plot the evolution of mean \hatalpha (showing the slope) and save each as a separate graph in analysis/graphs/.

The alpha hat plotted is the fitted power‐law exponent of each layer’s empirical spectral density (ESD) of its weight matrix.  Concretely:

1. ESD of a layer: you take that layer’s weight matrix, compute its eigenvalues (or singular values), and form the empirical spectral density.
2. Power‐law fit: you fit the tail of that density to a function of the form

   $$
     p(\lambda)\propto \lambda^{-\alpha}
   $$

   (above some cutoff $\lambda_{\min}$).  The fitting procedure (in `net_esd_estimator`) returns an estimate $\hat\alpha$ of that exponent.
3. What it means:

    Small $\hat\alpha$ (e.g.\ 2–3) → heavy tail, indicating a lot of large eigenvalues and strong correlations in that layer’s weights.
    Large $\hat\alpha$ (e.g.\ 5–10) → light tail, indicating a faster drop‐off in eigenvalues and more “random‐matrix‐like” behaviour.

When you plot the mean $\hat\alpha$ on the y-axis, you’re showing the average tail exponent across all layers at that point in training—a proxy for how “heavy‐tailed” your network’s weight spectra are over time.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_esd_by_epoch(esd_dir):
    """
    Scan esd_dir for esd_epoch_{k}.npy files, load each,
    and return sorted arrays of epochs and their mean α.
    """
    pattern = os.path.join(esd_dir, 'esd_epoch_*.npy')
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).split('_')[2].split('.')[0])
    )

    epochs = []
    alpha_means = []
    for f in files:
        epoch = int(os.path.basename(f).split('_')[2].split('.')[0])
        data = np.load(f, allow_pickle=True).item()
        if 'alpha' not in data:
            raise KeyError(
                f"'alpha' key not found in {f}; available keys: {list(data.keys())}"
            )
        epochs.append(epoch)
        alpha_means.append(np.mean(data['alpha']))

    return np.array(epochs), np.array(alpha_means)

def main():
    parser = argparse.ArgumentParser(description='Plot ESD α over fixed windows of epochs')
    parser.add_argument('--esd_dir', required=True,
                        help='Directory containing esd_epoch_{i}.npy files')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of epochs per segment window')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save segment plots')
    args = parser.parse_args()

    epochs, alpha_means = load_esd_by_epoch(args.esd_dir)
    total = len(epochs)
    if total == 0:
        raise ValueError(f"No ESD files found in {args.esd_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    w = args.window_size

    # iterate non-overlapping windows
    segment_idx = 0
    for start in range(0, total, w):
        end = min(start + w, total)
        seg_epochs = epochs[start:end]
        seg_alpha = alpha_means[start:end]

        if len(seg_epochs) < 2:
            # skip windows too small to plot a trend
            break

        # slope = Δα / Δepoch_count
        slope = (seg_alpha[-1] - seg_alpha[0]) / (seg_epochs[-1] - seg_epochs[0])

        plt.figure(figsize=(8,5))
        plt.plot(seg_epochs, seg_alpha, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Mean $\\alpha$')
        plt.title(
            f'Segment {segment_idx+1}: Epochs {seg_epochs[0]}–{seg_epochs[-1]}\n'
            f'Rate = {slope:.4f} α-units/epoch'
        )
        plt.grid(True)
        plt.tight_layout()

        fname = f'esd_segment_{segment_idx+1}_{seg_epochs[0]}_{seg_epochs[-1]}.png'
        outpath = os.path.join(args.output_dir, fname)
        plt.savefig(outpath)
        plt.close()
        print(f"Saved: {outpath}")

        segment_idx += 1

if __name__ == '__main__':
    main()