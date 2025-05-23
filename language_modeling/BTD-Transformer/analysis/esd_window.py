#!/usr/bin/env python3
"""
esd_window.py

Script to analyze per-epoch ESD metrics over fixed-size segments and plot the ESD evolution per segment.

Place this in BTD-Transformer/analysis/ and run:

    python esd_jacobian.py \
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
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_esd_by_epoch(esd_dir):
    """
    Load all esd_epoch_{i}.npy files and return sorted arrays of epochs and mean alphahat.
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
        epochs.append(epoch)
        alpha_means.append(np.mean(data['alphahat']))
    return np.array(epochs), np.array(alpha_means)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esd_dir', required=True,
                        help='Path to stats/ dir containing esd_epoch_{i}.npy files')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of epochs per segment window')
    parser.add_argument('--output_dir', default='graphs',
                        help='Directory under analysis/ to save segment plots')
    args = parser.parse_args()

    # Load epochs and mean alphahat
    epochs, alpha_means = load_esd_by_epoch(args.esd_dir)
    total_epochs = len(epochs)
    if total_epochs == 0:
        raise ValueError(f"No ESD files found in {args.esd_dir}")

    # Prepare output directory
    base_out = os.path.dirname(__file__)
    out_dir = os.path.join(base_out, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Compute and plot for each consecutive window segment
    w = args.window_size
    n_segments = total_epochs // w
    if n_segments == 0:
        raise ValueError(
            f"Not enough epochs ({total_epochs}) for even one segment of size {w}" )

    for seg in range(n_segments):
        start_idx = seg * w
        end_idx = start_idx + w
        seg_epochs = epochs[start_idx:end_idx+1]
        seg_alpha = alpha_means[start_idx:end_idx+1]
        # compute slope
        slope = (seg_alpha[-1] - seg_alpha[0]) / w

        plt.figure()
        plt.plot(seg_epochs, seg_alpha, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Mean $\\hat\\alpha$')
        plt.title(
            f'Segment {seg+1}: Epochs {seg_epochs[0]}–{seg_epochs[-1]}\n'
            f'Rate = {slope:.4f} per epoch'
        )
        plt.tight_layout()
        filename = f'esd_segment_{seg+1}_{seg_epochs[0]}_{seg_epochs[-1]}.png'
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
        print(f"Saved segment plot: {filename}")

if __name__ == '__main__':
    main()
