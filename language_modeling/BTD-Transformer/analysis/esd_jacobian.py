"""
esd_jacobian.py

Script to analyze per-epoch ESD metrics over sliding windows and plot the rate of change (Jacobian-like) of the ESD spectrum.

Place this in BTD-Transformer/analysis/ and run (for example )

    python esd_window.py \
        --esd_dir ../../checkpoints/tensorized/baseline/ptb-adam/bs120/tensor_transformer_3layer/head_1/max_step40000_max_epoch200_log_interval200/median_xmin_pos2/seed_51_lr_0.000125/stats \
        --window_size 20 \
        --output_dir graphs

This will read all esd_epoch_{i}.npy files, compute the finite-difference of mean \hatalpha over each sliding window, and save the resulting plot into analysis/graphs/.
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_esd_by_epoch(esd_dir):
    pattern = os.path.join(esd_dir, 'esd_epoch_*.npy')
    files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split('_')[2].split('.')[0]))
    epochs = []
    alphas = []
    for f in files:
        epoch = int(os.path.basename(f).split('_')[2].split('.')[0])
        data = np.load(f, allow_pickle=True).item()
        epochs.append(epoch)
        alphas.append(np.mean(data['alphahat']))
    return np.array(epochs), np.array(alphas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esd_dir', required=True,
                        help='Path to stats/ dir containing esd_epoch_{i}.npy')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of epochs per sliding window')
    parser.add_argument('--output_dir', default='graphs',
                        help='Directory under analysis/ to save plots')
    args = parser.parse_args()

    # Load epochs and mean alpha
    epochs, alpha_means = load_esd_by_epoch(args.esd_dir)
    print("EPOCHS", epochs)
    if len(epochs) < args.window_size + 1:
        raise ValueError(f"Need at least {args.window_size+1} epochs, found {len(epochs)}")

    # Compute finite differences over sliding windows
    centers = []  # center epoch of each window
    jacobian = []
    w = args.window_size
    for i in range(len(epochs) - w):
        start, end = epochs[i], epochs[i + w]
        delta = (alpha_means[i + w] - alpha_means[i]) / w
        center = epochs[i] + w // 2
        centers.append(center)
        jacobian.append(delta)

    # Prepare output
    out_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Plot rate of change of mean alpha
    plt.figure()
    plt.plot(centers, jacobian, marker='o')
    plt.xlabel('Epoch (window center)')
    plt.ylabel('d(mean $\\hat\\alpha$) / d(epoch)')
    plt.title(f'ESD Rate of Change over {w}-Epoch Windows')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'esd_jacobian_window{w}.png')
    plt.savefig(out_path)
    print(f"Jacobian plot saved to {out_path}")

if __name__ == '__main__':
    main()
