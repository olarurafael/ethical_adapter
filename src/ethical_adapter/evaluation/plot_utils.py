# src/ethical_adapter/plot_utils.py
import os
import logging
import matplotlib.pyplot as plt


def plot_cka_and_delta(cka_scores, delta_scores,
                       adapter_layers=None, output_path="./eval/cka_delta_plot.png"):
    """
    Plot layerwise CKA similarity and mean activation deltas on dual axes.
    """


    layers = sorted(cka_scores.keys())
    cka_vals = [cka_scores[i] for i in layers]
    delta_vals = [delta_scores.get(i, 0.0) for i in layers]

    _, ax1 = plt.subplots(figsize=(10, 5))

    # plot cka smilarity curve
    color1 = "tab:blue"
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("CKA similarity", color=color1)
    ax1.plot(layers, cka_vals, color=color1, marker="o", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.9, 1.02)

    # plot mean activation delta curve
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mean |Δ hidden|", color=color2)
    ax2.plot(layers, delta_vals, color=color2, marker="s", linewidth=2, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color2)

    # highlight adapter layers if provided
    if adapter_layers:
        for l in adapter_layers:
            ax1.axvline(l, color="gray", linestyle="--", alpha=0.4)

    plt.title("Layerwise CKA vs Δ Hidden Activation")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logging.info("Saved CKA and Δ activation plot to  %s", output_path)

