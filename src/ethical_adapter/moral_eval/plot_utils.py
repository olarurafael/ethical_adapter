# /src/ethical_adapter/moral_eval/plot_utils.py
import matplotlib.pyplot as plt


def plot_moral_deltas(deltas, output_path):
    """Plot per-foundation changes in moral probability."""
    plt.figure(figsize=(10, 5))
    plt.barh(list(deltas.keys()), list(deltas.values()))
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Δ moral probability (adapter - vanilla)")
    plt.title("Per-foundation Moral Shifts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
