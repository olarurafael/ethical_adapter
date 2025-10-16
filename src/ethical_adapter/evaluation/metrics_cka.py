# src/ethical_adapter/metrics_cka.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
#  CKA utilities
# ------------------------------------------------------------
def _gram_linear(x):
    return x @ x.T


def _center_gram(gram):
    n = gram.size(0)
    unit = torch.ones_like(gram)
    return gram - unit @ gram / n - gram @ unit / n + unit @ gram @ unit / (n * n)


def linear_CKA(x, y):
    """
    Compute the linear CKA similarity between two activation matrices.

    Args:
        x, y (torch.Tensor): Activation matrices of shape (n_samples, feature_dim)

    Returns:
        float: Linear CKA similarity score between x and y
    """


    x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
    gram_x = _center_gram(_gram_linear(x))
    gram_y = _center_gram(_gram_linear(y))
    hsic = (gram_x * gram_y).sum()
    var_x = torch.norm(gram_x)
    var_y = torch.norm(gram_y)
    return (hsic / (var_x * var_y)).item()


# ------------------------------------------------------------
#  Layerwise representation extraction
# ------------------------------------------------------------
@torch.no_grad()
def compute_layerwise_CKA(model_a, model_b, tokenizer,
                          text_list, max_length=128):
    """
    Compute layer-wise CKA similarity between two models.

    Args:
        model_a, model_b: Models with comparable layer structures
        tokenizer: Tokenizer for input processing
        text_list: List of texts used for probing
        max_length: Sequence length for tokenization

    Returns:
        dict[int, float]: Average CKA score per layer
    """

    model_a.eval()
    model_b.eval()

    all_scores = {}

    # Iterate through probe texts and collect CKA scores for each layer
    for text in tqdm(text_list, desc="CKA evaluation"):
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding="max_length",
                           max_length=max_length).to(model_a.device)

        # get hidden states from both models
        out_a = model_a(**inputs, output_hidden_states=True)
        out_b = model_b(**inputs, output_hidden_states=True)
        hidden_a = out_a.hidden_states
        hidden_b = out_b.hidden_states

        num_layers = min(len(hidden_a), len(hidden_b))
        for i in range(num_layers):
            # Flatten to (seq_len, hidden_dim) and align sequence lengths
            ha = hidden_a[i].squeeze(0)   
            hb = hidden_b[i].squeeze(0)
            min_len = min(ha.size(0), hb.size(0))
            cka_val = linear_CKA(ha[:min_len], hb[:min_len])
            all_scores.setdefault(i, []).append(cka_val)

    # Aggregate CKA scores across all samples per layer
    avg_scores = {i: float(np.mean(v)) for i, v in all_scores.items()}
    return avg_scores

@torch.no_grad()
def compute_layerwise_deltas(model_a, model_b, tokenizer, text_list, max_length=128):
    """
    Compute per-layer mean hidden activation deltas between two models.
    Returns: {layer_idx: mean(|Δ|)}
    """
    model_a.eval()
    model_b.eval()
    all_deltas = {}

    for text in text_list:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        ).to(model_a.device)

        out_a = model_a(**inputs, output_hidden_states=True)
        out_b = model_b(**inputs, output_hidden_states=True)
        hidden_a = out_a.hidden_states
        hidden_b = out_b.hidden_states
        num_layers = min(len(hidden_a), len(hidden_b))

        for i in range(num_layers):
            ha = hidden_a[i].squeeze(0)
            hb = hidden_b[i].squeeze(0)
            min_len = min(ha.size(0), hb.size(0))
            delta = (ha[:min_len] - hb[:min_len]).abs().mean().item()
            all_deltas.setdefault(i, []).append(delta)

    return {i: float(torch.tensor(v).mean()) for i, v in all_deltas.items()}

