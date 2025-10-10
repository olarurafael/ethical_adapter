# scripts/sanity_check.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters


# load something small so it fits on one gpu
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# a short input
text = "Ethical AI matters because"
inputs = tokenizer(text, return_tensors="pt")

# baseline forward
with torch.no_grad():
    base_logits = model(**inputs).logits

# choose a couple of Linear layers to wrap
cfg = AdapterConfig(
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_modules=[
        "model.decoder.layers.10.self_attn.out_proj",
        "model.decoder.layers.11.self_attn.out_proj",
    ],
)

# inject adapters
res = inject_adapters(model, cfg)
print("injected:", res.injected_layers)

# run the same input again
with torch.no_grad():
    new_logits = res.model(**inputs).logits

# compare numerically
diff = (new_logits - base_logits).abs().max().item()
print(f"max abs diff: {diff:.6e}")