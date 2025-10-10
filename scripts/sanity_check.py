# scripts/sanity_check.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters
from ethical_adapter.introspect import print_layer_map, print_adapter_summary



# ------------------------------------------------------------
# load model + tokenizer
# ------------------------------------------------------------
model_name = "allenai/OLMo-2-0425-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=torch.bfloat16, 
    device_map="auto"
    )
model.eval()

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

model = freeze_model(model)

# ------------------------------------------------------------
# prepare input
# ------------------------------------------------------------
text = "Ethical AI matters because"
inputs = tokenizer(text, return_tensors="pt")

device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}


# ------------------------------------------------------------
# run baseline forward (before adapters)
# ------------------------------------------------------------
with torch.no_grad():
    base_logits = model(**inputs).logits


# ------------------------------------------------------------
# inspect model layout
# ------------------------------------------------------------
print_layer_map(model, pattern="layers")


# ------------------------------------------------------------
# define adapter config (mid + late layer)
# ------------------------------------------------------------
cfg = AdapterConfig(
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_modules=[
        "model.layers.8.mlp.down_proj",   # mid
        "model.layers.15.mlp.down_proj",  # late
    ],
)


# ------------------------------------------------------------
# inject adapters and move them to same device/dtype
# ------------------------------------------------------------
dtype = next(model.parameters()).dtype
device = next(model.parameters()).device

res = inject_adapters(model, cfg)
model = res.model.to(device=device, dtype=dtype)

print("\nInjected adapters:")
for name in res.injected_layers:
    print(" -", name)


# ------------------------------------------------------------
# re-run forward pass (after adapters)
# ------------------------------------------------------------
with torch.no_grad():
    new_logits = model(**inputs).logits

diff = (new_logits - base_logits).abs().max().item()
print(f"\nMax absolute difference after injection: {diff:.6e} (expected <0.3 for bf16)")

# float 32 difference: 0 
# float 16 difference: 3.125000e-02
# bf16 difference: 2.500000e-01


# ------------------------------------------------------------
# print adapter summary
# ------------------------------------------------------------
print()
print_adapter_summary(model)

