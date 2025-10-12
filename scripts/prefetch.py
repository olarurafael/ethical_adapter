# scripts/prefetch.py
from datasets import load_dataset
# pip install "datasets<3.0.0" , v.2.21.0 used


CFG = [
    {"name": "Anthropic/hh-rlhf", "split": "train"},                          
    {"name": "hendrycks/ethics", "config": "deontology", "split": "train"},   
    {"name": "hendrycks/ethics", "config": "justice", "split": "train"},
    {"name": "hendrycks/ethics", "config": "virtue", "split": "train"},       
    {"name": "hendrycks/ethics", "config": "utilitarianism", "split": "train"},
    {"name": "hendrycks/ethics", "config": "commonsense", "split": "train"},  
    {"name": "allenai/real-toxicity-prompts", "split": "train"},              
    {"name": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train"},  
    {"name": "Skylion007/openwebtext", "split": "train"},                     
]


CACHE_DIR = "./data"

def main():
    for c in CFG:
        print(f"Prefetching {c['name']} {c.get('config','')} {c['split']} -> {CACHE_DIR}")
        load_dataset(c["name"], 
                     c.get("config"), 
                     split=c["split"], 
                     cache_dir=CACHE_DIR,
                     trust_remote_code=True,
                     )

if __name__ == "__main__":
    main()
