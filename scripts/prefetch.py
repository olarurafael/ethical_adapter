# scripts/prefetch.py
from datasets import load_dataset
# pip install "datasets<3.0.0" , v.2.21.0 used


CFG = [
    # {"name": "Anthropic/hh-rlhf", "split": "train"},                          
    # {"name": "hendrycks/ethics", "config": "deontology", "split": "train"},   
    # {"name": "hendrycks/ethics", "config": "justice", "split": "train"},
    # {"name": "hendrycks/ethics", "config": "virtue", "split": "train"},       
    # {"name": "hendrycks/ethics", "config": "utilitarianism", "split": "train"},
    # {"name": "hendrycks/ethics", "config": "commonsense", "split": "train"},  
    # {"name": "allenai/real-toxicity-prompts", "split": "train"},              
    # {"name": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train"},  
    # {"name": "Skylion007/openwebtext", "split": "train"}, 
    # {"name": "bigcode/the-stack-smol", "split": "train"}, # code
    # {"name": "math_dataset", "config": "numbers__is_prime", "split": "train"}, # math
    # {"name": "dream", "split": "train"}, # logical reasoning
    # {"name": "glue", "config": "mrpc", "split": "train"},   # paraphrase
    # {"name": "yelp_review_full", "split": "train"}, # sentiment analysis
    {"name": "google/boolq", "split": "train"},
    {"name": "glue", "config": "sst2", "split": "train"},
    {"name": "civil_comments", "split": "train"},
    {"name": "unified_toxicity_annotations", "split": "train"},

                
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
