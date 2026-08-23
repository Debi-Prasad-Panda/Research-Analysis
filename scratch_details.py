import os, sys, re, json

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

def print_paper_details(fname, keywords):
    with open(os.path.join(paper_dir, fname), 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"\n=======================================================")
    print(f"PAPER: {fname}")
    print(f"=======================================================")
    for kw in keywords:
        pos = text.lower().find(kw.lower())
        if pos != -1:
            print(f"--- MATCH for '{kw}' (index {pos}) ---")
            print(text[max(0, pos-100):min(len(text), pos+1200)])
            print("---------------------------------------")

print_paper_details("NIPS-2017-attention-is-all-you-need-Paper.txt", ["model architecture", "scaled dot-product", "multi-head attention", "position-wise", "positional encoding"])
print_paper_details("Mamba Linear Time Sequence Modeling with Selective State Spaces.txt", ["selective state space", "algorithm 1", "hardware-aware", "figure 3", "2.2 state space models"])
print_paper_details("Mixture of Depths Dynamically allocating.txt", ["implementing mixture-of-depths", "routing", "capacity", "top-k", "mod transformer"])
print_paper_details("Sliding Window Attention Training for Efficient Large Language Models.txt", ["sliding window attention training", "sigmoid", "alibi", "rope", "algorithm"])
print_paper_details("Titans Learning to Memorize at Test Time.txt", ["learning to memorize at test time", "neural memory", "memory as a context", "memory as a gate", "memory as a layer", "surprise"])
print_paper_details("TransMamba A Sequence-Level Hybrid Transformer-Mamba.txt", ["2 method", "transpoint", "memory compression", "figure 3", "shared parameters"])
