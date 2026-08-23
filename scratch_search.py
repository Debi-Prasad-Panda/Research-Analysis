import os, sys, re, json

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

def search_text(fname, patterns):
    filepath = os.path.join(paper_dir, fname)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f'=== {fname} ===')
    for pat in patterns:
        matches = [m.start() for m in re.finditer(pat, text, re.IGNORECASE)]
        print(f'Pattern "{pat}": found {len(matches)} occurrences')
        for pos in matches[:2]:
            snippet = text[max(0, pos-80):min(len(text), pos+250)].replace('\n', ' ')
            print(f'   -> {snippet}')
    print('-'*50)

search_text('NIPS-2017-attention-is-all-you-need-Paper.txt', ['Scaled Dot-Product', 'Multi-Head Attention', 'Positional Encoding', 'Position-wise Feed-Forward'])
search_text('Mamba Linear Time Sequence Modeling with Selective State Spaces.txt', ['Algorithm 1', 'Selective Scan', 'Hardware-aware', '2.2 State Space Models'])
search_text('Mixture of Depths Dynamically allocating.txt', ['Routing', 'capacity', 'top-k', 'MoD Transformer', 'Residual'])
search_text('A Comprehensive Survey of Mixture of Experts.txt', ['Gating', 'Expert Network', 'Loss function', 'Routing strategy', 'DeepSeek'])
search_text('Sliding Window Attention Training for Efficient Large Language Models.txt', ['SWAT', 'decay', 'cache', 'Algorithm', 'Receptive field'])
search_text('Titans Learning to Memorize at Test Time.txt', ['Neural Memory', 'MAC', 'MAG', 'MAL', 'Surprise', 'Momentum'])
search_text('TransMamba A Sequence-Level Hybrid Transformer-Mamba.txt', ['Sequence-Level', 'Chunk', 'Router', 'Architecture', 'Hidden State'])
