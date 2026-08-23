import os, sys, re, json

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

def read_full_paper(pname):
    with open(os.path.join(paper_dir, pname), 'r', encoding='utf-8') as f:
        return f.read()

# Let's inspect details of each paper
print("=== 1. Attention Is All You Need ===")
t1 = read_full_paper("NIPS-2017-attention-is-all-you-need-Paper.txt")
# Find Section 3 Model Architecture
print(t1[t1.find("3 Model Architecture"):t1.find("4 Why Self-Attention")])

print("=== 2. Mamba ===")
t2 = read_full_paper("Mamba Linear Time Sequence Modeling with Selective State Spaces.txt")
print(t2[t2.find("3 Selective State Space Models"):t2.find("4 Empirical Evaluation")][:2500])

print("=== 3. Mixture of Depths ===")
t3 = read_full_paper("Mixture of Depths Dynamically allocating.txt")
print(t3[t3.find("3 Implementing Mixture-of-Depths Transformers"):t3.find("4 Results")][:2500])

print("=== 4. Mixture of Experts Survey ===")
t4 = read_full_paper("A Comprehensive Survey of Mixture of Experts.txt")
print(t4[t4.find("II. BASIC DESIGNS OF MOE"):t4.find("III. ALGORITHMS")][:2500])

print("=== 5. Sliding Window Attention Training (SWAT) ===")
t5 = read_full_paper("Sliding Window Attention Training for Efficient Large Language Models.txt")
print(t5[t5.find("3 Sliding Window Attention Training"):t5.find("4 Experiments")][:2500])

print("=== 6. Titans ===")
t6 = read_full_paper("Titans Learning to Memorize at Test Time.txt")
print(t6[t6.find("3 Learning to Memorize at Test Time"):t6.find("5 Experiments")][:2500])

print("=== 7. TransMamba ===")
t7 = read_full_paper("TransMamba A Sequence-Level Hybrid Transformer-Mamba.txt")
print(t7[t7.find("2 Method"):t7.find("3 Experiments")][:2500])
