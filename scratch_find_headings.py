import os, sys, json, re

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

# Let's inspect each paper's specific content thoroughly

def get_full_text(fname):
    with open(os.path.join(paper_dir, fname), 'r', encoding='utf-8') as f:
        return f.read()

# Let's write out specific extraction analysis for each paper
for p in sorted(os.listdir(paper_dir)):
    text = get_full_text(p)
    print(f"=== {p} ===")
    print(f"Length: {len(text)} chars")
    # find lines with section numbers or keywords
    matches = re.findall(r'\n([0-9]\s+[A-Z][^\n]+|[I|V|X]+\.\s+[A-Z][^\n]+)', text)
    print("Found headings:", matches[:10])
