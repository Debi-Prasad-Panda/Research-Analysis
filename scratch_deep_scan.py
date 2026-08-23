import os, sys, json, re

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

def full_deep_scan(fname):
    filepath = os.path.join(paper_dir, fname)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"==================================================")
    print(f"DEEP SCAN: {fname}")
    print(f"==================================================")
    # search for figures, tables, equations, architecture descriptions
    figs = re.findall(r'(Figure\s+\d+[:\.][^\n]+)', text, re.IGNORECASE)
    print("FIGURES:")
    for fig in figs[:10]:
        print("  -", fig.strip())
    
    tables = re.findall(r'(Table\s+\d+[:\.][^\n]+)', text, re.IGNORECASE)
    print("TABLES:")
    for tab in tables[:10]:
        print("  -", tab.strip())
    print("\n")

for p in sorted(os.listdir(paper_dir)):
    full_deep_scan(p)
