import os, sys, re

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

papers = sorted(os.listdir(paper_dir))

for p in papers:
    print('================================================================')
    print('PAPER:', p)
    print('================================================================')
    with open(os.path.join(paper_dir, p), 'r', encoding='utf-8') as f:
        text = f.read()
    
    # find section titles
    lines = text.split('\n')
    sec_lines = []
    for line in lines:
        line_s = line.strip()
        if re.match(r'^(?:[0-9IVX]+\.?|[0-9]+\.[0-9]+)\s+[A-Z]', line_s) and len(line_s) < 80:
            sec_lines.append(line_s)
    print('SECTIONS:')
    for s in sec_lines[:25]:
        print('  *', s)
    print()
