import os, sys, json

sys.stdout.reconfigure(encoding='utf-8')
paper_dir = r'a:\Downloads\24th OITS\understanding\extracted_papers'

def summarize_file(fname):
    with open(os.path.join(paper_dir, fname), 'r', encoding='utf-8') as f:
        text = f.read()
    print(f'================ {fname} ================')
    print(f'Length: {len(text)} characters')
    # print first 1000 characters
    print('--- START ---')
    print(text[:1200])
    print('--- END PREVIEW ---')

for p in sorted(os.listdir(paper_dir)):
    summarize_file(p)
