import pdfplumber
from pathlib import Path

pdf_path = Path('data/raw/institutional/admission_enrollment.pdf')

with pdfplumber.open(pdf_path) as pdf:
    print(f'Total pages: {len(pdf.pages)}')
    
    for i, page in enumerate(pdf.pages[:2], 1):  # First 2 pages
        text = page.extract_text()
        print(f'\n{"="*60}')
        print(f'PAGE {i} ({len(text)} chars)')
        print(f'{"="*60}')
        print(text[:500])  # First 500 chars
        print('...')
