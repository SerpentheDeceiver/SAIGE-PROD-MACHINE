import sys
sys.path.append('.')
from pathlib import Path
from scripts.extract_institutional import InstitutionalExtractor, DATA_RAW_DIR, MIN_WORDS_PER_AKO
import pdfplumber

extractor = InstitutionalExtractor()
pdf_path = DATA_RAW_DIR / 'admission_enrollment.pdf'

with pdfplumber.open(pdf_path) as pdf:
    full_text = ''
    for page in pdf.pages:
        full_text += page.extract_text() + '\n\n'

# Clean text
full_text = extractor._clean_text(full_text)
print(f'Total text: {len(full_text)} chars, {len(full_text.split())} words\n')

# Try semantic splitting
sections = extractor._split_by_semantic_boundaries(full_text)
print(f'Sections created: {len(sections)}\n')

for i, (title, content) in enumerate(sections, 1):
    word_count = len(content.split())
    status = '✓' if word_count >= MIN_WORDS_PER_AKO else '✗'
    print(f'{status} Section {i}: {word_count} words - {title[:60]}')
