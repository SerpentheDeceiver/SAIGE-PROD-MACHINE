import sys
sys.path.append('.')
from pathlib import Path
from scripts.extract_institutional import InstitutionalExtractor, DATA_RAW_DIR
import logging

logging.basicConfig(level=logging.DEBUG)

extractor = InstitutionalExtractor()
pdf_path = DATA_RAW_DIR / 'admission_enrollment.pdf'

print(f"\n{'='*60}")
print(f"DIAGNOSING: {pdf_path.name}")
print(f"{'='*60}\n")

akos = extractor.extract_from_pdf(pdf_path)
print(f"\nRESULT: {len(akos)} AKOs extracted")
