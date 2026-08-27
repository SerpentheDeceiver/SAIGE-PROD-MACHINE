"""
Institutional AKO Cleanup - Rule-based (Day 1 Implementation, v2.1)
Transforms semantic chunks into standalone, production-ready AKOs

CHANGES IN v2.1:
- cleanup_ako() now returns a shallow copy instead of mutating the input
  dict in place (defensive correctness fix - no behavioral change to a
  correctly-used pipeline, but prevents subtle aliasing bugs if this
  function is ever called on the same list twice, e.g. during testing).
- Records AKO count to the shared pipeline audit trail.
- No change to filtering logic - this stage was NOT where the 51->16 loss
  occurred (confirmed: this stage removed 0 duplicates / 0 invalid in the
  logged run). The loss happens in the next stage,
  llm_title_institutional.py, which previously did not exist in the
  pipeline and has now been reconstructed with a no-drop-on-failure
  guarantee - see that file.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_audit import record_stage  # noqa: E402


def remove_references(text: str, section_title: str) -> str:
    """Remove vague references and make text standalone."""
    text = re.sub(r'\bthis section\b', f'the {section_title} section', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthis program\b', 'the program', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthis course\b', 'the course', text, flags=re.IGNORECASE)

    text = re.sub(r'\bas mentioned (above|earlier|previously)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bas per above\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\brefer to (table|figure|section) \d+\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_bullets(text: str) -> str:
    """Normalize bullet points and formatting."""
    text = re.sub(r'^[•●○►▪▫-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_noise(text: str) -> str:
    """Remove headers, footers, page numbers."""
    text = re.sub(r'\bPage \d+ of \d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+/\d+\b', '', text)
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'_{3,}', '', text)
    return text.strip()


def compute_content_hash(text: str) -> str:
    """Compute hash for deduplication."""
    sample = text[:150].lower().strip()
    return hashlib.md5(sample.encode()).hexdigest()


def cleanup_ako(ako: Dict) -> Dict:
    """Apply all cleanup rules to a single AKO. Returns a NEW dict."""
    ako = dict(ako)  # defensive copy - don't mutate caller's object

    content = ako.get('content', '')
    section_title = ako.get('section_title', 'this section')

    content = remove_noise(content)
    content = remove_references(content, section_title)
    content = normalize_bullets(content)

    ako['content'] = content
    ako['last_updated'] = '2026-01-28'
    ako['cleanup_version'] = 'v2.2_rule_based'
    ako['content_hash'] = compute_content_hash(content)

    return ako


def deduplicate_akos(akos: List[Dict]) -> List[Dict]:
    """Remove near-duplicate AKOs."""
    seen_hashes = set()
    unique_akos = []
    duplicates_removed = 0

    for ako in akos:
        content_hash = ako.get('content_hash', '')

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_akos.append(ako)
        else:
            duplicates_removed += 1

    print(f"Removed {duplicates_removed} duplicate AKOs")
    return unique_akos


def validate_ako(ako: Dict) -> bool:
    """Check if AKO meets quality standards."""
    content = ako.get('content', '')
    word_count = len(content.split())

    if word_count < 30:
        return False
    if not ako.get('section_title'):
        return False
    if not ako.get('category') or ako.get('category') == 'unknown':
        return False

    return True


def cleanup_institutional_akos():
    """Main cleanup pipeline."""
    PROD_ROOT = Path(__file__).resolve().parent.parent.parent
    input_path = PROD_ROOT / 'akos' / 'institutional' / 'institutional.json'
    output_path = PROD_ROOT / 'akos' / 'institutional' / 'institutional_clean.json'

    print("=" * 60)
    print("INSTITUTIONAL AKO CLEANUP - DAY 1")
    print("=" * 60)

    with open(input_path, 'r', encoding='utf-8') as f:
        akos = json.load(f)

    print(f"\nLoaded {len(akos)} AKOs")

    print("Applying cleanup rules...")
    cleaned_akos = [cleanup_ako(ako) for ako in akos]

    print("Deduplicating...")
    unique_akos = deduplicate_akos(cleaned_akos)

    print("Validating...")
    valid_akos = [ako for ako in unique_akos if validate_ako(ako)]
    invalid_count = len(unique_akos) - len(valid_akos)

    print(f"Removed {invalid_count} invalid AKOs")
    print(f"Final count: {len(valid_akos)} AKOs")

    word_counts = [len(ako['content'].split()) for ako in valid_akos]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    optimal_count = sum(1 for wc in word_counts if 75 <= wc <= 500)
    optimal_pct = (optimal_count / len(word_counts) * 100) if word_counts else 0

    print(f"\n{'='*60}")
    print("CLEANUP STATISTICS")
    print(f"{'='*60}")
    print(f"  Total AKOs: {len(valid_akos)}")
    print(f"  Avg words: {avg_words:.1f}")
    print(f"  Optimal length (75-500): {optimal_pct:.1f}%")
    print(f"  Duplicates removed: {len(cleaned_akos) - len(unique_akos)}")
    print(f"  Invalid removed: {invalid_count}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_akos, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {output_path}")
    print("=" * 60)

    record_stage(
        "cleanup",
        input_file=str(input_path),
        output_file=str(output_path),
        input_count=len(akos),
        output_count=len(valid_akos),
        extra={
            "duplicates_removed": len(cleaned_akos) - len(unique_akos),
            "invalid_removed": invalid_count,
        },
    )

    return {
        'total_akos': len(valid_akos),
        'avg_words': avg_words,
        'optimal_pct': optimal_pct,
        'duplicates_removed': len(cleaned_akos) - len(unique_akos),
        'invalid_removed': invalid_count
    }


if __name__ == '__main__':
    stats = cleanup_institutional_akos()
