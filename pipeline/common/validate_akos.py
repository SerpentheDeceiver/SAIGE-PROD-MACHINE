#!/usr/bin/env python3
"""
Atomic Knowledge Object (AKO) Validation Script for SAIGE

This script validates institutional knowledge chunks to ensure they are atomic,
non-repetitive, and properly sized for optimal RAG performance.

Run this on ProdMachine BEFORE indexing to detect bad AKOs.

Usage:
    python validate_akos.py <input_chunks_file.json>
    
Input format (JSON):
    [
        {"text": "chunk content", "id": "chunk_001", "metadata": {...}},
        ...
    ]
"""

import re
import json
import sys
from typing import List, Tuple, Dict

# ============================================================
# Validation Rules (aligned with ServerMachine config)
# ============================================================

MIN_AKO_CHARS = 200
MAX_AKO_CHARS = 800  # Must match MAX_CHUNK_CHARS in rag_config.py
MAX_BULLET_POINTS = 3  # More than 3 bullets = likely over-merged

# Topic pairs that should NOT appear together (indicates mixing)
MIXED_TOPIC_KEYWORDS = [
    ("admission", "fee"),
    ("admission", "placement"),
    ("fee", "hostel"),
    ("library", "sports"),
    ("exam", "attendance"),
    ("syllabus", "timetable"),
]


def validate_ako(chunk: str, chunk_id: str) -> Tuple[bool, List[str]]:
    """
    Validate if a chunk is a good Atomic Knowledge Object.
    
    Args:
        chunk: The text content of the chunk
        chunk_id: Unique identifier for the chunk
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # ----------------------------------------------------------------
    # Rule 1: Size constraint
    # ----------------------------------------------------------------
    if len(chunk) < MIN_AKO_CHARS:
        issues.append(f"Too short ({len(chunk)} < {MIN_AKO_CHARS} chars)")
    if len(chunk) > MAX_AKO_CHARS:
        issues.append(f"Too long ({len(chunk)} > {MAX_AKO_CHARS} chars)")
    
    # ----------------------------------------------------------------
    # Rule 2: Bullet point check (indicator of over-merging)
    # ----------------------------------------------------------------
    bullet_count = chunk.count('\n-') + chunk.count('\n•') + chunk.count('\n*')
    if bullet_count > MAX_BULLET_POINTS:
        issues.append(
            f"Too many bullet points ({bullet_count} > {MAX_BULLET_POINTS}). "
            "Likely over-merged. Split into separate AKOs."
        )
    
    # ----------------------------------------------------------------
    # Rule 3: Mixed topic detection
    # ----------------------------------------------------------------
    chunk_lower = chunk.lower()
    for topic1, topic2 in MIXED_TOPIC_KEYWORDS:
        if topic1 in chunk_lower and topic2 in chunk_lower:
            # Check if they're not part of compound terms
            if not (f"{topic1} {topic2}" in chunk_lower or f"{topic2} {topic1}" in chunk_lower):
                issues.append(f"Mixed topics detected: '{topic1}' and '{topic2}'")
    
    # ----------------------------------------------------------------
    # Rule 4: Repeated phrases (indicator of copy-paste merging)
    # ----------------------------------------------------------------
    sentences = re.split(r'[.!?]', chunk)
    if len(sentences) > 2:
        for i, sent1 in enumerate(sentences[:-1]):
            for sent2 in sentences[i+1:]:
                # Simple similarity check (more than 50% word overlap)
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())
                if len(words1) > 5 and len(words2) > 5:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if overlap > 0.5:
                        issues.append("Repeated/similar sentences detected")
                        break
            if "Repeated" in str(issues):
                break
    
    # ----------------------------------------------------------------
    # Rule 5: Empty or whitespace-only content
    # ----------------------------------------------------------------
    if not chunk.strip():
        issues.append("Empty or whitespace-only content")
    
    return (len(issues) == 0, issues)


def validate_ako_dataset(chunks: List[Dict]) -> Dict:
    """
    Validate entire institutional dataset.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and 'id' keys
    
    Returns:
        Validation report dictionary
    """
    total = len(chunks)
    valid = 0
    invalid = 0
    issues_by_type = {}
    invalid_chunks = []
    
    print(f"Validating {total} institutional AKOs...\n")
    print("=" * 80)
    
    for chunk_dict in chunks:
        chunk_text = chunk_dict.get('text', '')
        chunk_id = chunk_dict.get('id', 'unknown')
        
        is_valid, issues = validate_ako(chunk_text, chunk_id)
        
        if is_valid:
            valid += 1
        else:
            invalid += 1
            invalid_chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'issues': issues
            })
            
            print(f"❌ INVALID AKO: {chunk_id}")
            print(f"   Chunk preview: {chunk_text[:100]}...")
            for issue in issues:
                print(f"   - {issue}")
                # Count issue types
                issue_type = issue.split('(')[0].strip()
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
            print()
    
    print("=" * 80)
    print(f"\nVALIDATION SUMMARY:")
    print(f"  Total AKOs: {total}")
    print(f"  ✓ Valid: {valid} ({valid/total*100:.1f}%)")
    print(f"  ✗ Invalid: {invalid} ({invalid/total*100:.1f}%)")
    print()
    
    if issues_by_type:
        print("Issues by type:")
        for issue_type, count in sorted(issues_by_type.items(), key=lambda x: -x[1]):
            print(f"  - {issue_type}: {count}")
    
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "issues": issues_by_type,
        "invalid_chunks": invalid_chunks
    }


def main():
    """Main entry point for AKO validation."""
    if len(sys.argv) < 2:
        print("Usage: python validate_akos.py <input_chunks_file.json>")
        print("\nInput format (JSON):")
        print('  [{"text": "chunk content", "id": "chunk_001"}, ...]')
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    if not isinstance(chunks, list):
        print("Error: Input must be a JSON array of chunk objects")
        sys.exit(1)
    
    # Validate dataset
    report = validate_ako_dataset(chunks)
    
    # Save report
    output_file = input_file.replace('.json', '_validation_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Validation report saved to: {output_file}")
    
    # Exit with error code if validation failed
    if report['invalid'] > 0:
        print(f"\n⚠ WARNING: {report['invalid']} invalid AKOs detected.")
        print("Please fix these issues before indexing.")
        sys.exit(1)
    else:
        print("\n✓ All AKOs are valid!")
        sys.exit(0)


if __name__ == "__main__":
    # Example usage for testing
    if len(sys.argv) == 1:
        print("Running example validation...\n")
        
        test_chunks = [
            {
                "text": "Library Facilities: The central library houses over 50,000 books, 200+ journals, and digital resources. Open 9 AM to 9 PM on weekdays, 9 AM to 5 PM on weekends. Students can borrow up to 5 books at a time.",
                "id": "inst_001"
            },
            {
                "text": "Admission Process: Applications open in May. Fee structure: Rs. 50,000 per semester. Placement: 85% rate.",
                "id": "inst_002"  # Bad: mixed topics
            },
            {
                "text": "Sports: Cricket, Football, Basketball, Tennis, Badminton, Table Tennis, Volleyball, Athletics",
                "id": "inst_003"  # Bad: list without context
            },
            {
                "text": "Too short",
                "id": "inst_004"  # Bad: too short
            },
        ]
        
        report = validate_ako_dataset(test_chunks)
        
        print("\n" + "=" * 80)
        print("Example validation complete. Use with real data:")
        print("  python validate_akos.py institutional_chunks.json")
    else:
        main()
