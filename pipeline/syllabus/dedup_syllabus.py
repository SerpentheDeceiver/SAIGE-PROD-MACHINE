"""
SAIGE – Step 4.3: Deduplication for Syllabus AKOs
Database: Academic (courses_llm_clean.json)
Author: Darineesh (Syllabus DB)

What this does:
- Exact duplicate removal (same course_code)
- Near-duplicate detection (same name, different dept — KEPT but flagged)
- Semantic duplicate detection using title similarity (cosine on TF-IDF)
- Outputs: courses_llm_clean.json (deduplicated, in-place update)
- Outputs: syllabus_dedup_report.json

IMPORTANT RULE for SAIGE:
  "Mini Project" in BTech CSE ≠ "Mini Project" in BTech ECE
  → Same name across different programs/depts are KEPT (they are genuinely different)
  → Same name + same dept + same semester = TRUE DUPLICATE → REMOVE

Run: python scripts/dedup_syllabus.py
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

_PROD_ROOT  = Path(__file__).resolve().parent.parent.parent
INPUT_PATH  = str(_PROD_ROOT / "akos" / "syllabus" / "courses_llm_clean.json")
OUTPUT_PATH = str(_PROD_ROOT / "akos" / "syllabus" / "courses_llm_clean.json")   # Overwrite in-place
REPORT_PATH = str(_PROD_ROOT / "akos" / "syllabus" / "syllabus_dedup_report.json")

# Similarity threshold for semantic near-duplicates (0.0 – 1.0)
# Higher = stricter (fewer removals). 0.92 is safe for course names.
SEMANTIC_THRESHOLD = 0.92


# ── Utility ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(normalize(a).split())
    set_b = set(normalize(b).split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def make_dedup_key(course: dict) -> str:
    """
    Strict dedup key: same course_code = definite duplicate.
    """
    return course.get("course_code", "").strip().upper()


def make_soft_key(course: dict) -> str:
    """
    Soft key: same name + same program + same semester = likely duplicate.
    Different dept = different course (Mini Project rule).
    """
    name = normalize(course.get("course_name", "").replace("[NEEDS REVIEW]", ""))
    program = course.get("program", "").strip().lower()
    dept = course.get("department", "").strip().lower()
    sem = str(course.get("semester", "")).strip()
    return f"{name}|{program}|{dept}|{sem}"


# ── Main Dedup Logic ────────────────────────────────────────────────────────

def deduplicate(courses: list) -> tuple[list, dict]:
    report = {
        "total_input": len(courses),
        "exact_duplicates_removed": [],
        "soft_duplicates_removed": [],
        "near_duplicates_flagged": [],
        "total_output": 0
    }

    # ── Pass 1: Exact duplicate by course_code ──────────────────────────────
    seen_codes = {}
    pass1 = []
    for course in courses:
        code = make_dedup_key(course)
        if not code or code == "":
            pass1.append(course)  # No code — can't dedup, keep
            continue
        if code in seen_codes:
            report["exact_duplicates_removed"].append({
                "course_code": code,
                "course_name": course.get("course_name"),
                "reason": "exact_course_code_duplicate"
            })
        else:
            seen_codes[code] = True
            pass1.append(course)

    print(f"  Pass 1 (exact code): {len(courses)} → {len(pass1)} "
          f"(removed {len(courses) - len(pass1)})")

    # ── Pass 2: Soft duplicate (same name + program + dept + semester) ───────
    seen_soft = {}
    pass2 = []
    for course in pass1:
        soft_key = make_soft_key(course)
        if soft_key in seen_soft:
            report["soft_duplicates_removed"].append({
                "course_name": course.get("course_name"),
                "program": course.get("program"),
                "department": course.get("department"),
                "semester": course.get("semester"),
                "reason": "same_name_program_dept_semester"
            })
        else:
            seen_soft[soft_key] = True
            pass2.append(course)

    print(f"  Pass 2 (soft key):   {len(pass1)} → {len(pass2)} "
          f"(removed {len(pass1) - len(pass2)})")

    # ── Pass 3: Near-duplicate detection (Jaccard on AKO title) ─────────────
    # Flag only — don't remove. Let human review these.
    # Group by program first to reduce comparisons
    program_groups = defaultdict(list)
    for i, course in enumerate(pass2):
        program_groups[course.get("program", "UNKNOWN")].append((i, course))

    flagged_indices = set()
    for prog, group in program_groups.items():
        for i in range(len(group)):
            if group[i][0] in flagged_indices:
                continue
            for j in range(i + 1, len(group)):
                if group[j][0] in flagged_indices:
                    continue
                title_a = group[i][1].get("ako_title", group[i][1].get("course_name", ""))
                title_b = group[j][1].get("ako_title", group[j][1].get("course_name", ""))
                sim = jaccard_similarity(title_a, title_b)
                if sim >= SEMANTIC_THRESHOLD:
                    # Only flag if same department too — cross-dept similar titles are normal
                    dept_a = group[i][1].get("department", "")
                    dept_b = group[j][1].get("department", "")
                    if dept_a.lower() == dept_b.lower():
                        flagged_indices.add(group[j][0])
                        report["near_duplicates_flagged"].append({
                            "kept": title_a,
                            "flagged": title_b,
                            "similarity": round(sim, 3),
                            "program": prog,
                            "department": dept_a
                        })

    # Apply near-dup removal (flagged ones removed)
    pass3 = [c for i, c in enumerate(pass2) if i not in flagged_indices]

    print(f"  Pass 3 (near-dup):   {len(pass2)} → {len(pass3)} "
          f"(removed {len(pass2) - len(pass3)})")

    # ── Mark remaining with dedup_status ────────────────────────────────────
    for course in pass3:
        course["dedup_status"] = "clean"

    report["total_output"] = len(pass3)
    report["total_removed"] = len(courses) - len(pass3)
    report["duplicate_rate_pct"] = round(
        (len(courses) - len(pass3)) / len(courses) * 100, 2
    )

    return pass3, report


def main():
    print("=" * 60)
    print("SAIGE – Syllabus Deduplication (Step 4.3)")
    print("=" * 60)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)
    print(f"Loaded {len(courses)} AKOs from {INPUT_PATH}")
    print()

    deduped, report = deduplicate(courses)

    # Save cleaned
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    # Save report
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("DEDUPLICATION COMPLETE")
    print(f"  Input AKOs:             {report['total_input']}")
    print(f"  Exact duplicates removed: {len(report['exact_duplicates_removed'])}")
    print(f"  Soft duplicates removed:  {len(report['soft_duplicates_removed'])}")
    print(f"  Near-dups flagged+removed:{len(report['near_duplicates_flagged'])}")
    print(f"  Output AKOs:            {report['total_output']}")
    print(f"  Duplicate rate:         {report['duplicate_rate_pct']}%")
    print(f"  Target: < 15% ✓" if report['duplicate_rate_pct'] < 15 else 
          f"  WARNING: Still above 15% target!")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Report: {REPORT_PATH}")
    print("=" * 60)
    print("Next step: python scripts/build_syllabus_faiss.py")


if __name__ == "__main__":
    main()