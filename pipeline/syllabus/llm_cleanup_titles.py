"""
SAIGE – Step 4.2: LLM Cleanup + Title Generation for Syllabus AKOs

What this does:
- Fixes truncated course names using context clues
- Generates clean, short titles for each AKO
- Rewrites unit content into clean standalone AKO text
- Flags low-quality entries (no units, no outcomes)
- Outputs: courses_llm_clean.json

Run: python scripts/llm_cleanup_titles.py
"""

import json
import os
import requests
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
_PROD_ROOT  = Path(__file__).resolve().parent.parent.parent
INPUT_PATH  = str(_PROD_ROOT / "akos" / "syllabus" / "courses.json")
OUTPUT_PATH = str(_PROD_ROOT / "akos" / "syllabus" / "courses_llm_clean.json")
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"     # Aligned with institutional pipeline model
USE_LLM = True              # Set False to run rule-based cleanup only (faster test)
BATCH_SIZE  = 50                 # Log progress every N entries
# ───────────────────────────────────────────────────────────────────────────


def ollama_generate(prompt: str, max_tokens: int = 200) -> str:
    """Call local Ollama LLM and return response text."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1}
        }, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Ollama error] {e}")
    return ""


def fix_truncated_name(course: dict) -> str:
    """Rule-based fix for truncated course names using unit titles as context."""
    name = course.get("course_name", "").strip()
    
    # Already looks complete
    if not name.endswith((" and", " of", " in", " the", " for", " to", " &", " or", " -", "–")):
        return name
    
    # Try to reconstruct from unit titles
    units = course.get("units", [])
    if units:
        unit_titles = [u.get("unit_title", "") for u in units if u.get("unit_title")]
        if unit_titles:
            # Use first unit title as hint — often matches course name
            hint = unit_titles[0].title()
            # If name is a prefix of the hint, return the hint
            if hint.lower().startswith(name.lower()[:15]):
                return hint
    
    # Use textbook refs as hint
    refs = course.get("textbook_references", [])
    if refs and len(refs) > 0:
        ref = refs[0]
        # Sometimes the full course name appears in references
        if len(name) < 30 and name.lower() in ref.lower():
            idx = ref.lower().find(name.lower())
            candidate = ref[idx:idx+60].split(",")[0].strip()
            if len(candidate) > len(name):
                return candidate

    # Flag it but keep original
    return name + " [NEEDS REVIEW]"


def build_ako_title(course: dict) -> str:
    """Generate a clean short title: 'CourseCode – CourseName (Program, Sem X)'"""
    code = course.get("course_code", "").strip()
    name = course.get("course_name", "").strip().rstrip("[NEEDS REVIEW]").strip()
    program = course.get("program", "")
    dept = course.get("department", "")
    sem = course.get("semester", "")
    
    # Remove trailing junk punctuation from name
    name = name.rstrip(" -–&").strip()
    
    title = f"{code} – {name}"
    if program or sem:
        meta = f"{program}, Sem {sem}" if sem else program
        title += f" ({meta})"
    
    return title


def build_ako_text(course: dict) -> str:
    """
    Build a clean standalone AKO text for embedding.
    Each AKO = one course with all its knowledge, self-contained.
    No external references like 'see above' or 'as mentioned'.
    """
    parts = []

    name = course.get("course_name", "").strip().replace(" [NEEDS REVIEW]", "")
    code = course.get("course_code", "")
    program = course.get("program", "")
    degree = course.get("degree_level", "")
    dept = course.get("department", "")
    sem = course.get("semester", "")
    credits = course.get("credits", {}).get("total", 0)

    # Header block
    parts.append(f"Course: {name}")
    parts.append(f"Course Code: {code}")
    parts.append(f"Program: {program} ({degree}) | Department: {dept}")
    parts.append(f"Semester: {sem} | Total Credits: {credits}")

    # Course Outcomes
    outcomes = course.get("course_outcomes", [])
    if outcomes:
        co_texts = [f"CO{i+1}: {o.get('description','')}" 
                    for i, o in enumerate(outcomes) if o.get("description")]
        if co_texts:
            parts.append("Course Outcomes: " + " | ".join(co_texts))

    # Units (the actual knowledge content)
    units = course.get("units", [])
    if units:
        for u in units:
            unit_num = u.get("unit_number", "")
            unit_title = u.get("unit_title", "")
            content = u.get("content", "").strip()
            hours = u.get("hours", "")
            
            if content:
                unit_text = f"Unit {unit_num} – {unit_title}"
                if hours:
                    unit_text += f" ({hours} hrs)"
                unit_text += f": {content}"
                parts.append(unit_text)
    else:
        # No units — add a placeholder note so retrieval still works
        parts.append("Note: Detailed unit content not available for this course.")

    return "\n".join(parts)


def llm_rewrite_ako(raw_text: str, title: str) -> str:
    """
    Use LLM to rewrite the AKO into a clean, standalone, noise-free unit.
    Removes: PDF artifacts, repetition, formatting noise, broken references.
    """
    prompt = f"""You are cleaning a university course knowledge unit for a RAG chatbot.
Rewrite the following course information as a clean, standalone paragraph.
Rules:
- Remove any references like "see above", "as mentioned", "refer to"
- Fix any obvious OCR errors or broken text
- Keep all factual content: course name, code, outcomes, unit topics
- Output must be self-contained — someone reading only this should understand the course
- Do NOT add any information not present in the input
- Keep it concise (under 300 words)

Title: {title}

Raw content:
{raw_text[:1500]}

Cleaned version:"""
    
    result = ollama_generate(prompt, max_tokens=350)
    # Fallback to raw if LLM fails or returns empty
    if not result or len(result) < 50:
        return raw_text
    return result


def compute_quality_flags(course: dict) -> dict:
    """Flag quality issues for the report."""
    flags = []
    
    if not course.get("units"):
        flags.append("NO_UNITS")
    if not course.get("course_outcomes"):
        flags.append("NO_OUTCOMES")
    if not course.get("course_objectives"):
        flags.append("NO_OBJECTIVES")
    
    name = course.get("course_name", "")
    if name.endswith((" and", " of", " in", " the", " for", " to", " &", " or", " -")):
        flags.append("TRUNCATED_NAME")
    if "[NEEDS REVIEW]" in name:
        flags.append("NAME_UNRESOLVED")
    
    credits = course.get("credits", {}).get("total", -1)
    if credits == 0:
        flags.append("ZERO_CREDITS")
    
    return {"quality_flags": flags, "quality_score": max(0, 1.0 - len(flags) * 0.15)}


def main():
    print("=" * 60)
    print("SAIGE – Syllabus AKO Cleanup (Step 4.2)")
    print("=" * 60)

    # Load
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)
    print(f"Loaded {len(courses)} courses from {INPUT_PATH}")

    if USE_LLM:
        # Quick Ollama health check
        try:
            test = ollama_generate("Say OK", max_tokens=5)
            if test:
                print(f"Ollama connected ✓ (model: {OLLAMA_MODEL})")
            else:
                print("WARNING: Ollama returned empty. Falling back to rule-based only.")
        except:
            print("WARNING: Ollama not reachable. Running rule-based cleanup only.")

    cleaned = []
    stats = {"total": len(courses), "truncated_fixed": 0, "llm_cleaned": 0,
             "no_units": 0, "no_outcomes": 0, "flags_total": 0}

    for i, course in enumerate(courses):
        if i % BATCH_SIZE == 0:
            print(f"  Processing {i}/{len(courses)}...")

        # Step 1: Fix truncated name
        original_name = course.get("course_name", "")
        fixed_name = fix_truncated_name(course)
        if fixed_name != original_name:
            stats["truncated_fixed"] += 1
        course["course_name"] = fixed_name

        # Step 2: Generate clean title
        title = build_ako_title(course)

        # Step 3: Build raw AKO text
        raw_ako = build_ako_text(course)

        # Step 4: LLM rewrite (if enabled)
        if USE_LLM:
            ako_text = llm_rewrite_ako(raw_ako, title)
            if ako_text != raw_ako:
                stats["llm_cleaned"] += 1
        else:
            ako_text = raw_ako

        # Step 5: Quality flags
        quality = compute_quality_flags(course)
        if "NO_UNITS" in quality["quality_flags"]:
            stats["no_units"] += 1
        if "NO_OUTCOMES" in quality["quality_flags"]:
            stats["no_outcomes"] += 1
        stats["flags_total"] += len(quality["quality_flags"])

        # Build cleaned entry
        cleaned_entry = {
            **course,
            "ako_title": title,
            "ako_text": ako_text,
            "ako_text_raw": raw_ako,
            **quality
        }
        cleaned.append(cleaned_entry)

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("CLEANUP COMPLETE")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Total AKOs: {stats['total']}")
    print(f"  Truncated names fixed: {stats['truncated_fixed']}")
    print(f"  LLM-rewritten AKOs: {stats['llm_cleaned']}")
    print(f"  Missing units (flagged): {stats['no_units']}")
    print(f"  Missing outcomes (flagged): {stats['no_outcomes']}")
    print(f"  Total quality flags: {stats['flags_total']}")
    print("=" * 60)
    print("Next step: python scripts/dedup_syllabus.py")


if __name__ == "__main__":
    main()
