"""
SAIGE - Institutional LLM Title Generation (reconstructed, v1.0)

WHY THIS FILE EXISTS:
This pipeline stage was missing from the last code handoff. Evidence of its
existence and behavior was reconstructed from three sources:
  1. build_institutional_faiss.py expects an `llm_title` field on every AKO
     (falling back to section_title if absent) - so a title-generation stage
     must run between cleanup and FAISS build.
  2. dedup_institutional.py ranks AKOs by cleanup_version strings
     "ollama_llm_v2" / "ollama_llm_v1", which don't exist anywhere in
     cleanup_institutional.py's output (that stage tags "v2.2_rule_based").
  3. dedup_institutional.py's BAD_TITLES set includes the literal string
     "6-12 words" - almost certainly a prompt instruction the LLM echoed
     back as if it were the actual title, which someone then had to
     special-case defensively after the fact.

THE CRITICAL FIX (this is the actual root cause of the 51 -> 41 -> 16 loss):
Whatever the original script did on an LLM failure (empty response, parse
failure, too-short output), it appears to have DROPPED the AKO rather than
falling back to the existing section_title. This script guarantees the
opposite: output count == input count, always. LLM success or failure only
changes which title field the AKO ends up with (llm_title vs a rule-based
fallback) and its cleanup_version tag - it never removes a record. Failures
are logged to debug_llm_failures.json for visibility, exactly like before,
but logging a failure is no longer the same thing as losing the AKO.

Run: python pipeline/institutional/llm_title_institutional.py
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_audit import record_stage  # noqa: E402

PROD_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROD_ROOT / "akos" / "institutional" / "institutional_clean.json"
OUTPUT_PATH = PROD_ROOT / "akos" / "institutional" / "institutional_titled.json"
DEBUG_LOG_PATH = PROD_ROOT / "akos" / "institutional" / "debug_llm_failures.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # Aligned with syllabus pipeline model
REQUEST_TIMEOUT = 60
MAX_RETRIES = 1

# Deliberately does NOT instruct "6-12 words" verbatim in the prompt - the
# BAD_TITLES entry in dedup_institutional.py strongly suggests the original
# prompt's exact instruction text leaked into LLM output as a literal title
# on failure. Phrasing the constraint differently (and validating the
# output shape rather than trusting it) avoids reproducing that failure mode.
TITLE_PROMPT_TEMPLATE = """Write one short, specific title for the institutional policy text below.
The title must describe what the text is actually about (e.g. "Hostel Room Allocation Process", not a generic label like "Institutional Information").
Respond with ONLY the title itself - no quotes, no explanation, no preamble.

Category: {category}
Text:
{content}

Title:"""

MIN_TITLE_WORDS = 2
MAX_TITLE_WORDS = 12
GENERIC_TITLE_PATTERNS = [
    r'^\s*$',
    r'^title\s*:?\s*$',
    r'^\d+[\-\s]\d+\s*words?$',       # catches "6-12 words" style echoes
    r'^(institutional|general)\s+information$',
]


def ollama_generate(prompt: str, max_tokens: int = 40) -> str:
    """Call local Ollama LLM and return response text. Empty string on any failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.2},
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Ollama error] {e}")
    return ""


def clean_title_candidate(raw: str) -> str:
    """Strip quotes/preamble artifacts the model sometimes adds despite instructions."""
    t = raw.strip()
    t = t.strip('"\'')
    t = re.sub(r'^(title\s*:?\s*)', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def is_valid_title(title: str) -> bool:
    """
    Reject the two failure modes that previously caused silent data loss:
    empty/too-short output, and the model echoing prompt-instruction text
    back as if it were the title (e.g. a literal "6-12 words").
    """
    if not title:
        return False

    word_count = len(title.split())
    if word_count < MIN_TITLE_WORDS or word_count > MAX_TITLE_WORDS:
        return False

    lowered = title.lower()
    for pattern in GENERIC_TITLE_PATTERNS:
        if re.match(pattern, lowered):
            return False

    return True


def generate_llm_title(ako: Dict) -> Tuple[str, str, str]:
    """
    Returns (title, cleanup_version, failure_reason_or_empty).
    NEVER returns an empty title - falls back to section_title on any
    failure, so the caller always has something usable and the AKO is
    never a candidate for being dropped downstream because of this step.
    """
    content = ako.get("content", "")
    category = ako.get("category", "unknown")
    section_title = ako.get("section_title", "Institutional Information")

    prompt = TITLE_PROMPT_TEMPLATE.format(category=category, content=content[:1200])

    last_reason = ""
    for attempt in range(MAX_RETRIES + 1):
        raw = ollama_generate(prompt)
        if not raw:
            last_reason = "empty_response"
            continue

        try:
            candidate = clean_title_candidate(raw)
        except Exception:
            last_reason = "parse_error"
            continue

        if not is_valid_title(candidate):
            word_count = len(candidate.split())
            if word_count < MIN_TITLE_WORDS:
                last_reason = f"too_short_{word_count}_words"
            elif word_count > MAX_TITLE_WORDS:
                last_reason = f"too_long_{word_count}_words"
            else:
                last_reason = "generic_or_echoed_instruction"
            continue

        return candidate, "ollama_llm_v2", ""

    # Every attempt failed - fall back to the rule-based title we already
    # have. This is the load-bearing fix: the AKO survives regardless.
    return section_title, "rule_based_fallback", last_reason or "unknown_failure"


def main():
    print("=" * 60)
    print("SAIGE - Institutional LLM Title Generation")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"ERROR: input not found: {INPUT_PATH}")
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        akos: List[Dict] = json.load(f)

    print(f"Loaded {len(akos)} AKOs from {INPUT_PATH}")

    # Quick Ollama health check, matching the syllabus pipeline's pattern.
    test = ollama_generate("Say OK", max_tokens=5)
    if test:
        print(f"Ollama connected ✓ (model: {OLLAMA_MODEL})")
    else:
        print("WARNING: Ollama not reachable or returned empty. "
              "All AKOs will use rule-based title fallback (none will be dropped).")

    titled_akos: List[Dict] = []
    failures: List[Dict] = []
    llm_successes = 0
    llm_fallbacks = 0

    for i, ako in enumerate(akos):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(akos)}...")

        title, cleanup_version, failure_reason = generate_llm_title(ako)

        new_ako = dict(ako)
        new_ako["llm_title"] = title
        new_ako["cleanup_version"] = cleanup_version

        if cleanup_version == "ollama_llm_v2":
            llm_successes += 1
        else:
            llm_fallbacks += 1
            failures.append({
                "ako_id": ako.get("ako_id", "unknown"),
                "section_title": ako.get("section_title", ""),
                "reason": failure_reason,
                "fallback_title_used": title,
            })

        # No `continue`, no filtering, no dropping — every input AKO
        # produces exactly one output AKO. This is the fix.
        titled_akos.append(new_ako)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(titled_akos, f, indent=2, ensure_ascii=False)

    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("LLM TITLE GENERATION COMPLETE")
    print(f"  Total input: {len(akos)}")
    print(f"  Total output: {len(titled_akos)}  (must equal input — verified below)")
    print(f"  LLM successes: {llm_successes}")
    print(f"  Rule-based fallbacks: {llm_fallbacks}")
    print(f"  Debug log saved to: {DEBUG_LOG_PATH}")
    print(f"  Output saved to: {OUTPUT_PATH}")
    print("=" * 60)

    assert len(titled_akos) == len(akos), (
        f"INVARIANT VIOLATED: output count ({len(titled_akos)}) != input count "
        f"({len(akos)}). This should be impossible — every AKO is appended "
        f"exactly once regardless of LLM outcome. If this fires, it's a new bug, "
        f"not the original one."
    )

    record_stage(
        "llm_title",
        input_file=str(INPUT_PATH),
        output_file=str(OUTPUT_PATH),
        input_count=len(akos),
        output_count=len(titled_akos),
        extra={"llm_successes": llm_successes, "llm_fallbacks": llm_fallbacks},
    )

    print("\nNext step: python pipeline/institutional/dedup_institutional.py")


if __name__ == "__main__":
    main()
