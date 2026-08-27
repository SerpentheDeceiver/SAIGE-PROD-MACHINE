"""
Administrative Regulations Extractor for SAIGE ProdMachine

Extracts structured Administrative Knowledge Objects (AKOs) from
regulation PDFs.

This module processes administrative and policy documents (NOT syllabi).

Fixes:
  1. content_end_line no longer bleeds into next section (+50 removed)
  2. Chunks > MAX_CHUNK_CHARS are split into sub-chunks (500-1200 chars)
  3. content_end_line clamped to len(lines) to prevent index errors
  4. _extract_metadata scans ALL program patterns, picks most-mentioned
  5. metadata_list and documents[] are kept in sync (no more misalignment)
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

try:
    import pymupdf  # PyMuPDF
except ImportError:
    print("ERROR: pymupdf not installed. Run: pip install pymupdf")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Chunk size limits (FIX 2) ─────────────────────────────────────────────────
MAX_CHUNK_CHARS = 1200   # hard ceiling per chunk
TARGET_CHUNK_CHARS = 800  # aim for this size

# Feature flags
ENABLE_TITLE_NORMALIZATION = True
ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX = False
ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX = False

# Paths
SCRIPT_DIR = Path(__file__).parent
PROD_ROOT = SCRIPT_DIR.parent
DATA_RAW_DIR = PROD_ROOT / "data" / "raw" / "administrative"
AKOS_DIR = PROD_ROOT / "akos"

# Strong policy keywords
STRONG_POLICY_KEYWORDS = [
    'ELIGIBILITY', 'ADMISSION', 'DURATION', 'WITHDRAWAL', 'REGISTRATION',
    'ENROLLMENT', 'EXAMINATION', 'ASSESSMENT', 'GRADING', 'AWARD',
    'HONOURS', 'EXIT', 'CREDIT FRAMEWORK', 'MULTIPLE ENTRY', 'APPEALS',
    'ACADEMIC COMMITTEE', 'DISCIPLINE', 'ATTENDANCE',
    'PROGRAMME STRUCTURE', 'CURRICULUM FRAMEWORK', 'ACADEMIC BANK',
    'MULTIPLE EXIT', 'AWARD STRUCTURE',
    'NEP REGULATIONS', 'NATIONAL EDUCATION POLICY',
    'ACADEMIC INTEGRITY', 'CODE OF CONDUCT', 'DISCIPLINARY ACTION',
    'ACADEMIC APPEALS BOARD', 'ACADEMIC REGULATIONS', 'GENERAL REGULATIONS',
    'EXAMINATION REGULATIONS', 'ATTENDANCE REGULATIONS', 'DISCIPLINARY REGULATIONS',
    'PROGRAM REGULATIONS', 'COURSE REGULATIONS', 'STUDENT CONDUCT',
]

WEAK_POLICY_KEYWORDS = [
    'POLICY', 'REGULATIONS', 'RULES', 'REQUIREMENTS', 'FRAMEWORK',
    'STRUCTURE', 'OPTIONS', 'CRITERIA', 'PROCESS', 'ACTION',
]

HARD_REJECTION_PATTERNS = [
    r'^\COURSE$', r'^\COURSES$', r'^\CREDIT$', r'^\CREDITS$',
    r'^\POINTS$', r'^\AVERAGE$', r'^\TOTAL$', r'^\HOURS$', r'^\MARKS$',
    r'^ELECTIVE\s*[IVX]*$', r'^CORE\s*\(.+\)$', r'^PROJECT\s*PHASE.*',
    r'^TABLE.*', r'^FIGURE.*', r'^ACADEMIC PRESS.*', r'^REFERENCES$',
    r'^BIBLIOGRAPHY$', r'^PAGE\s+\d+', r'^\d+\s*[-–]\s*\d+$',
    r'^\(\d+\)', r'^\d+\)', r'^\d+\.',
]


def is_section_header(line: str) -> bool:
    """Determine if a line should be considered a section header."""
    line = line.strip()
    line_upper = line.upper()

    if re.match(r'^[\(\)\d\s\-\=\.:\\/\|]+$', line) or \
       re.match(r'.*\d.*HOURS.*$|\d.*MARKS.*$|\d.*CREDITS.*$|\d.*POINTS.*$',
                line, re.IGNORECASE):
        return False

    if len(line) < 10:
        return False

    alphabetic_words = [word for word in re.findall(r'\b[A-Za-z]+\b', line)]
    if len(alphabetic_words) < 2:
        return False

    for pattern in HARD_REJECTION_PATTERNS:
        if re.search(pattern, line_upper, re.IGNORECASE):
            return False

    is_uppercase = line.isupper()
    is_title_case = line.istitle()
    if not (is_uppercase or is_title_case):
        return False

    has_strong_keyword = False
    for keyword in STRONG_POLICY_KEYWORDS:
        if keyword in line_upper:
            if (line_upper.startswith(keyword.split()[0]) or
                (keyword in line_upper and len(line_upper.strip()) <= len(keyword) + 15) and
                not re.search(r'\d+\.\d+', line_upper) and
                not re.search(r'\([a-z]\)', line_upper.lower()) and
                not line_upper.count('.') > 1):
                has_strong_keyword = True
                break

    if has_strong_keyword:
        return True

    has_weak_keyword = any(keyword in line_upper for keyword in WEAK_POLICY_KEYWORDS)
    if has_weak_keyword:
        words = line.split()
        if len(words) >= 3:
            return not re.match(r'^\w+\s+\w+\s+\w+', line, re.IGNORECASE)

    return False


YEAR_PATTERNS = [
    r'(\d{4}[-–]\d{2,4})',
    r'Regulation[s]?\s*[:\-]?\s*(\d{4})',
    r'Academic\s+Year\s*[:\-]?\s*(\d{4}[-–]\d{2,4})',
    r'Effective\s+from\s*[:\-]?\s*(\d{4}[-–]?\d{2,4}?)',
]

PROGRAM_PATTERNS = [
    (r'\bB\.?\s*Tech\b', 'B.Tech'),
    (r'\bM\.?\s*Tech\b', 'M.Tech'),
    (r'\bMBA\b', 'MBA'),
    (r'\bMCA\b', 'MCA'),
    (r'\bM\.?\s*Sc\b', 'M.Sc'),
    (r'\bPhD\b', 'PhD'),
    (r'\bPh\.?\s*D\.?\b', 'PhD'),
]


def split_content_into_chunks(content: str, target: int = TARGET_CHUNK_CHARS,
                               max_size: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    FIX 2: Split large content into chunks of target size.
    Splits on sentence boundaries where possible.
    """
    if len(content) <= max_size:
        return [content]

    # Split into sentences
    sentences = re.split(r'(?<=[.?!])\s+', content.strip())

    chunks = []
    current: list[str] = []
    cur_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # If single sentence exceeds max, split on word boundaries
        if sent_len > max_size:
            words = sent.split()
            piece: list[str] = []
            piece_len = 0
            for w in words:
                if piece_len + len(w) + 1 > max_size and piece:
                    chunks.append(' '.join(piece))
                    piece = [w]
                    piece_len = len(w)
                else:
                    piece.append(w)
                    piece_len += len(w) + 1
            if piece:
                sent = ' '.join(piece)
                sent_len = len(sent)
            else:
                continue

        if cur_len + sent_len + 1 > max_size and current:
            chunks.append(' '.join(current))
            current = []
            cur_len = 0

        current.append(sent)
        cur_len += sent_len + 1

        if cur_len >= target:
            chunks.append(' '.join(current))
            current = []
            cur_len = 0

    if current:
        chunks.append(' '.join(current))

    return [c.strip() for c in chunks if len(c.strip()) >= 20]


@dataclass
class AdministrativeAKO:
    """Administrative Knowledge Object for Regulation Sections"""
    object_type: str = "regulation_section"
    program: str = ""
    regulation_name: str = ""
    regulation_year: Optional[str] = None
    authority: str = "PTU"
    section_title: str = ""
    content: str = ""
    section_category: str = "GENERAL"
    source: Dict[str, str] = None
    chunk_index: int = 1      # NEW: which chunk this is
    chunk_total: int = 1      # NEW: total chunks for this section

    def __post_init__(self):
        if self.source is None:
            self.source = {"pdf": "", "page_range": ""}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.section_title:
            errors.append("Missing section_title")
        if not self.content or len(self.content.strip()) < 10:
            errors.append("Missing or too short content")
        if not self.regulation_name:
            errors.append("Missing regulation_name")
        return len(errors) == 0, errors


class AdministrativeExtractor:
    """Extracts structured regulation sections from administrative PDFs"""

    def __init__(self):
        self.akos: List[AdministrativeAKO] = []

    def extract_from_pdf(self, pdf_path: Path) -> List[AdministrativeAKO]:
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return []

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            doc = pymupdf.open(pdf_path)
            pages_text = []

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages_text.append((page_num, text.strip()))

            doc.close()

            full_text = "\n\n".join([text for _, text in pages_text])

            regulation_name, regulation_year, program = self._extract_metadata(
                full_text, pdf_path.name
            )

            sections = self._extract_sections(
                full_text, pdf_path.name, regulation_name,
                regulation_year, program, pages_text
            )

            logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")
            return sections

        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}", exc_info=True)
            return []

    def _extract_metadata(self, text: str, filename: str) -> Tuple[str, Optional[str], str]:
        """
        FIX 4: Count occurrences of each program keyword and pick the most-mentioned.
        This prevents multi-program PDFs from always returning the first match.
        """
        regulation_year = None
        for pattern in YEAR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                regulation_year = match.group(1)
                break

        # Count mentions of each program
        program_counts: Dict[str, int] = {}
        for pattern, label in PROGRAM_PATTERNS:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                program_counts[label] = program_counts.get(label, 0) + count

        if program_counts:
            program = max(program_counts, key=program_counts.__getitem__)
        else:
            # Fallback to filename
            fn = filename.lower()
            if 'btech' in fn or 'b.tech' in fn:
                program = 'B.Tech'
            elif 'mtech' in fn or 'm.tech' in fn:
                program = 'M.Tech'
            elif 'mba' in fn:
                program = 'MBA'
            elif 'mca' in fn:
                program = 'MCA'
            elif 'msc' in fn:
                program = 'M.Sc'
            elif 'phd' in fn:
                program = 'PhD'
            else:
                program = 'Unknown'

        regulation_name = self._extract_regulation_name(text, filename)
        return regulation_name, regulation_year, program

    def _extract_regulation_name(self, text: str, filename: str) -> str:
        lines = text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if any(keyword in line.upper() for keyword in
                       ['REGULATION', 'RULES', 'POLICY', 'GUIDELINES']):
                    return line

        name = Path(filename).stem
        name = re.sub(r'^\d+\s+', '', name)
        name = re.sub(r'_\d{4}', '', name)
        name = name.replace('_', ' ').title()
        return name

    def _extract_sections(
        self, text: str, pdf_name: str, regulation_name: str,
        regulation_year: Optional[str], program: str,
        pages_text: List[Tuple[int, str]]
    ) -> List[AdministrativeAKO]:
        """
        FIX 1: content_end_line is now exactly next_header_line (no +50 bleed).
        FIX 3: content_end_line clamped to len(lines).
        FIX 2: Large sections are split into chunks after extraction.
        """
        sections = []
        seen_titles = set()
        rejected_headers = set()

        lines = text.split('\n')
        section_positions = []

        for i, line in enumerate(lines):
            if is_section_header(line):
                section_positions.append((i, line.strip()))
            else:
                stripped = line.strip()
                if stripped and len(rejected_headers) < 5:
                    rejected_headers.add(stripped[:50])

        if rejected_headers:
            logger.info(f"Filtered non-policy headers in {pdf_name}: {list(rejected_headers)}")

        if not section_positions:
            logger.warning(f"No section headers found in {pdf_name}")
            content = self._clean_text(text)
            if len(content) > 50:
                ako = AdministrativeAKO(
                    program=program, regulation_name=regulation_name,
                    regulation_year=regulation_year,
                    section_title="GENERAL REGULATIONS", content=content,
                    source={"pdf": pdf_name,
                            "page_range": self._get_page_range(text, pages_text)}
                )
                sections.append(ako)
            return sections

        for i, (line_idx, section_title) in enumerate(section_positions):
            normalized_title = self._normalize_section_title(section_title)

            dedup_key = f"{pdf_name}:{normalized_title}"
            if dedup_key in seen_titles:
                logger.warning(f"  ⚠ Skipped duplicate: {normalized_title}")
                continue
            seen_titles.add(dedup_key)

            content_start_line = line_idx + 1

            # FIX 1 + FIX 3: no +50 bleed, clamped to len(lines)
            if i + 1 < len(section_positions):
                next_header_line = section_positions[i + 1][0]
                content_end_line = min(next_header_line, len(lines))  # FIX 1 & 3
            else:
                content_end_line = len(lines)

            content_lines = lines[content_start_line:content_end_line]

            filtered = []
            for line in content_lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                if line.isupper() and len(line.split()) <= 3:
                    continue
                if re.match(r'^[\(\)\d\s\-\=\.:\\/\|]+$', line):
                    continue
                filtered.append(line)

            section_content = '\n'.join(filtered)
            section_content = self._clean_text(section_content)

            if len(section_content.strip()) < 20:
                continue

            section_category = self._derive_section_category(normalized_title)

            # FIX 2: split large sections into chunks
            chunks = split_content_into_chunks(section_content)
            total = len(chunks)

            for idx, chunk_text in enumerate(chunks, start=1):
                title = normalized_title
                if total > 1:
                    title = f"{normalized_title} (part {idx}/{total})"

                ako = AdministrativeAKO(
                    program=program, regulation_name=regulation_name,
                    regulation_year=regulation_year,
                    section_title=title, content=chunk_text,
                    section_category=section_category,
                    chunk_index=idx, chunk_total=total,
                    source={
                        "pdf": pdf_name,
                        "page_range": self._get_page_range(chunk_text, pages_text)
                    }
                )
                is_valid, errors = ako.validate()
                if is_valid:
                    sections.append(ako)
                    logger.info(f"  ✓ Extracted: {title}")
                else:
                    logger.warning(f"  ✗ Skipped invalid section: {errors}")

        return sections

    def _normalize_section_title(self, title: str) -> str:
        normalized = re.sub(r'^\s*(\d+|[IVX]+)[\.)]?\s*', '', title, flags=re.IGNORECASE)
        normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)
        normalized = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', normalized)
        normalized = re.sub(r'\bTHE([A-Z])', r'THE \1', normalized)
        normalized = re.sub(r'\bOF([A-Z])', r'OF \1', normalized)
        normalized = re.sub(r'\bFOR([A-Z])', r'FOR \1', normalized)
        normalized = re.sub(r'\bAND([A-Z])', r'AND \1', normalized)

        if '(' in normalized and ')' not in normalized:
            normalized = re.sub(r'\s*\(.*', '', normalized)
        normalized = re.sub(r'[()]', '', normalized)

        canonical_map = {
            "AWARD OFGRADES": "AWARD OF GRADES",
            "ELIGIBILITY FORADMISSION": "ELIGIBILITY FOR ADMISSION",
            "DURATION OF THEPROGRAMME": "DURATION OF THE PROGRAMME",
            "SEMESTEREXAMINATION": "SEMESTER EXAMINATION",
            "ACADEMIC APPEALS BOARD (AAB": "ACADEMIC APPEALS BOARD",
            "ACADEMIC APPEALS BOARD AAB": "ACADEMIC APPEALS BOARD",
        }
        normalized_upper = normalized.upper()
        for variant, canonical in canonical_map.items():
            if variant in normalized_upper:
                normalized = canonical
                break

        normalized = normalized.upper()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]$', '', normalized)
        normalized = normalized.strip()

        if len(normalized) <= 5:
            return ""
        return normalized

    def _normalize_section_title_post_processing(self, title: str, content: str) -> str:
        if not ENABLE_TITLE_NORMALIZATION:
            return title
        if len(title) >= 25:
            return title
        title_upper = title.upper()
        if not (title_upper.endswith(' FOR') or title_upper.endswith(' OF') or
                title_upper.endswith(' AND') or title_upper.endswith(' THE')):
            return title
        if (re.search(r'\d', title) or
            any(k in title_upper for k in [
                'EXAMINATION', 'ASSESSMENT', 'AWARD', 'GRADING', 'ELIGIBILITY',
                'ADMISSION', 'DURATION', 'WITHDRAWAL', 'REGISTRATION'
            ])):
            return title

        first_sentence = ""
        for ending in ['. ', '? ', '! ', '\n']:
            if ending in content:
                first_sentence = content.split(ending)[0].strip()
                break
        if not first_sentence:
            first_sentence = content.strip()

        words = first_sentence.split()[:5]
        completion_phrase = ' '.join(words).strip()
        potential_title = f"{title} {completion_phrase}".strip()

        if len(potential_title) > 80:
            return title
        if potential_title != title:
            logger.info(f"[TITLE FIX] '{title}' → '{potential_title}'")
        return potential_title

    def _apply_eligibility_for_admission_fix(self, ako: AdministrativeAKO,
                                              pdf_filename: str) -> AdministrativeAKO:
        if not ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX:
            return ako
        if ako.section_title != "ELIGIBILITY FOR":
            return ako
        if not ako.program or ako.program == "Unknown":
            return ako
        if "Regulations" not in pdf_filename:
            return ako
        content_lower = ako.content.lower()
        if not any(k in content_lower for k in
                   ["admission", "programme", "candidate", "eligibility"]):
            return ako
        original_title = ako.section_title
        ako.section_title = "ELIGIBILITY FOR ADMISSION"
        logger.info(f"[OPTIONAL FIX] '{original_title}' → '{ako.section_title}'")
        return ako

    def _apply_mca_eligibility_cosmetic_fix(self, ako: AdministrativeAKO) -> AdministrativeAKO:
        if not ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX:
            return ako
        if (ako.section_title == "ELIGIBILITY FOR" and
                ako.program == "MCA" and
                ako.source.get("pdf") == "MCA_Regulations_2021.pdf"):
            ako.section_title = "ELIGIBILITY FOR ADMISSION"
            logger.info("[COSMETIC FIX] MCA truncated title normalized")
        return ako

    def _derive_section_category(self, title: str) -> str:
        t = title.upper()
        if 'ELIGIBILITY' in t or 'ADMISSION' in t:
            return 'ELIGIBILITY'
        elif 'EXAMINATION' in t or 'ASSESSMENT' in t:
            return 'EXAMINATION'
        elif 'GRADING' in t or 'GRADE' in t:
            return 'GRADING'
        elif 'CREDIT' in t or 'ACADEMIC BANK' in t:
            return 'CREDIT'
        elif 'EXIT' in t or 'AWARD' in t or 'MULTIPLE' in t:
            return 'EXIT'
        elif 'PROGRAMME' in t or 'STRUCTURE' in t or 'CURRICULUM' in t:
            return 'PROGRAMME'
        else:
            return 'GENERAL'

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Table\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Figure\s+\d+', '', text, flags=re.IGNORECASE)

        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if line and (re.search(r'[A-Za-z]', line) or len(line.split()) > 2):
                cleaned.append(line)
        text = ' '.join(cleaned)  # join with space for embedding

        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        text = text.replace('\u2026', '...')
        return text.strip()

    def _get_page_range(self, content: str, pages_text: List[Tuple[int, str]]) -> str:
        if len(pages_text) == 1:
            return str(pages_text[0][0])
        elif len(pages_text) > 1:
            return f"{pages_text[0][0]}-{pages_text[-1][0]}"
        return "N/A"

    def extract_from_directory(self, dir_path: Path) -> List[AdministrativeAKO]:
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        all_akos = []
        pdf_files = sorted(dir_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")

        for pdf_path in pdf_files:
            akos = self.extract_from_pdf(pdf_path)
            all_akos.extend(akos)

        return all_akos

    def save_akos(self, akos: List[AdministrativeAKO], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if ENABLE_TITLE_NORMALIZATION:
            logger.info("Applying optional title normalization post-processing...")
            for ako in akos:
                ako.section_title = self._normalize_section_title_post_processing(
                    ako.section_title, ako.content
                )

        if ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX:
            for ako in akos:
                self._apply_eligibility_for_admission_fix(
                    ako, ako.source.get("pdf", ""))

        if ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX:
            for ako in akos:
                self._apply_mca_eligibility_cosmetic_fix(ako)

        akos_dict = [ako.to_dict() for ako in akos]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(akos_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(akos)} administrative AKOs to {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("SAIGE ProdMachine - Administrative Regulations Extractor")
    logger.info("=" * 60)

    extractor = AdministrativeExtractor()

    if not DATA_RAW_DIR.exists():
        logger.error(f"Administrative regulations directory not found: {DATA_RAW_DIR}")
        return

    logger.info(f"\nExtracting from: {DATA_RAW_DIR}")
    akos = extractor.extract_from_directory(DATA_RAW_DIR)

    if not akos:
        logger.warning("No sections extracted. Check PDF format and extraction patterns.")
        return

    output_path = AKOS_DIR / "administrative.json"
    extractor.save_akos(akos, output_path)

    logger.info("\n" + "=" * 60)
    logger.info("Administrative Extraction Complete!")
    logger.info(f"Total sections extracted: {len(akos)}")
    logger.info(f"Output: {output_path}")

    # Log chunk size stats
    sizes = [len(a.content) for a in akos]
    oversized = sum(1 for s in sizes if s > MAX_CHUNK_CHARS)
    logger.info(f"Chunk stats — min:{min(sizes)} max:{max(sizes)} avg:{int(sum(sizes)/len(sizes))}")
    logger.info(f"Oversized (>{MAX_CHUNK_CHARS}): {oversized}  ← should be 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()