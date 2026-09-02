"""
Administrative Regulations Extractor for SAIGE ProdMachine

Extracts structured Administrative Knowledge Objects (AKOs) from regulation PDFs.
This module processes administrative and policy documents (NOT syllabi).

Author: SAIGE NLP Engineering Team
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

# Feature flag for optional title normalization
ENABLE_TITLE_NORMALIZATION = True

# Feature flag for specific "ELIGIBILITY FOR" fix (disabled by default)
ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX = False

# Feature flag for MCA-specific cosmetic fix (disabled by default)
ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX = False

# Paths
SCRIPT_DIR = Path(__file__).parent
PROD_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATA_RAW_DIR = PROD_ROOT / "data" / "raw" / "administrative"
AKOS_DIR = PROD_ROOT / "akos"

# Strong policy keywords (only these qualify as MAIN section headers)
STRONG_POLICY_KEYWORDS = [
    # Core academic policy sections
    'ELIGIBILITY', 'ADMISSION', 'DURATION', 'WITHDRAWAL', 'REGISTRATION',
    'ENROLLMENT', 'EXAMINATION', 'ASSESSMENT', 'GRADING', 'AWARD',
    'HONOURS', 'EXIT', 'CREDIT FRAMEWORK', 'MULTIPLE ENTRY', 'APPEALS',
    'ACADEMIC COMMITTEE', 'DISCIPLINE', 'ATTENDANCE',

    # Program structure keywords (policy-related only)
    'PROGRAMME STRUCTURE', 'CURRICULUM FRAMEWORK', 'ACADEMIC BANK',
    'MULTIPLE EXIT', 'AWARD STRUCTURE',

    # NEP and regulatory keywords
    'NEP REGULATIONS', 'NATIONAL EDUCATION POLICY',

    # Academic integrity and conduct
    'ACADEMIC INTEGRITY', 'CODE OF CONDUCT', 'DISCIPLINARY ACTION',

    # Major administrative sections
    'ACADEMIC APPEALS BOARD', 'ACADEMIC REGULATIONS', 'GENERAL REGULATIONS',
    'EXAMINATION REGULATIONS', 'ATTENDANCE REGULATIONS', 'DISCIPLINARY REGULATIONS',
    'PROGRAM REGULATIONS', 'COURSE REGULATIONS', 'STUDENT CONDUCT',
]

# Weak keywords that can only qualify if line has strong semantic content
WEAK_POLICY_KEYWORDS = [
    'POLICY', 'REGULATIONS', 'RULES', 'REQUIREMENTS', 'FRAMEWORK',
    'STRUCTURE', 'OPTIONS', 'CRITERIA', 'PROCESS', 'ACTION',
]

# Hard rejection patterns (case-insensitive)
HARD_REJECTION_PATTERNS = [
    r'^COURSE$', r'^COURSES$', r'^CREDIT$', r'^CREDITS$',
    r'^POINTS$', r'^AVERAGE$', r'^TOTAL$', r'^HOURS$', r'^MARKS$',
    r'^ELECTIVE\s*[IVX]*$', r'^CORE\s*\(.*\)$', r'^PROJECT\s*PHASE.*',
    r'^TABLE.*', r'^FIGURE.*', r'^ACADEMIC PRESS.*', r'^REFERENCES$',
    r'^BIBLIOGRAPHY$', r'^PAGE\s+\d+', r'^\d+\s*[-–]\s*\d+$',  # Simple ranges
    r'^\(\d+\)', r'^\d+\)', r'^\d+\.',  # Numbered lists
]

def is_section_header(line: str) -> bool:
    """
    Determine if a line should be considered a section header.

    HARD NEGATIVE FILTERS (must all pass):
    - Line length >= 10 characters
    - Line has at least 2 alphabetic words
    - Does NOT contain only numbers, brackets, or units
    - Does NOT match hard rejection patterns
    - Is NOT in a dense table block

    SEMANTIC STRENGTH REQUIREMENTS (must pass):
    - Mostly uppercase OR title-case
    - Contains at least ONE strong policy keyword OR
      (contains weak keyword + has strong semantic content)
    - Has at least 2 meaningful words
    """
    line = line.strip()
    line_upper = line.upper()

    # HARD NEGATIVE FILTER A: Line contains ONLY numbers, brackets, equals, or units
    # Reject lines that are mostly numeric/symbolic with no substantial text
    if re.match(r'^[\(\)\d\s\-\=\.\:\/\|]+$', line) or \
       re.match(r'.*\d.*HOURS.*$|\d.*MARKS.*$|\d.*CREDITS.*$|\d.*POINTS.*$', line, re.IGNORECASE):
        return False

    # HARD NEGATIVE FILTER B: Line length < 10 OR fewer than 2 alphabetic words
    if len(line) < 10:
        return False

    alphabetic_words = [word for word in re.findall(r'\b[A-Za-z]+\b', line)]
    if len(alphabetic_words) < 2:
        return False

    # HARD NEGATIVE FILTER C: Matches rejection patterns
    for pattern in HARD_REJECTION_PATTERNS:
        if re.search(pattern, line_upper, re.IGNORECASE):
            return False

    # HARD NEGATIVE FILTER D: Dense table block heuristic
    # (This would need context from surrounding lines - simplified here)
    # For now, we'll rely on other filters

    # SEMANTIC STRENGTH: Check formatting
    is_uppercase = line.isupper()
    is_title_case = line.istitle()

    if not (is_uppercase or is_title_case):
        return False

    # SEMANTIC STRENGTH: Check for strong policy keywords
    # Be VERY restrictive - only accept MAJOR section headers, not sub-headers
    has_strong_keyword = False
    for keyword in STRONG_POLICY_KEYWORDS:
        if keyword in line_upper:
            # VERY strict checks for major sections only:
            # 1. Line must start with the keyword (or be very close to it)
            # 2. Line should not contain multiple sections or be too long
            # 3. Should not look like a sub-section (contain numbers like 1.1, 2.3, etc.)
            if (line_upper.startswith(keyword.split()[0]) or  # Starts with first word of keyword
                (keyword in line_upper and len(line_upper.strip()) <= len(keyword) + 15) and  # Short line with keyword
                not re.search(r'\d+\.\d+', line_upper) and  # Not a sub-section like "1.1"
                not re.search(r'\([a-z]\)', line_upper.lower()) and  # Not a sub-point like "(a)"
                not line_upper.count('.') > 1):  # Not a complex reference
                has_strong_keyword = True
                break

    if has_strong_keyword:
        return True

    # SEMANTIC STRENGTH: Check for weak keywords with semantic strength
    has_weak_keyword = any(keyword in line_upper for keyword in WEAK_POLICY_KEYWORDS)

    if has_weak_keyword:
        # Additional check: must have at least 3 words and not look like a table entry
        words = line.split()
        if len(words) >= 3:
            # Not a table entry if it doesn't look like "COURSE CODE NAME"
            return not re.match(r'^\w+\s+\w+\s+\w+', line, re.IGNORECASE)

    return False

# Regulation year patterns
YEAR_PATTERNS = [
    r'(\d{4}[-–]\d{2,4})',  # 2024-25, 2021-22
    r'Regulation[s]?\s*[:\-]?\s*(\d{4})',  # Regulations: 2021
    r'Academic\s+Year\s*[:\-]?\s*(\d{4}[-–]\d{2,4})',
    r'Effective\s+from\s*[:\-]?\s*(\d{4}[-–]?\d{2,4}?)',
]

# Program name patterns
PROGRAM_PATTERNS = [
    r'\bB\.?\s*Tech\b',
    r'\bM\.?\s*Tech\b',
    r'\bMBA\b',
    r'\bMCA\b',
    r'\bM\.?\s*Sc\b',
    r'\bPhD\b',
    r'\bPh\.?\s*D\.?\b',
]


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
    section_category: str = "GENERAL"  # For retrieval bias
    source: Dict[str, str] = None

    def __post_init__(self):
        if self.source is None:
            self.source = {"pdf": "", "page_range": ""}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the administrative AKO"""
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
        """
        Extract all regulation sections from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of AdministrativeAKO objects
        """
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
            
            # Combine all pages
            full_text = "\n\n".join([text for _, text in pages_text])
            
            # Extract metadata from PDF
            regulation_name, regulation_year, program = self._extract_metadata(
                full_text, pdf_path.name
            )
            
            # Extract sections
            sections = self._extract_sections(
                full_text,
                pdf_path.name,
                regulation_name,
                regulation_year,
                program,
                pages_text
            )
            
            logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")
            return sections
            
        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}", exc_info=True)
            return []
    
    def _extract_metadata(
        self,
        text: str,
        filename: str
    ) -> Tuple[str, Optional[str], str]:
        """
        Extract regulation name, year, and program from text/filename.
        
        Returns:
            (regulation_name, regulation_year, program)
        """
        # Extract regulation year
        regulation_year = None
        for pattern in YEAR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                regulation_year = match.group(1)
                break
        
        # Extract program from text
        program = "Unknown"
        for pattern in PROGRAM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize program name
                if 'B' in pattern and 'Tech' in pattern:
                    program = "B.Tech"
                elif 'M' in pattern and 'Tech' in pattern:
                    program = "M.Tech"
                elif 'MBA' in pattern:
                    program = "MBA"
                elif 'MCA' in pattern:
                    program = "MCA"
                elif 'Sc' in pattern:
                    program = "M.Sc"
                elif 'PhD' in pattern or 'Ph.D' in pattern:
                    program = "PhD"
                break
        
        # If not found in text, try filename
        if program == "Unknown":
            filename_lower = filename.lower()
            if 'btech' in filename_lower or 'b.tech' in filename_lower:
                program = "B.Tech"
            elif 'mtech' in filename_lower or 'm.tech' in filename_lower:
                program = "M.Tech"
            elif 'mba' in filename_lower:
                program = "MBA"
            elif 'mca' in filename_lower:
                program = "MCA"
            elif 'msc' in filename_lower or 'm.sc' in filename_lower:
                program = "M.Sc"
            elif 'phd' in filename_lower:
                program = "PhD"
        
        # Extract regulation name (from title or first major heading)
        regulation_name = self._extract_regulation_name(text, filename)
        
        return regulation_name, regulation_year, program
    
    def _extract_regulation_name(self, text: str, filename: str) -> str:
        """Extract regulation name from text or filename"""
        # Try to find title in first few lines
        lines = text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title
                if any(keyword in line.upper() for keyword in ['REGULATION', 'RULES', 'POLICY', 'GUIDELINES']):
                    return line
        
        # Fallback: use filename (cleaned)
        name = Path(filename).stem
        name = re.sub(r'^\d+\s+', '', name)  # Remove leading dates
        name = re.sub(r'_\d{4}', '', name)  # Remove year suffixes
        name = name.replace('_', ' ').title()
        return name
    
    def _extract_sections(
        self,
        text: str,
        pdf_name: str,
        regulation_name: str,
        regulation_year: Optional[str],
        program: str,
        pages_text: List[Tuple[int, str]]
    ) -> List[AdministrativeAKO]:
        """Extract regulation sections from text with improved header detection and deduplication"""
        sections = []
        # Maps normalized_title -> list of content-similarity keys already kept
        # for that title in this PDF. A combined document (e.g. one PDF
        # covering both MBA and IB tracks) can legitimately repeat a heading
        # like "ASSESSMENT METHOD" with entirely different content per track —
        # dedup must compare content, not just title, or the second track's
        # real policy text gets silently dropped as a "duplicate".
        seen_titles: Dict[str, List[str]] = {}
        rejected_headers = set()  # Track rejected headers for logging

        # Split text into lines and find section headers
        lines = text.split('\n')
        section_positions = []

        for i, line in enumerate(lines):
            if is_section_header(line):
                section_positions.append((i, line.strip()))
            else:
                # Track rejected headers (sample a few for logging)
                stripped = line.strip()
                if stripped and len(rejected_headers) < 5:  # Limit to avoid spam
                    rejected_headers.add(stripped[:50])  # Truncate long lines

        # Log rejected headers once per PDF
        if rejected_headers:
            logger.info(f"Filtered non-policy headers in {pdf_name}: {list(rejected_headers)}")

        if not section_positions:
            logger.warning(f"No section headers found in {pdf_name}")
            # Create one section for entire document
            content = self._clean_text(text)
            if len(content) > 50:
                ako = AdministrativeAKO(
                    program=program,
                    regulation_name=regulation_name,
                    regulation_year=regulation_year,
                    section_title="GENERAL REGULATIONS",
                    content=content,
                    source={"pdf": pdf_name, "page_range": self._get_page_range(text, pages_text)}
                )
                sections.append(ako)
            return sections

        # Extract sections between consecutive headers
        for i, (line_idx, section_title) in enumerate(section_positions):
            # Normalize section title
            normalized_title = self._normalize_section_title(section_title)

            # Determine content start and end line indices
            content_start_line = line_idx + 1

            # Find end position (next header or end of text)
            # Allow more content by extending boundaries
            if i + 1 < len(section_positions):
                next_header_line = section_positions[i + 1][0]
                # Stop exactly at the next header — do NOT bleed into the next section
                content_end_line = next_header_line
            else:
                content_end_line = len(lines)

            # Extract section content (from lines after header to next header)
            content_lines = lines[content_start_line:content_end_line]

            # Clean and filter the content lines
            filtered_content_lines = []
            for line in content_lines:
                line = line.strip()
                if not line:
                    continue

                # Basic filtering - skip obvious non-content
                if len(line) < 3:  # Too short
                    continue
                if line.isupper() and len(line.split()) <= 3:  # Short uppercase headers
                    continue
                if re.match(r'^[\(\)\d\s\-\=\.\:\/\|]+$', line):  # Mostly symbols/numbers
                    continue

                filtered_content_lines.append(line)

            section_content = '\n'.join(filtered_content_lines)
            section_content = self._clean_text(section_content)

            # Skip if content is too short — this is the TOC-entry / running-header
            # case: a header-like line immediately followed by another header-like
            # line leaves almost nothing in between. IMPORTANT: this check happens
            # BEFORE the dedup registration below, so a near-empty occurrence of a
            # title (e.g. from a Table of Contents) never consumes the dedup slot
            # and blocks the real, content-bearing section of the same title that
            # appears later in the document.
            if len(section_content.strip()) < 50:
                continue

            # Deduplication check — only reached once we know this occurrence has
            # substantial content. Compares BOTH title and content: a repeated
            # heading with genuinely different content underneath (e.g. an
            # "ASSESSMENT METHOD" subsection under an MBA track vs. the same
            # heading under an IB track in one combined regulations PDF) is
            # kept as a separate section. Only a near-identical repeat — same
            # title AND essentially the same opening text — is treated as a
            # true duplicate and skipped.
            content_key = re.sub(r'\s+', ' ', section_content.strip().lower())[:200]
            existing_content_keys = seen_titles.get(normalized_title, [])
            if content_key in existing_content_keys:
                logger.warning(f"  ⚠ Skipped duplicate section: {normalized_title} in {pdf_name}")
                continue

            # Derive section category
            section_category = self._derive_section_category(normalized_title)

            # Create AKO
            ako = AdministrativeAKO(
                program=program,
                regulation_name=regulation_name,
                regulation_year=regulation_year,
                section_title=normalized_title,
                content=section_content,
                section_category=section_category,
                source={
                    "pdf": pdf_name,
                    "page_range": self._get_page_range(section_content, pages_text)
                }
            )

            is_valid, errors = ako.validate()
            if is_valid:
                sections.append(ako)
                # Only claim this content-key once actually kept — a title can
                # now hold multiple distinct content-keys (e.g. MBA vs IB track)
                seen_titles.setdefault(normalized_title, []).append(content_key)
                logger.info(f"  ✓ Extracted section: {normalized_title}")
            else:
                logger.warning(f"  ✗ Skipped invalid section: {errors}")
                # Do NOT add to seen_titles — an invalid section must not block
                # a later, valid occurrence of the same title either.

        return sections

    def _normalize_section_title(self, title: str) -> str:
        """
        FINAL comprehensive section title normalization for administrative regulations.
        This is the LAST PATCH before freeze - handles concatenated words, parentheses, and canonical forms.
        """
        # 1. Remove leading numbering patterns
        # Examples: "1 ", "3.", "10 ", "8.", "I ", "II.", etc.
        normalized = re.sub(r'^\s*(\d+|[IVX]+)[\.\)]?\s*', '', title, flags=re.IGNORECASE)

        # 2. FIX 1 - WORD BOUNDARY REPAIR (MANDATORY)
        # Insert spaces between concatenated uppercase words
        # Examples: FORADMISSION → FOR ADMISSION, THEPROGRAMME → THE PROGRAMME
        # Use regex to find uppercase-to-uppercase transitions that should be separated

        # First, handle lowercase-to-uppercase (e.g., "FORADMISSION")
        normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)

        # Then handle uppercase-to-uppercase when followed by lowercase (e.g., "OFGRADES" -> "OF GRADES")
        # This is more complex - look for patterns like "OFGRADES" where "OF" should be separated from "GRADES"
        normalized = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', normalized)

        # Additional fix for common concatenated words
        normalized = re.sub(r'\bTHE([A-Z])', r'THE \1', normalized)  # THEPROGRAMME -> THE PROGRAMME
        normalized = re.sub(r'\bOF([A-Z])', r'OF \1', normalized)    # OFGRADES -> OF GRADES
        normalized = re.sub(r'\bFOR([A-Z])', r'FOR \1', normalized)  # FORADMISSION -> FOR ADMISSION
        normalized = re.sub(r'\bAND([A-Z])', r'AND \1', normalized)  # ANDENROLLMENT -> AND ENROLLMENT

        # 3. FIX 2 - PARENTHESIS CLEANUP (MANDATORY)
        # Remove dangling or incomplete parentheses
        # If '(' exists with no matching ')', strip everything from '(' onward
        if '(' in normalized and ')' not in normalized:
            normalized = re.sub(r'\s*\(.*', '', normalized)

        # Remove any remaining incomplete brackets
        normalized = re.sub(r'[()]', '', normalized)

        # 4. FIX 3 - CANONICAL TITLE MAP (SAFE WHITELIST)
        # Map common variants to canonical forms AFTER regex cleanup
        canonical_map = {
            "AWARD OFGRADES": "AWARD OF GRADES",
            "ELIGIBILITY FORADMISSION": "ELIGIBILITY FOR ADMISSION",
            "DURATION OF THEPROGRAMME": "DURATION OF THE PROGRAMME",
            "SEMESTEREXAMINATION": "SEMESTER EXAMINATION",
            "ACADEMIC APPEALS BOARD (AAB": "ACADEMIC APPEALS BOARD",
            "ACADEMIC APPEALS BOARD AAB": "ACADEMIC APPEALS BOARD",
        }

        # Apply canonical mappings
        normalized_upper = normalized.upper()
        for variant, canonical in canonical_map.items():
            if variant in normalized_upper:
                normalized = canonical
                break

        # 5. FIX 4 - FINAL TITLE STANDARD
        # Ensure final section_title standards
        normalized = normalized.upper()  # Convert to uppercase
        normalized = re.sub(r'\s+', ' ', normalized)  # Single spaces only
        normalized = re.sub(r'[^\w\s]$', '', normalized)  # No trailing punctuation
        normalized = normalized.strip()

        # Length check (though this should be handled upstream)
        if len(normalized) <= 5:
            return ""

        return normalized

    def _normalize_section_title_post_processing(self, title: str, content: str) -> str:
        """
        OPTIONAL post-processing to normalize truncated section titles using content.
        Only applies to titles that appear incomplete based on specific rules.

        Args:
            title: Current section title
            content: Section content

        Returns:
            Potentially improved title, or original title if no improvement possible
        """
        if not ENABLE_TITLE_NORMALIZATION:
            return title

        # RULE 1: Apply ONLY if section_title length < 25 characters
        if len(title) >= 25:
            return title

        # RULE 2: Apply ONLY if section_title ends with specific words
        title_upper = title.upper()
        if not (title_upper.endswith(' FOR') or
                title_upper.endswith(' OF') or
                title_upper.endswith(' AND') or
                title_upper.endswith(' THE')):
            return title

        # RULE 6: Do NOT normalize if title contains numbers or specific keywords
        if (re.search(r'\d', title) or  # Contains numbers
            any(keyword in title_upper for keyword in [
                'EXAMINATION', 'ASSESSMENT', 'AWARD', 'GRADING', 'ELIGIBILITY',
                'ADMISSION', 'DURATION', 'WITHDRAWAL', 'REGISTRATION'
            ])):
            return title

        # RULE 3: Extract the first sentence from content
        # Find first sentence (up to first period, question mark, or newline)
        first_sentence = ""
        sentence_endings = ['. ', '? ', '! ', '\n']

        for ending in sentence_endings:
            if ending in content:
                first_sentence = content.split(ending)[0].strip()
                break

        if not first_sentence:
            first_sentence = content.strip()

        # RULE 4: Extract minimal phrase (max 5 words) from the beginning of the sentence
        # Focus on the first meaningful words that complete the title
        words = first_sentence.split()
        if len(words) > 5:
            words = words[:5]

        # Filter out common stop words and get meaningful completion
        completion_phrase = ' '.join(words).strip()

        # RULE 5: Resulting title must be <= 80 characters
        potential_title = f"{title} {completion_phrase}".strip()
        if len(potential_title) > 80:
            return title  # Too long, keep original

        # RULE 7: Log every normalization
        if potential_title != title:
            logger.info(f"[TITLE FIX] '{title}' → '{potential_title}'")

        return potential_title

    def _apply_eligibility_for_admission_fix(self, ako: AdministrativeAKO, pdf_filename: str) -> AdministrativeAKO:
        """
        OPTIONAL heuristic to fix "ELIGIBILITY FOR" titles in specific cases.
        Only applies to exact match with strict conditions.

        Args:
            ako: AdministrativeAKO object
            pdf_filename: Source PDF filename

        Returns:
            Potentially modified AKO
        """
        if not ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX:
            return ako

        # CONDITION 1: section_title must be exactly "ELIGIBILITY FOR"
        if ako.section_title != "ELIGIBILITY FOR":
            return ako

        # CONDITION 2: program must be known
        if not ako.program or ako.program == "Unknown":
            return ako

        # CONDITION 3: PDF filename must contain "Regulations"
        if "Regulations" not in pdf_filename:
            return ako

        # CONDITION 4: content must contain relevant keywords
        content_lower = ako.content.lower()
        required_keywords = ["admission", "programme", "candidate", "eligibility"]

        if not any(keyword in content_lower for keyword in required_keywords):
            return ako

        # ALL conditions met - apply the fix
        original_title = ako.section_title
        ako.section_title = "ELIGIBILITY FOR ADMISSION"

        logger.info(f"[OPTIONAL FIX] '{original_title}' → '{ako.section_title}' (filename-based)")

        return ako

    def _apply_mca_eligibility_cosmetic_fix(self, ako: AdministrativeAKO) -> AdministrativeAKO:
        """
        COSMETIC-ONLY fix for MCA_Regulations_2021.pdf edge case.
        Extremely specific and locked to exact conditions.
        """
        if not ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX:
            return ako

        # ALL conditions must match exactly
        if (ako.section_title == "ELIGIBILITY FOR" and
            ako.program == "MCA" and
            ako.source.get("pdf") == "MCA_Regulations_2021.pdf"):

            original_title = ako.section_title
            ako.section_title = "ELIGIBILITY FOR ADMISSION"

            logger.info("[COSMETIC FIX] MCA truncated title normalized to 'ELIGIBILITY FOR ADMISSION'")

        return ako

    def _derive_section_category(self, title: str) -> str:
        """
        Derive section category for retrieval bias.
        Returns one of: ELIGIBILITY, EXAMINATION, ASSESSMENT, GRADING,
                       ADMISSION, CREDIT, EXIT, PROGRAMME, GENERAL
        """
        title_upper = title.upper()

        # Check for specific categories in order of priority
        if 'ELIGIBILITY' in title_upper or 'ADMISSION' in title_upper:
            return 'ELIGIBILITY'
        elif 'EXAMINATION' in title_upper or 'ASSESSMENT' in title_upper:
            return 'EXAMINATION'
        elif 'GRADING' in title_upper or 'GRADE' in title_upper:
            return 'GRADING'
        elif 'CREDIT' in title_upper or 'ACADEMIC BANK' in title_upper:
            return 'CREDIT'
        elif 'EXIT' in title_upper or 'AWARD' in title_upper or 'MULTIPLE' in title_upper:
            return 'EXIT'
        elif 'PROGRAMME' in title_upper or 'STRUCTURE' in title_upper or 'CURRICULUM' in title_upper:
            return 'PROGRAMME'
        elif 'ATTENDANCE' in title_upper or 'DISCIPLINE' in title_upper or 'CONDUCT' in title_upper:
            return 'GENERAL'
        else:
            return 'GENERAL'

    def _clean_text(self, text: str) -> str:
        """
        Clean text for embedding: remove page numbers, table markers, excessive whitespace.
        Light cleaning that preserves content integrity.
        """
        if not text:
            return ""

        # Remove page numbers and table markers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Standalone numbers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Table\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Figure\s+\d+', '', text, flags=re.IGNORECASE)

        # Remove isolated numeric-only lines (common in tables)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep line if it has alphabetic content or is substantial
            if line and (re.search(r'[A-Za-z]', line) or len(line.split()) > 2):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Collapse repeated whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize Unicode characters
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')
        text = text.replace('\u2026', '...')

        return text.strip()
    
    def _get_page_range(self, content: str, pages_text: List[Tuple[int, str]]) -> str:
        """Estimate page range for content (simplified)"""
        # Find pages that might contain this content
        # This is approximate - in production, you'd track page boundaries more precisely
        if len(pages_text) == 1:
            return str(pages_text[0][0])
        elif len(pages_text) > 1:
            return f"{pages_text[0][0]}-{pages_text[-1][0]}"
        return "N/A"
    
    def extract_from_directory(self, dir_path: Path) -> List[AdministrativeAKO]:
        """
        Extract sections from all PDFs in a directory.
        
        Args:
            dir_path: Path to directory containing PDFs
            
        Returns:
            List of all AdministrativeAKO objects
        """
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
        """
        Save administrative AKOs to JSON file.
        
        Args:
            akos: List of AdministrativeAKO objects
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # OPTIONAL: Apply title normalization post-processing
        if ENABLE_TITLE_NORMALIZATION:
            logger.info("Applying optional title normalization post-processing...")
            for ako in akos:
                original_title = ako.section_title
                improved_title = self._normalize_section_title_post_processing(
                    original_title, ako.content
                )
                ako.section_title = improved_title

        # OPTIONAL: Apply specific "ELIGIBILITY FOR" fix (secondary fallback)
        if ENABLE_ELIGIBILITY_FOR_ADMISSION_FIX:
            logger.info("Applying optional 'ELIGIBILITY FOR' fix...")
            for ako in akos:
                pdf_filename = ako.source.get("pdf", "")
                self._apply_eligibility_for_admission_fix(ako, pdf_filename)

        # COSMETIC: Apply MCA-specific fix (final edge case)
        if ENABLE_MCA_ELIGIBILITY_COSMETIC_FIX:
            logger.info("Applying MCA cosmetic fix...")
            for ako in akos:
                self._apply_mca_eligibility_cosmetic_fix(ako)

        # Convert to dictionaries
        akos_dict = [ako.to_dict() for ako in akos]
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(akos_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(akos)} administrative AKOs to {output_path}")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("SAIGE ProdMachine - Administrative Regulations Extractor")
    logger.info("="*60)
    
    extractor = AdministrativeExtractor()
    
    # Extract from administrative directory
    if not DATA_RAW_DIR.exists():
        logger.error(f"Administrative regulations directory not found: {DATA_RAW_DIR}")
        return
    
    logger.info(f"\nExtracting from: {DATA_RAW_DIR}")
    akos = extractor.extract_from_directory(DATA_RAW_DIR)
    
    if not akos:
        logger.warning("No sections extracted. Check PDF format and extraction patterns.")
        return
    
    # Save to AKOs directory
    output_path = AKOS_DIR / "administrative" / "administrative.json"
    extractor.save_akos(akos, output_path)
    
    logger.info("\n" + "="*60)
    logger.info(f"Administrative Extraction Complete!")
    logger.info(f"Total sections extracted: {len(akos)}")
    logger.info(f"Output: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
