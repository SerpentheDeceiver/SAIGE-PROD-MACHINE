"""
Syllabus Extractor Module for SAIGE ProdMachine

Extracts structured Academic Knowledge Objects (AKOs) from Indian university
syllabus PDFs.

Fixes:
  1. LTP_PATTERN: [---] char class replaced with proper [-\u2013\u2014] dash class
     → credits were parsing as 0 or garbage (1872, 5396) due to regex not matching
  2. _extract_references(): $ changed to re.MULTILINE flag so it stops at section end
     → textbook references were being cut off or missed entirely
  3. _extract_unit_topics() secondary split: [/,--\-] replaced with correct pattern
     → was splitting on wrong chars, producing noise fragments
  4. All other regex patterns using [---] inside char classes also fixed
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

# Paths
SCRIPT_DIR = Path(__file__).parent
PROD_ROOT = SCRIPT_DIR.parent
DATA_RAW_DIR = PROD_ROOT / "data" / "raw" / "academics"
AKOS_DIR = PROD_ROOT / "akos"

# ── Regex patterns ────────────────────────────────────────────────────────────
COURSE_CODE_PATTERN = re.compile(
    r'([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+(.+?)(?=\n|Periods|Credit|Course\s+Code|Department:|$)',
    re.MULTILINE | re.IGNORECASE
)

UNIT_HEADER_PATTERN = re.compile(
    r'UNIT\s*[-\u2013\u2014:]?\s*([IVX]+|I{1,3}|[1-5])\b',  # FIX 1: proper dash
    re.IGNORECASE
)

COURSE_OUTCOME_PATTERN = re.compile(
    r'CO(\d+)\s+(.+?)(?=CO\d+|UNIT|Lecture Periods|$)',
    re.IGNORECASE | re.DOTALL
)

PERIODS_PATTERN = re.compile(
    r'Periods?\s*[:\-]\s*(\d+)',
    re.IGNORECASE
)

# FIX 1: was r'(\d+)\s*[---]\s*(\d+)\s*[---]\s*(\d+)'
# [---] in a character class only matches literal '-', '–' but NOT as separator
# The original char class [---] is actually [- to -] which is just [-]
# We need to match L-T-P format where separator is a literal hyphen or en-dash
LTP_PATTERN = re.compile(
    r'(\d+)\s*[-\u2013\u2014]\s*(\d+)\s*[-\u2013\u2014]\s*(\d+)',
    re.IGNORECASE
)

ROMAN_NUMERALS = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
}


@dataclass
class CourseAKO:
    """Academic Knowledge Object for a Course"""
    object_type: str = "course"
    program: str = ""
    degree_level: str = ""
    department: str = ""
    regulation_year: Optional[str] = None
    course_code: str = ""
    course_name: str = ""
    semester: str = ""
    credits: Dict[str, int] = None
    contact_hours: Dict[str, int] = None
    course_objectives: List[str] = None
    course_outcomes: List[Dict[str, str]] = None
    units: List[Dict[str, Any]] = None
    textbook_references: List[str] = None
    source: Dict[str, str] = None

    def __post_init__(self):
        if self.credits is None:
            self.credits = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}
        if self.contact_hours is None:
            self.contact_hours = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}
        if self.course_objectives is None:
            self.course_objectives = []
        if self.course_outcomes is None:
            self.course_outcomes = []
        if self.units is None:
            self.units = []
        if self.textbook_references is None:
            self.textbook_references = []
        if self.source is None:
            self.source = {"pdf": "", "page_range": ""}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.course_code:
            errors.append("Missing course_code (PRIMARY KEY)")
        if not self.course_name:
            errors.append("Missing course_name")
        if not self.units:
            logger.warning(f"Course {self.course_code} has no units - may be a lab course")
        return len(errors) == 0, errors


class SyllabusExtractor:
    """Extracts structured course information from syllabus PDFs"""

    def __init__(self):
        self.courses: List[CourseAKO] = []

    def extract_from_pdf(self, pdf_path: Path) -> List[CourseAKO]:
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
            filename = pdf_path.stem
            program, degree_level, department = self._parse_filename(filename)
            regulation_year = self._extract_regulation_year(full_text)

            courses = self._extract_courses(
                full_text, pdf_path.name, program, degree_level,
                department, regulation_year, pages_text
            )

            logger.info(f"Extracted {len(courses)} courses from {pdf_path.name}")
            return courses

        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}", exc_info=True)
            return []

    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        parts = filename.lower().split('_')

        degree_level = ""
        if parts[0] == 'ug':
            degree_level = "UG"
        elif parts[0] == 'pg':
            degree_level = "PG"

        program = ""
        if 'btech' in parts or 'b.tech' in parts:
            program = "B.Tech"
        elif 'mtech' in parts or 'm.tech' in parts:
            program = "M.Tech"
        elif 'mba' in parts:
            program = "MBA"
        elif 'mca' in parts:
            program = "MCA"
        elif 'msc' in parts or 'm.sc' in parts:
            program = "M.Sc"

        dept_map = {
            'cse': 'CSE', 'ece': 'ECE', 'eee': 'EEE', 'eie': 'EIE',
            'me': 'ME', 'civil': 'Civil', 'chemical': 'Chemical',
            'it': 'IT', 'mt': 'MT', 'ce': 'Civil',
            'ib': 'International Business', 'ievd': 'IEVD',
            'datascience': 'Data Science', 'infosec': 'Information Security',
            'wireless': 'Wireless Communication', 'drives': 'Drives',
            'instrumentation': 'Instrumentation', 'iot': 'IoT',
            'energy': 'Energy', 'pdm': 'PDM',
            'environmental': 'Environmental', 'structural': 'Structural'
        }

        department = "Unknown"
        for part in parts:
            if part in dept_map:
                department = dept_map[part]
                break

        return program, degree_level, department

    def _extract_regulation_year(self, text: str) -> Optional[str]:
        patterns = [
            r'Regulation[s]?\s*[:\-]?\s*(\d{4}[-\u2013]\d{2,4})',
            r'Effective from.*?(\d{4}[-\u2013]\d{2,4})',
            r'Academic year\s*(\d{4}[-\u2013]\d{2,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_courses(
        self, text: str, pdf_name: str, program: str, degree_level: str,
        department: str, regulation_year: Optional[str],
        pages_text: List[Tuple[int, str]]
    ) -> List[CourseAKO]:
        courses = []
        course_sections = self._split_into_course_sections(text)

        for section_text, start_pos in course_sections:
            course = self._parse_course_section(
                section_text, pdf_name, program, degree_level,
                department, regulation_year, pages_text, start_pos
            )
            if course:
                is_valid, errors = course.validate()
                if is_valid:
                    courses.append(course)
                    logger.info(f"  ✓ Extracted: {course.course_code} - {course.course_name}")
                else:
                    logger.warning(f"  ✗ Skipped invalid course: {errors}")

        return courses

    def _split_into_course_sections(self, text: str) -> List[Tuple[str, int]]:
        sections = []

        course_header_pattern = re.compile(
            r'Course\s+Code\s+Course\s+Name.*?\n([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+(.+?)'
            r'(?=Course\s+Code|Department:|$)',
            re.IGNORECASE | re.DOTALL
        )

        matches = list(course_header_pattern.finditer(text))

        if matches:
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((text[start_pos:end_pos], start_pos))
        else:
            matches = list(COURSE_CODE_PATTERN.finditer(text))
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((text[start_pos:end_pos], start_pos))

        return sections

    def _parse_course_section(
        self, section_text: str, pdf_name: str, program: str, degree_level: str,
        department: str, regulation_year: Optional[str],
        pages_text: List[Tuple[int, str]], start_pos: int
    ) -> Optional[CourseAKO]:
        course = CourseAKO()
        course.program = program
        course.degree_level = degree_level
        course.department = department
        course.regulation_year = regulation_year
        course.source["pdf"] = pdf_name

        if not course.degree_level:
            if course.program in ["MBA", "MCA", "M.Tech", "M.Sc"]:
                course.degree_level = "PG"
            elif course.program in ["B.Tech", "B.E"]:
                course.degree_level = "UG"

        code_match = COURSE_CODE_PATTERN.search(section_text)
        if not code_match:
            header_pattern = re.compile(
                r'Course\s+Code\s+Course\s+Name.*?\n([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+'
                r'(.+?)(?=\n|Periods|Credit|$)',
                re.IGNORECASE | re.DOTALL
            )
            code_match = header_pattern.search(section_text)

        if not code_match:
            return None

        course.course_code = code_match.group(1).strip()
        course.course_name = re.sub(r'\s+', ' ', code_match.group(2).strip())
        course.semester = self._extract_semester(section_text)

        credits = self._extract_credits(section_text)
        if credits:
            course.credits = credits

        hours = self._extract_contact_hours(section_text)
        if hours:
            course.contact_hours = hours

        course.course_objectives = self._extract_course_objectives(section_text)
        course.course_outcomes = self._extract_course_outcomes(section_text)
        course.units = self._extract_units(section_text, course.course_outcomes)

        if not course.units:
            logger.error(f"No units detected for course {course.course_code}")
        else:
            all_empty = all(
                unit.get("hours", 0) == 0 and len(unit.get("topics", [])) == 0
                for unit in course.units
            )
            if all_empty:
                logger.error(f"Units detected but empty for course {course.course_code}")

            for unit in course.units:
                unit_num = unit.get("unit_number", "?")
                hours_val = unit.get("hours", 0)
                topics = unit.get("topics", [])

                if unit.get("unit_title"):
                    if re.search(r'Periods?\s*[:\-]\s*\d+', section_text, re.IGNORECASE):
                        if hours_val == 0:
                            logger.error(
                                f"Unit {unit_num} in {course.course_code} has "
                                f"'Periods' but hours = 0"
                            )

                for topic in topics:
                    if re.search(r'Periods?\s*[:\-]\s*\d+', topic, re.IGNORECASE):
                        logger.error(
                            f"Unit {unit_num} in {course.course_code} has "
                            f"'Periods' in topic: {topic}"
                        )

        course.textbook_references = self._extract_references(section_text)
        course.source["page_range"] = self._extract_page_range(
            section_text, pages_text, start_pos
        )

        return course

    def _extract_semester(self, text: str) -> str:
        patterns = [
            r'Semester\s*[:\-]?\s*([IVX]+|I{1,3}|[1-8])',
            r'(\d+)\s*(?:st|nd|rd|th)\s*Semester',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sem = match.group(1).strip()
                if sem.upper() in ROMAN_NUMERALS:
                    return str(ROMAN_NUMERALS[sem.upper()])
                return sem
        return ""

    def _extract_credits(self, text: str) -> Optional[Dict[str, int]]:
        """FIX 1: Use proper dash character class in LTP_PATTERN."""
        credits = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}

        ltp_match = LTP_PATTERN.search(text)
        if ltp_match:
            credits["lecture"] = int(ltp_match.group(1))
            credits["tutorial"] = int(ltp_match.group(2))
            credits["practical"] = int(ltp_match.group(3))
            credits["total"] = credits["lecture"] + credits["tutorial"] + credits["practical"]
            return credits

        credit_patterns = [
            (r'L\s*[:\-]?\s*(\d+)', "lecture"),
            (r'T\s*[:\-]?\s*(\d+)', "tutorial"),
            (r'P\s*[:\-]?\s*(\d+)', "practical"),
            (r'Credit[s]?\s*[:\-]?\s*(\d+)', "total"),
        ]

        for pattern, key in credit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                credits[key] = int(match.group(1))

        if (credits["total"] == 0 and
                (credits["lecture"] > 0 or credits["tutorial"] > 0 or credits["practical"] > 0)):
            credits["total"] = (credits["lecture"] + credits["tutorial"] + credits["practical"])

        return credits if (credits["total"] > 0 or credits["lecture"] > 0) else None

    def _extract_contact_hours(self, text: str) -> Optional[Dict[str, int]]:
        hours = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}
        patterns = [
            (r'Lecture\s+Periods?\s*[:\-]?\s*(\d+)', "lecture"),
            (r'Tutorial\s+Periods?\s*[:\-]?\s*(\d+)', "tutorial"),
            (r'Practical\s+Periods?\s*[:\-]?\s*(\d+)', "practical"),
            (r'Total\s+Periods?\s*[:\-]?\s*(\d+)', "total"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hours[key] = int(match.group(1))
        return hours if (hours["total"] > 0 or hours["lecture"] > 0) else None

    def _extract_course_objectives(self, text: str) -> List[str]:
        objectives = []
        obj_pattern = re.compile(
            r'Course\s+Objective[s]?\s*[:\-]?\s*(.+?)(?=Course\s+Outcome|UNIT|$)',
            re.IGNORECASE | re.DOTALL
        )
        match = obj_pattern.search(text)
        if match:
            obj_text = match.group(1).strip()
            lines = re.split(r'^\d+[\.)\s]*', obj_text, flags=re.MULTILINE)
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    objectives.append(line)
        return objectives

    def _extract_course_outcomes(self, text: str) -> List[Dict[str, str]]:
        outcomes = []
        co_section_pattern = re.compile(
            r'Course\s+Outcome[s]?[:\-]?\s*(.+?)'
            r'(?=UNIT|Lecture Periods|Reference Books|CO-PO Mapping|$)',
            re.IGNORECASE | re.DOTALL
        )
        co_section_match = co_section_pattern.search(text)
        if co_section_match:
            co_text = co_section_match.group(1)
            co_matches = COURSE_OUTCOME_PATTERN.finditer(co_text)
            for co_match in co_matches:
                co_id = f"CO{co_match.group(1)}"
                description = re.sub(r'\s+', ' ', co_match.group(2).strip())
                if description:
                    outcomes.append({"co_id": co_id, "description": description})
        return outcomes

    def _extract_units(
        self, text: str, course_outcomes: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        unit_slices = self._slice_units(text)
        if not unit_slices:
            return []
        units = []
        for unit_num_str, unit_content in unit_slices:
            unit = self._parse_single_unit(unit_num_str, unit_content, course_outcomes)
            if unit:
                units.append(unit)
        return units

    def _slice_units(self, text: str) -> List[Tuple[str, str]]:
        unit_matches = list(UNIT_HEADER_PATTERN.finditer(text))
        if not unit_matches:
            return []

        end_markers = [
            re.search(r'\bReference\s+Books?\b', text, re.IGNORECASE),
            re.search(r'\bTextbook[s]?\b', text, re.IGNORECASE),
            re.search(r'CO-PO\s+Mapping', text, re.IGNORECASE),
            re.search(r'Lecture\s+Periods?\s*:', text, re.IGNORECASE),
        ]
        end_positions = [m.start() for m in end_markers if m]
        section_end = min(end_positions) if end_positions else len(text)

        unit_slices = []
        for i, match in enumerate(unit_matches):
            unit_num_str = match.group(1).strip()
            start_pos = match.start()
            end_pos = unit_matches[i + 1].start() if i + 1 < len(unit_matches) else section_end
            unit_content = text[start_pos:end_pos].strip()
            unit_slices.append((unit_num_str, unit_content))

        return unit_slices

    def _parse_single_unit(
        self, unit_num_str: str, unit_content: str,
        course_outcomes: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        unit_number = self._normalize_unit_number(unit_num_str)
        unit_title = self._extract_unit_title(unit_content)
        hours = self._extract_unit_hours(unit_content)
        topics = self._extract_unit_topics(unit_content)
        descriptive_content = self._extract_unit_descriptive_content(unit_content)
        mapped_cos = self._extract_mapped_cos(unit_content, course_outcomes)

        return {
            "unit_number": unit_number,
            "unit_title": unit_title,
            "hours": hours,
            "topics": topics,
            "content": descriptive_content,
            "mapped_course_outcomes": mapped_cos
        }

    def _normalize_unit_number(self, unit_num: str) -> str:
        unit_num = unit_num.strip().upper()
        if unit_num in ROMAN_NUMERALS:
            num = ROMAN_NUMERALS[unit_num]
            roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
            return roman_map.get(num, unit_num)
        try:
            num = int(unit_num)
            roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
            return roman_map.get(num, unit_num)
        except ValueError:
            return unit_num

    def _extract_unit_title(self, unit_content: str) -> str:
        lines = unit_content.split('\n')
        first_line = lines[0].strip() if lines else ""

        if first_line:
            title_match = re.search(
                r'UNIT\s*[-\u2013\u2014:]?\s*[IVX]+[:\s]+(.+?)(?=\n|Periods|$)',
                first_line, re.IGNORECASE
            )
            if title_match:
                title = re.sub(r'[:\-\u2013\u2014]\s*$', '', title_match.group(1)).strip()
                if title and len(title) > 3 and not PERIODS_PATTERN.search(title):
                    return title

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if PERIODS_PATTERN.search(line):
                continue
            if re.match(r'^CO\d+[,\s]*$', line, re.IGNORECASE):
                continue
            if len(line) > 3:
                title = re.sub(r'[:\-\u2013\u2014]\s*$', '', line).strip()
                return title

        return ""

    def _extract_unit_hours(self, unit_content: str) -> int:
        match = PERIODS_PATTERN.search(unit_content)
        return int(match.group(1)) if match else 0

    def _extract_unit_topics(self, unit_content: str) -> List[str]:
        stop_patterns = [
            r'\bCO-PO\s+Mapping\b',
            r'\bCourse\s+Outcome\b',
            r'\bReference\s+Books?\b',
            r'\bTextbook[s]?\b',
            r'Lecture\s+Periods?\s*:',
        ]

        content = unit_content
        for pattern in stop_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                content = content[:match.start()]
                break

        lines = content.split('\n')
        topic_lines = []
        found_title = False

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if PERIODS_PATTERN.search(line):
                continue
            if not found_title and len(line) > 3 and not PERIODS_PATTERN.search(line):
                found_title = True
                continue
            if re.match(r'^CO\d+[,\s]*$', line, re.IGNORECASE):
                continue
            if re.match(r'^(UNIT|Reference|Textbook|Lecture|Tutorial|Practical|Total)',
                        line, re.IGNORECASE):
                continue
            topic_lines.append(line)

        topics = []
        for line in topic_lines:
            if re.search(r'Periods?\s*[:\-]\s*\d+', line, re.IGNORECASE):
                continue
            cleaned = re.sub(r'^[\d\w]+[\.)\s]*', '', line)
            cleaned = re.sub(r'^[-•]\s*', '', cleaned)
            cleaned = re.sub(r'\s*CO\d+[,\s]*$', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*CO\d+[,\s]*CO\d+[,\s]*$', '', cleaned, flags=re.IGNORECASE)

            if ';' in cleaned:
                sub_topics = [t.strip() for t in cleaned.split(';')]
                for sub_topic in sub_topics:
                    if sub_topic and len(sub_topic) > 3:
                        topics.append(sub_topic)
            else:
                if cleaned and len(cleaned) > 3:
                    topics.append(cleaned)

        # FIX 3: Secondary split — was r'[/,--\-]' which is a broken char class
        # [--\-] means char range from '-' to '\' which is wrong
        # Correct: split on '/', ',', literal '--', or ' and '
        split_topics = []
        for topic in topics:
            parts = re.split(r'\s*/\s*|\s*,\s*|\s+--\s+|\s+and\s+', topic,
                             flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 4:
                    part = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
                    split_topics.append(part)

        cleaned_topics = []
        seen = set()
        for topic in split_topics:
            topic = topic.strip()
            topic = re.sub(r'[,;\.]+$', '', topic)
            topic = re.sub(r'^[-\u2013\u2014]\s*', '', topic)
            if re.search(r'Periods?\s*[:\-]\s*\d+', topic, re.IGNORECASE):
                continue
            if topic and len(topic) >= 4 and topic.lower() not in seen:
                cleaned_topics.append(topic)
                seen.add(topic.lower())

        return cleaned_topics

    def _extract_unit_descriptive_content(self, unit_content: str) -> str:
        lines = unit_content.split('\n')
        if lines and lines[0].strip().startswith(('UNIT', 'Unit')):
            lines = lines[1:]

        structured_start_indices = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if PERIODS_PATTERN.search(line):
                continue
            if len(line) > 3 and not PERIODS_PATTERN.search(line):
                if len(line) < 100 and not line.startswith(('-', '•', '*')):
                    continue
            if (line.startswith(('-', '•', '*')) or
                    re.match(r'^CO\d+', line, re.IGNORECASE) or
                    re.match(r'^\d+[\.)\s]*', line)):
                structured_start_indices.append(i)
                break

        if structured_start_indices:
            descriptive_lines = []
            for line in lines[:structured_start_indices[0]]:
                line = line.strip()
                if (line and len(line) > 10 and not PERIODS_PATTERN.search(line) and
                        not line.upper().startswith(('UNIT', 'COURSE', 'REFERENCE', 'TEXTBOOK'))):
                    descriptive_lines.append(line)
            if descriptive_lines:
                return ' '.join(descriptive_lines).strip()

        return ""

    def _extract_mapped_cos(
        self, unit_content: str, course_outcomes: List[Dict[str, str]]
    ) -> List[str]:
        mapped_cos = []
        co_pattern = re.compile(r'CO(\d+)', re.IGNORECASE)
        matches = co_pattern.findall(unit_content)

        for co_num in matches:
            co_id = f"CO{co_num}"
            if any(co["co_id"] == co_id for co in course_outcomes):
                if co_id not in mapped_cos:
                    mapped_cos.append(co_id)

        return sorted(mapped_cos)

    def _extract_references(self, text: str) -> List[str]:
        """
        FIX 2: Add re.MULTILINE so $ matches end of line, not end of string.
        Old regex stopped at 'CO-PO Mapping' OR end of entire string — but
        'CO-PO Mapping' often doesn't appear, so references were grabbed with
        everything after them included, or cut off completely.
        """
        references = []
        ref_pattern = re.compile(
            r'(?:Reference\s+Books?|Textbook[s]?|References?)\s*[:\-]?\s*'
            r'(.+?)(?=CO-PO\s+Mapping|$)',
            re.IGNORECASE | re.DOTALL | re.MULTILINE  # FIX 2: added re.MULTILINE
        )

        match = ref_pattern.search(text)
        if match:
            ref_text = match.group(1).strip()
            lines = re.split(r'^\d+[\.)\s]*', ref_text, flags=re.MULTILINE)
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    line = re.sub(r'\s+', ' ', line)
                    references.append(line)

        return references

    def _extract_page_range(
        self, section_text: str, pages_text: List[Tuple[int, str]], start_pos: int
    ) -> str:
        course_code_match = COURSE_CODE_PATTERN.search(section_text)
        if not course_code_match:
            return "N/A"

        course_code = course_code_match.group(1)
        page_nums = [
            page_num for page_num, page_text in pages_text
            if course_code in page_text
        ]

        if page_nums:
            return str(page_nums[0]) if len(page_nums) == 1 else f"{min(page_nums)}-{max(page_nums)}"
        return "N/A"

    def extract_from_directory(self, dir_path: Path) -> List[CourseAKO]:
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        all_courses = []
        pdf_files = sorted(dir_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")

        for pdf_path in pdf_files:
            courses = self.extract_from_pdf(pdf_path)
            all_courses.extend(courses)

        return all_courses

    def save_courses(self, courses: List[CourseAKO], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        courses_dict = [course.to_dict() for course in courses]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(courses_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(courses)} courses to {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("SAIGE ProdMachine - Syllabus Extractor")
    logger.info("=" * 60)

    extractor = SyllabusExtractor()

    if not DATA_RAW_DIR.exists():
        logger.error(f"Academics directory not found: {DATA_RAW_DIR}")
        return

    logger.info(f"\nExtracting courses from: {DATA_RAW_DIR}")
    courses = extractor.extract_from_directory(DATA_RAW_DIR)

    if not courses:
        logger.warning("No courses extracted. Check PDF format and extraction patterns.")
        return

    output_path = AKOS_DIR / "courses.json"
    extractor.save_courses(courses, output_path)

    logger.info("\n" + "=" * 60)
    logger.info("Syllabus Extraction Complete!")
    logger.info(f"Total courses extracted: {len(courses)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()