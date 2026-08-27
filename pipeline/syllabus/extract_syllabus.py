"""
Syllabus Extractor Module for SAIGE ProdMachine

Extracts structured Academic Knowledge Objects (AKOs) from Indian university
syllabus PDFs. This module compiles syllabus PDFs into structured course objects
with 100% structural correctness.

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

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
PROD_ROOT    = SCRIPT_DIR.parent.parent
DATA_RAW_DIR = PROD_ROOT / "data" / "raw" / "academics"
AKOS_DIR     = PROD_ROOT / "akos"

# Regex patterns for extraction
COURSE_CODE_PATTERN = re.compile(
    r'([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+(.+?)(?=\n|Periods|Credit|Course\s+Code|Department:|$)',
    re.MULTILINE | re.IGNORECASE
)

# UNIT_HEADER_PATTERN: Matches ONLY unit headers (not content)
# Examples: "UNIT I", "UNIT – II", "UNIT-III", "UNIT : IV", "UNIT 4"
UNIT_HEADER_PATTERN = re.compile(
    r'UNIT\s*[-–—:]?\s*([IVX]+|I{1,3}|[1-5])\b',
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

LTP_PATTERN = re.compile(
    r'(\d+)\s*[-–]\s*(\d+)\s*[-–]\s*(\d+)',
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
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the course AKO. Returns (is_valid, list_of_errors)"""
        errors = []
        
        if not self.course_code:
            errors.append("Missing course_code (PRIMARY KEY)")
        
        if not self.course_name:
            errors.append("Missing course_name")
        
        # FIX 4: Only skip if course_code or course_name missing
        # department, program, degree_level are not mandatory for skipping
        # (degree_level has fallback inference)
        
        # Units are optional for some courses (e.g., lab courses with experiments)
        # But we should log a warning if no units found
        if not self.units:
            logger.warning(f"Course {self.course_code} has no units - this may be a lab course")
        
        return len(errors) == 0, errors


class SyllabusExtractor:
    """Extracts structured course information from syllabus PDFs"""
    
    def __init__(self):
        self.courses: List[CourseAKO] = []
    
    def extract_from_pdf(self, pdf_path: Path) -> List[CourseAKO]:
        """
        Extract all courses from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of CourseAKO objects
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
            
            # Extract metadata from filename
            filename = pdf_path.stem
            program, degree_level, department = self._parse_filename(filename)
            
            # Extract regulation year if present in text
            regulation_year = self._extract_regulation_year(full_text)
            
            # Extract all courses
            courses = self._extract_courses(
                full_text,
                pdf_path.name,
                program,
                degree_level,
                department,
                regulation_year,
                pages_text
            )
            
            logger.info(f"Extracted {len(courses)} courses from {pdf_path.name}")
            return courses
            
        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}", exc_info=True)
            return []
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse program, degree level, and department from filename.
        
        Examples:
            ug_btech_cse -> ("B.Tech", "UG", "CSE")
            pg_mtech_cse_datascience -> ("M.Tech", "PG", "CSE")
        """
        parts = filename.lower().split('_')
        
        # Determine degree level
        if parts[0] == 'ug':
            degree_level = "UG"
        elif parts[0] == 'pg':
            degree_level = "PG"
        else:
            degree_level = ""
        
        # Determine program
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
        else:
            program = ""
        
        # Determine department
        dept_map = {
            'cse': 'CSE',
            'ece': 'ECE',
            'eee': 'EEE',
            'eie': 'EIE',
            'me': 'ME',
            'civil': 'Civil',
            'chemical': 'Chemical',
            'it': 'IT',
            'mt': 'MT',
            'ce': 'Civil',
            'ib': 'International Business',
            'ievd': 'IEVD',
            'datascience': 'Data Science',
            'infosec': 'Information Security',
            'wireless': 'Wireless Communication',
            'drives': 'Drives',
            'instrumentation': 'Instrumentation',
            'iot': 'IoT',
            'energy': 'Energy',
            'pdm': 'PDM',
            'environmental': 'Environmental',
            'structural': 'Structural'
        }
        
        department = ""
        for part in parts:
            if part in dept_map:
                department = dept_map[part]
                break
        
        # If department not found, try to infer from course codes later
        if not department:
            department = "Unknown"
        
        return program, degree_level, department
    
    def _extract_regulation_year(self, text: str) -> Optional[str]:
        """Extract regulation year from text"""
        patterns = [
            r'Regulation[s]?\s*[:\-]?\s*(\d{4}[-–]\d{2,4})',
            r'Effective from.*?(\d{4}[-–]\d{2,4})',
            r'Academic year\s*(\d{4}[-–]\d{2,4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_courses(
        self,
        text: str,
        pdf_name: str,
        program: str,
        degree_level: str,
        department: str,
        regulation_year: Optional[str],
        pages_text: List[Tuple[int, str]]
    ) -> List[CourseAKO]:
        """Extract all courses from text"""
        courses = []
        
        # Split text into potential course sections
        # Look for course code patterns followed by course details
        course_sections = self._split_into_course_sections(text)
        
        for section_text, start_pos in course_sections:
            course = self._parse_course_section(
                section_text,
                pdf_name,
                program,
                degree_level,
                department,
                regulation_year,
                pages_text,
                start_pos
            )
            
            if course:
                is_valid, errors = course.validate()
                if is_valid:
                    courses.append(course)
                    logger.info(f"  ✓ Extracted course: {course.course_code} - {course.course_name}")
                else:
                    logger.warning(f"  ✗ Skipped invalid course: {errors}")
        
        return courses
    
    def _split_into_course_sections(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into individual course sections.
        A course section starts with "Course Code" header or a course code pattern
        and ends before the next course section.
        """
        sections = []
        
        # Look for course sections by finding "Course Code" headers followed by actual codes
        # Pattern: "Course Code" header, then a course code like CSUC108, MAUC101, etc.
        course_header_pattern = re.compile(
            r'Course\s+Code\s+Course\s+Name.*?\n([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+(.+?)(?=Course\s+Code|Department:|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = list(course_header_pattern.finditer(text))
        
        if matches:
            # Found structured course sections
            for i, match in enumerate(matches):
                start_pos = match.start()
                
                # Determine end position
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos]
                sections.append((section_text, start_pos))
        else:
            # Fallback: find course codes directly
            matches = list(COURSE_CODE_PATTERN.finditer(text))
            
            for i, match in enumerate(matches):
                start_pos = match.start()
                
                # Determine end position (start of next course or end of text)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos]
                sections.append((section_text, start_pos))
        
        return sections
    
    def _parse_course_section(
        self,
        section_text: str,
        pdf_name: str,
        program: str,
        degree_level: str,
        department: str,
        regulation_year: Optional[str],
        pages_text: List[Tuple[int, str]],
        start_pos: int
    ) -> Optional[CourseAKO]:
        """Parse a single course section into a CourseAKO"""
        course = CourseAKO()
        course.program = program
        course.degree_level = degree_level
        course.department = department
        course.regulation_year = regulation_year
        course.source["pdf"] = pdf_name
        
        # FIX 4: Degree level inference fallback
        if not course.degree_level:
            if course.program in ["MBA", "MCA"]:
                course.degree_level = "PG"
            elif course.program in ["B.Tech", "B.E"]:
                course.degree_level = "UG"
        
        # Extract course code and name
        # Try multiple patterns
        code_match = None
        
        # Pattern 1: Direct course code at start of section
        code_match = COURSE_CODE_PATTERN.search(section_text)
        
        # Pattern 2: Course code after "Course Code" header
        if not code_match:
            header_pattern = re.compile(
                r'Course\s+Code\s+Course\s+Name.*?\n([A-Z]{2,5}[UC]{0,2}\d{3}[A-Z]?)\s+(.+?)(?=\n|Periods|Credit|$)',
                re.IGNORECASE | re.DOTALL
            )
            code_match = header_pattern.search(section_text)
        
        if not code_match:
            return None
        
        course.course_code = code_match.group(1).strip()
        course.course_name = code_match.group(2).strip()
        
        # Clean course name (remove extra whitespace, newlines)
        course.course_name = re.sub(r'\s+', ' ', course.course_name).strip()
        
        # Extract semester
        course.semester = self._extract_semester(section_text)
        
        # Extract credits (L-T-P format)
        credits = self._extract_credits(section_text)
        if credits:
            course.credits = credits
        
        # Extract contact hours
        hours = self._extract_contact_hours(section_text)
        if hours:
            course.contact_hours = hours
        
        # Extract course objectives
        course.course_objectives = self._extract_course_objectives(section_text)
        
        # Extract course outcomes
        course.course_outcomes = self._extract_course_outcomes(section_text)
        
        # Extract units
        course.units = self._extract_units(section_text, course.course_outcomes)
        
        # Validate units (mandatory validation)
        if not course.units:
            logger.error(f"No units detected for course {course.course_code}")
        else:
            # Check if all units are empty
            all_empty = all(
                unit.get("hours", 0) == 0 and len(unit.get("topics", [])) == 0
                for unit in course.units
            )
            if all_empty:
                logger.error(f"Units detected but empty for course {course.course_code}")
            
            # VALIDATION UPDATE: Check hours > 0 for theory units and topics don't contain "Periods"
            for unit in course.units:
                unit_num = unit.get("unit_number", "?")
                hours = unit.get("hours", 0)
                topics = unit.get("topics", [])
                unit_title = unit.get("unit_title", "")
                
                # For theory units (non-empty title), hours should typically be > 0
                # But we only error if we detect "Periods" in the section but hours = 0
                # Check if "Periods" pattern exists in section for this unit
                if unit_title:  # Theory unit
                    # Search for "Periods" pattern in the section text around this unit
                    unit_section = section_text
                    if re.search(r'Periods?\s*[:\-]\s*\d+', unit_section, re.IGNORECASE):
                        if hours == 0:
                            logger.error(f"Unit {unit_num} in course {course.course_code} has 'Periods' mentioned but hours = 0")
                
                # Check topics don't contain "Periods"
                for topic in topics:
                    if re.search(r'Periods?\s*[:\-]\s*\d+', topic, re.IGNORECASE):
                        logger.error(f"Unit {unit_num} in course {course.course_code} has 'Periods' in topic: {topic}")
        
        # Extract textbook references
        course.textbook_references = self._extract_references(section_text)
        
        # Extract page range
        course.source["page_range"] = self._extract_page_range(
            section_text, pages_text, start_pos
        )
        
        return course
    
    def _extract_semester(self, text: str) -> str:
        """Extract semester number from text"""
        patterns = [
            r'Semester\s*[:\-]?\s*([IVX]+|I{1,3}|[1-8])',
            r'(\d+)\s*(?:st|nd|rd|th)\s*Semester',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sem = match.group(1).strip()
                # Convert roman numerals to numbers
                if sem.upper() in ROMAN_NUMERALS:
                    return str(ROMAN_NUMERALS[sem.upper()])
                return sem
        
        return ""
    
    def _extract_credits(self, text: str) -> Optional[Dict[str, int]]:
        """Extract credits in L-T-P format"""
        credits = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}
        
        # Look for L-T-P pattern
        ltp_match = LTP_PATTERN.search(text)
        if ltp_match:
            credits["lecture"] = int(ltp_match.group(1))
            credits["tutorial"] = int(ltp_match.group(2))
            credits["practical"] = int(ltp_match.group(3))
            credits["total"] = credits["lecture"] + credits["tutorial"] + credits["practical"]
            return credits
        
        # Look for individual credit fields
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
        
        if credits["total"] == 0 and (credits["lecture"] > 0 or credits["tutorial"] > 0 or credits["practical"] > 0):
            credits["total"] = credits["lecture"] + credits["tutorial"] + credits["practical"]
        
        return credits if credits["total"] > 0 or credits["lecture"] > 0 else None
    
    def _extract_contact_hours(self, text: str) -> Optional[Dict[str, int]]:
        """Extract contact hours (lecture, tutorial, practical)"""
        hours = {"lecture": 0, "tutorial": 0, "practical": 0, "total": 0}
        
        # Look for "Lecture Periods", "Tutorial Periods", "Practical Periods"
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
        
        return hours if hours["total"] > 0 or hours["lecture"] > 0 else None
    
    def _extract_course_objectives(self, text: str) -> List[str]:
        """Extract course objectives"""
        objectives = []
        
        # Look for "Course Objectives" section
        obj_pattern = re.compile(
            r'Course\s+Objective[s]?\s*[:\-]?\s*(.+?)(?=Course\s+Outcome|UNIT|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        match = obj_pattern.search(text)
        if match:
            obj_text = match.group(1).strip()
            # Split by numbered list or bullets
            lines = re.split(r'^\d+[\.\)]\s*', obj_text, flags=re.MULTILINE)
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short lines
                    objectives.append(line)
        
        return objectives
    
    def _extract_course_outcomes(self, text: str) -> List[Dict[str, str]]:
        """Extract course outcomes (CO1, CO2, etc.)"""
        outcomes = []
        
        # Look for "Course Outcome" section
        co_section_pattern = re.compile(
            r'Course\s+Outcome[s]?[:\-]?\s*(.+?)(?=UNIT|Lecture Periods|Reference Books|CO-PO Mapping|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        co_section_match = co_section_pattern.search(text)
        if co_section_match:
            co_text = co_section_match.group(1)
            
            # Extract individual COs
            co_matches = COURSE_OUTCOME_PATTERN.finditer(co_text)
            for co_match in co_matches:
                co_id = f"CO{co_match.group(1)}"
                description = co_match.group(2).strip()
                # Clean description (remove extra whitespace, newlines)
                description = re.sub(r'\s+', ' ', description).strip()
                if description:
                    outcomes.append({
                        "co_id": co_id,
                        "description": description
                    })
        
        return outcomes
    
    def _extract_units(
        self,
        text: str,
        course_outcomes: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Extract units with topics and mapped course outcomes.
        Uses two-stage slicing strategy: detect headers, then slice content.
        """
        # STAGE 1: Find all UNIT header positions
        unit_slices = self._slice_units(text)
        
        if not unit_slices:
            return []
        
        # STAGE 2: Parse each unit slice
        units = []
        for unit_num_str, unit_content in unit_slices:
            unit = self._parse_single_unit(unit_num_str, unit_content, course_outcomes)
            if unit:
                units.append(unit)
        
        return units
    
    def _slice_units(self, text: str) -> List[Tuple[str, str]]:
        """
        STAGE 1: Slice text between UNIT headers.
        Returns list of (unit_number_str, unit_content) tuples.
        """
        # Find all UNIT header matches with positions
        unit_matches = list(UNIT_HEADER_PATTERN.finditer(text))
        
        if not unit_matches:
            return []
        
        unit_slices = []
        
        # Find end markers (where units section ends)
        end_markers = [
            re.search(r'\bReference\s+Books?\b', text, re.IGNORECASE),
            re.search(r'\bTextbook[s]?\b', text, re.IGNORECASE),
            re.search(r'CO-PO\s+Mapping', text, re.IGNORECASE),
            re.search(r'Lecture\s+Periods?\s*:', text, re.IGNORECASE),
        ]
        end_positions = [m.start() for m in end_markers if m]
        section_end = min(end_positions) if end_positions else len(text)
        
        # Slice between each UNIT header
        for i, match in enumerate(unit_matches):
            unit_num_str = match.group(1).strip()
            start_pos = match.start()
            
            # Determine end position
            if i + 1 < len(unit_matches):
                # Next unit starts here
                end_pos = unit_matches[i + 1].start()
            else:
                # Last unit: go until section end marker
                end_pos = section_end
            
            # Extract unit content (from after header to next header or end)
            unit_content = text[start_pos:end_pos].strip()
            
            unit_slices.append((unit_num_str, unit_content))
        
        return unit_slices
    
    def _parse_single_unit(
        self,
        unit_num_str: str,
        unit_content: str,
        course_outcomes: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        STAGE 2: Parse a single unit's content.
        Extracts: unit_number, unit_title, hours, topics, content, mapped_course_outcomes
        """
        # Normalize unit number
        unit_number = self._normalize_unit_number(unit_num_str)

        # Extract unit title (first non-empty line after UNIT header, must NOT contain "Periods")
        unit_title = self._extract_unit_title(unit_content)

        # Extract hours from "Periods: XX"
        hours = self._extract_unit_hours(unit_content)

        # Extract topics (stop at end markers)
        topics = self._extract_unit_topics(unit_content)

        # Extract descriptive content (paragraphs that are not structured elements)
        descriptive_content = self._extract_unit_descriptive_content(unit_content)

        # Extract mapped course outcomes
        mapped_cos = self._extract_mapped_cos(unit_content, course_outcomes)

        return {
            "unit_number": unit_number,
            "unit_title": unit_title,
            "hours": hours,
            "topics": topics,
            "content": descriptive_content,  # Add descriptive content
            "mapped_course_outcomes": mapped_cos
        }
    
    def _normalize_unit_number(self, unit_num: str) -> str:
        """Normalize unit number to I, II, III, IV, V format"""
        unit_num = unit_num.strip().upper()
        
        # Convert roman numerals or numbers to standard format
        if unit_num in ROMAN_NUMERALS:
            num = ROMAN_NUMERALS[unit_num]
            roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
            return roman_map.get(num, unit_num)
        
        # If it's a number string
        try:
            num = int(unit_num)
            roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
            return roman_map.get(num, unit_num)
        except ValueError:
            return unit_num
    
    def _extract_unit_title(self, unit_content: str) -> str:
        """
        Extract unit title: first NON-empty line after UNIT header.
        Must NOT contain "Periods".
        Handles titles on same line as UNIT header (e.g., "UNIT III : Advanced Topics").
        """
        lines = unit_content.split('\n')
        
        # Check first line: might have title after UNIT header
        first_line = lines[0].strip() if lines else ""
        
        # Try to extract title from first line if it contains UNIT header
        if first_line:
            # Pattern: "UNIT III : Advanced Topics" or "UNIT-III Advanced Topics"
            title_match = re.search(
                r'UNIT\s*[-–—:]?\s*[IVX]+[:\s]+(.+?)(?=\n|Periods|$)',
                first_line,
                re.IGNORECASE
            )
            if title_match:
                title = title_match.group(1).strip()
                # Clean up: remove trailing colons, dashes
                title = re.sub(r'[:–—]\s*$', '', title).strip()
                if title and len(title) > 3 and not PERIODS_PATTERN.search(title):
                    return title
        
        # If not found on first line, check subsequent lines
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Skip if contains "Periods"
            if PERIODS_PATTERN.search(line):
                continue
            
            # Skip if it's just a CO reference
            if re.match(r'^CO\d+[,\s]*$', line, re.IGNORECASE):
                continue
            
            # This should be the title
            if len(line) > 3:
                # Clean up: remove any trailing colons, dashes
                title = re.sub(r'[:–—]\s*$', '', line).strip()
                return title
        
        return ""
    
    def _extract_unit_hours(self, unit_content: str) -> int:
        """
        Extract hours/periods for a unit.
        Detects "Periods : X" OR "Periods: X"
        Returns integer X, or 0 if not found.
        """
        match = PERIODS_PATTERN.search(unit_content)
        if match:
            return int(match.group(1))
        return 0
    
    def _extract_unit_topics(self, unit_content: str) -> List[str]:
        """
        Extract topics from unit content.
        Handles bullet points, numbered lists, or semicolon-separated topics.
        STOPS when encountering: CO-PO, Course Outcome, Reference Books, Textbooks
        """
        # Find stop markers
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
        
        # Remove UNIT header line
        lines = content.split('\n')
        # Skip first line (UNIT header) and find where topics start
        topic_lines = []
        found_title = False
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Skip "Periods: XX" line
            if PERIODS_PATTERN.search(line):
                continue
            
            # Skip unit title (first substantial line after header)
            if not found_title and len(line) > 3 and not PERIODS_PATTERN.search(line):
                found_title = True
                continue
            
            # Skip if it's just a CO reference line
            if re.match(r'^CO\d+[,\s]*$', line, re.IGNORECASE):
                continue
            
            # Skip lines that are clearly not topics
            if re.match(r'^(UNIT|Reference|Textbook|Lecture|Tutorial|Practical|Total)', line, re.IGNORECASE):
                continue
            
            topic_lines.append(line)
        
        # Extract topics from lines
        topics = []
        for line in topic_lines:
            # FIX 1: Remove any line that matches "Periods : X" or "Periods: X"
            if re.search(r'Periods?\s*[:\-]\s*\d+', line, re.IGNORECASE):
                continue
            
            # Remove numbering (1., 2., i., ii., etc.)
            cleaned = re.sub(r'^[\d\w]+[\.\)]\s*', '', line)
            cleaned = re.sub(r'^[-•]\s*', '', cleaned)
            
            # Remove trailing CO references (CO1, CO2, etc.)
            cleaned = re.sub(r'\s*CO\d+[,\s]*$', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*CO\d+[,\s]*CO\d+[,\s]*$', '', cleaned, flags=re.IGNORECASE)
            
            # Split by semicolons if present
            if ';' in cleaned:
                sub_topics = [t.strip() for t in cleaned.split(';')]
                for sub_topic in sub_topics:
                    sub_topic = sub_topic.strip()
                    if sub_topic and len(sub_topic) > 3:
                        topics.append(sub_topic)
            else:
                if cleaned and len(cleaned) > 3:
                    topics.append(cleaned)
        
        # FIX 3: Split compound topics (secondary splitting)
        split_topics = []
        for topic in topics:
            # Split on: '/', ',', '–', '-', ' and '
            # Use regex to handle various separators
            parts = re.split(r'\s*[/,–\-]\s*|\s+and\s+', topic, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                # Discard fragments shorter than 4 characters
                if part and len(part) >= 4:
                    # Normalize casing (title case)
                    part = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
                    split_topics.append(part)
        
        # Clean and deduplicate
        cleaned_topics = []
        seen = set()
        for topic in split_topics:
            topic = topic.strip()
            # Remove trailing punctuation
            topic = re.sub(r'[,;\.]+$', '', topic)
            # Remove leading dashes
            topic = re.sub(r'^[-–—]\s*', '', topic)
            # FIX 1: Final check - ensure no "Periods" in topics
            if re.search(r'Periods?\s*[:\-]\s*\d+', topic, re.IGNORECASE):
                continue
            if topic and len(topic) >= 4 and topic.lower() not in seen:
                cleaned_topics.append(topic)
                seen.add(topic.lower())
        
        return cleaned_topics

    def _extract_unit_descriptive_content(self, unit_content: str) -> str:
        """
        Extract descriptive paragraphs from unit content.
        This captures explanatory text that is not part of structured elements
        like topics, CO mappings, or headers.

        Returns:
            String containing descriptive paragraphs, or empty string if none found.
        """
        # Remove the UNIT header line
        lines = unit_content.split('\n')
        if lines and lines[0].strip().startswith(('UNIT', 'Unit')):
            lines = lines[1:]  # Skip header line

        # Find where structured content starts (title, topics, COs)
        structured_start_indices = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip lines we've already identified as structured
            if PERIODS_PATTERN.search(line):
                continue

            # Check if this is the unit title (first substantial line)
            if len(line) > 3 and not PERIODS_PATTERN.search(line):
                # Check if this looks like a title (not too long, no bullets)
                if len(line) < 100 and not line.startswith(('-', '•', '*')):
                    # This might be the title, skip it and continue looking for content
                    continue

            # Check for topic bullets or CO references
            if (line.startswith(('-', '•', '*')) or
                re.match(r'^CO\d+', line, re.IGNORECASE) or
                re.match(r'^\d+[\.\)]\s*', line)):  # Numbered lists
                structured_start_indices.append(i)
                break

        # If we found structured content, extract everything before it
        if structured_start_indices:
            descriptive_lines = []
            for line in lines[:structured_start_indices[0]]:
                line = line.strip()
                if line and len(line) > 10 and not PERIODS_PATTERN.search(line):
                    # Filter out headers and very short lines
                    if not line.upper().startswith(('UNIT', 'COURSE', 'REFERENCE', 'TEXTBOOK')):
                        descriptive_lines.append(line)

            if descriptive_lines:
                return ' '.join(descriptive_lines).strip()

        return ""

    def _extract_mapped_cos(
        self,
        unit_content: str,
        course_outcomes: List[Dict[str, str]]
    ) -> List[str]:
        """Extract mapped course outcomes for a unit"""
        mapped_cos = []
        
        # Look for CO references in unit content
        co_pattern = re.compile(r'CO(\d+)', re.IGNORECASE)
        matches = co_pattern.findall(unit_content)
        
        for co_num in matches:
            co_id = f"CO{co_num}"
            # Verify this CO exists in course outcomes
            if any(co["co_id"] == co_id for co in course_outcomes):
                if co_id not in mapped_cos:
                    mapped_cos.append(co_id)
        
        return sorted(mapped_cos)
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract textbook references"""
        references = []
        
        # Look for "Reference Books" or "Textbook" section
        ref_pattern = re.compile(
            r'(?:Reference\s+Books?|Textbook[s]?|References?)\s*[:\-]?\s*(.+?)(?=CO-PO Mapping|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        match = ref_pattern.search(text)
        if match:
            ref_text = match.group(1).strip()
            # Split by numbered list
            lines = re.split(r'^\d+[\.\)]\s*', ref_text, flags=re.MULTILINE)
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short lines
                    # Clean up the reference
                    line = re.sub(r'\s+', ' ', line)
                    references.append(line)
        
        return references
    
    def _extract_page_range(
        self,
        section_text: str,
        pages_text: List[Tuple[int, str]],
        start_pos: int
    ) -> str:
        """Extract page range for the course section"""
        # Find which pages contain this section
        # This is approximate - we find the first page that contains the course code
        course_code_match = COURSE_CODE_PATTERN.search(section_text)
        if not course_code_match:
            return "N/A"
        
        course_code = course_code_match.group(1)
        
        # Find pages containing this course code
        page_nums = []
        for page_num, page_text in pages_text:
            if course_code in page_text:
                page_nums.append(page_num)
        
        if page_nums:
            if len(page_nums) == 1:
                return str(page_nums[0])
            else:
                return f"{min(page_nums)}-{max(page_nums)}"
        
        return "N/A"
    
    def extract_from_directory(self, dir_path: Path) -> List[CourseAKO]:
        """
        Extract courses from all PDFs in a directory.
        
        Args:
            dir_path: Path to directory containing PDFs
            
        Returns:
            List of all CourseAKO objects
        """
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
        """
        Save courses to JSON file.
        
        Args:
            courses: List of CourseAKO objects
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionaries
        courses_dict = [course.to_dict() for course in courses]
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(courses_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(courses)} courses to {output_path}")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("SAIGE ProdMachine - Syllabus Extractor")
    logger.info("="*60)
    
    extractor = SyllabusExtractor()
    
    # Extract from academics directory
    if not DATA_RAW_DIR.exists():
        logger.error(f"Academics directory not found: {DATA_RAW_DIR}")
        return
    
    logger.info(f"\nExtracting courses from: {DATA_RAW_DIR}")
    courses = extractor.extract_from_directory(DATA_RAW_DIR)
    
    if not courses:
        logger.warning("No courses extracted. Check PDF format and extraction patterns.")
        return
    
    # Save to AKOs directory
    output_path = AKOS_DIR / "syllabus" / "courses.json"
    extractor.save_courses(courses, output_path)
    
    logger.info("\n" + "="*60)
    logger.info(f"Syllabus Extraction Complete!")
    logger.info(f"Total courses extracted: {len(courses)}")
    logger.info(f"Output: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
