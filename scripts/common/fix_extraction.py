"""
Quick patch to fix institutional extraction - Day 1
This script modifies extract_institutional.py to accept more content
"""

import re
from pathlib import Path

script_path = Path('scripts/extract_institutional.py')

with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("Applying fixes to extract_institutional.py...")

# Fix 1: Lower MIN_WORDS_PER_AKO
content = re.sub(
    r'MIN_WORDS_PER_AKO = 150',
    'MIN_WORDS_PER_AKO = 75',
    content
)
print("✓ Fix 1: Lowered MIN_WORDS_PER_AKO to 75")

# Fix 2: Lower TARGET_WORDS
content = re.sub(
    r'TARGET_WORDS = 250',
    'TARGET_WORDS = 150',
    content
)
print("✓ Fix 2: Lowered TARGET_WORDS to 150")

# Fix 3: Relax category detection (require only 1 keyword match instead of 2)
old_pattern = r'# Require minimum score of 2\s+if max_score < 2:\s+return ""'
new_pattern = '# Require minimum score of 1\n        if max_score < 1:\n            return ""'
content = re.sub(old_pattern, new_pattern, content)
print("✓ Fix 3: Relaxed category detection (1 keyword minimum)")

# Fix 4: Disable mixed category check (too aggressive)
old_mixed = r'# Check mixed categories\s+if self\._has_mixed_categories\(section_content, content_category\):\s+logger\.debug\(f"  ⚠ Skipped mixed categories: \{section_title\[:50\]\}\.\.\.">\)\s+continue'
new_mixed = '# Check mixed categories (DISABLED for Day 1 - too aggressive)\n                    # if self._has_mixed_categories(section_content, content_category):\n                    #     logger.debug(f"  ⚠ Skipped mixed categories: {section_title[:50]}...")\n                    #     continue'
content = re.sub(old_mixed, new_mixed, content, flags=re.DOTALL)
print("✓ Fix 4: Disabled mixed category check")

# Save backup
backup_path = script_path.parent / f"{script_path.stem}_backup_day1.py"
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"✓ Backup saved to: {backup_path}")

# Save fixed version
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n✅ All fixes applied to {script_path}")
print("\nNow run: python scripts\\extract_institutional.py")