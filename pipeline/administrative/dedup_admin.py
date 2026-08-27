import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROD_ROOT  = SCRIPT_DIR.parent.parent

INPUT  = PROD_ROOT / "akos" / "administrative" / "administrative.json"
OUTPUT = PROD_ROOT / "akos" / "administrative" / "administrative_dedup.json"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()
deduped = []

for item in data:
    key = (
        item["section_title"].strip().lower(),
        item["program"].strip().lower()
    )
    if key not in seen:
        seen.add(key)
        deduped.append(item)

print(f"Original: {len(data)}")
print(f"After title+program dedup: {len(deduped)}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(deduped, f, indent=2, ensure_ascii=False)

print("Stronger dedup completed.")