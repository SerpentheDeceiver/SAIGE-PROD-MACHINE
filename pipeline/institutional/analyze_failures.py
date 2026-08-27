import json
from pathlib import Path
from collections import Counter

_f = Path(__file__).resolve().parent.parent.parent / "akos" / "institutional" / "debug_llm_failures.json"

with open(_f, "r", encoding="utf-8") as f:
    failures = json.load(f)

errors = Counter([f["error"] for f in failures])

print("Failure breakdown:")
for k, v in errors.items():
    print(k, "->", v)