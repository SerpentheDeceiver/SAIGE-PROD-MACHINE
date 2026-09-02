"""
SAIGE - Institutional Dedup v2.1

CHANGES IN v2.1:
- INPUT now points to institutional_titled.json instead of
  institutional_clean.json. MIGRATION NOTE: this is a breaking path change.
  The previous pipeline had dedup reading institutional_clean.json directly,
  but the (previously missing, now reconstructed) llm_title_institutional.py
  stage sits between cleanup and dedup and writes institutional_titled.json.
  Pointing dedup at the old filename would silently read stale/pre-titling
  data again - update any external scripts or docs referencing
  institutional_clean.json as dedup's input.
- Records AKO count to the shared pipeline audit trail.
- Soft dedup now uses config/embedding.yaml through the shared embedding
  loader. The previous hidden MiniLM dependency has been removed so offline
  builds fail fast unless the configured common embedding model is staged.
"""
import json
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_audit import record_stage  # noqa: E402
from pipeline.common.embedding_config import load_embedding_config, load_sentence_transformer  # noqa: E402

_PROD_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# CHANGED: was institutional_clean.json — see migration note above.
INPUT = str(_PROD_ROOT / "akos" / "institutional" / "institutional_titled.json")
OUTPUT = str(_PROD_ROOT / "akos" / "institutional" / "institutional_final.json")
REPORT = str(_PROD_ROOT / "data" / "reports" / "institutional_dedup_report.json")

EMBEDDING_CONFIG = load_embedding_config()
MODEL_NAME = EMBEDDING_CONFIG.model_name

STRICT_THRESHOLD = 0.90
GENERIC_THRESHOLD = 0.80
CROSSCAT_GENERIC = True
DEDUPE_WITHIN_CATEGORY = True

BAD_TITLES = {
    "general information",
    "information",
    "overview",
    "untitled",
    "title",
    "6-12 words",
    "n/a",
}


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _hash(text: str) -> str:
    return hashlib.md5((text or "").strip().encode("utf-8", errors="ignore")).hexdigest()


def _title(x: Dict) -> str:
    return (_as_str(x.get("llm_title")) or _as_str(x.get("section_title"))).strip()


def _content(x: Dict) -> str:
    return _as_str(x.get("content")).strip()


def _category(x: Dict) -> str:
    return (_as_str(x.get("category")).strip().lower() or "unknown")


def _is_bad_title(t: str) -> bool:
    tt = (t or "").strip().lower()
    if not tt:
        return True
    if tt in BAD_TITLES:
        return True
    if "general information" in tt:
        return True
    if len(tt.split()) < 3:
        return True
    return False


def pick_better(a: Dict, b: Dict) -> Dict:
    def cleanup_rank(x: Dict) -> int:
        cv = _as_str(x.get("cleanup_version")).lower()
        if "ollama_llm_v2" in cv:
            return 3
        if "ollama_llm_v1" in cv or "llm" in cv:
            return 2
        if "rule" in cv:
            return 1
        return 0

    ta = _title(a)
    tb = _title(b)
    sa = (0 if _is_bad_title(ta) else 1, len(_content(a).split()), cleanup_rank(a))
    sb = (0 if _is_bad_title(tb) else 1, len(_content(b).split()), cleanup_rank(b))
    return a if sa >= sb else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    in_path = Path(INPUT)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}\n"
            f"Did you run llm_title_institutional.py first? Dedup now reads "
            f"institutional_titled.json, not institutional_clean.json."
        )

    data: List[Dict] = json.load(open(in_path, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} AKOs from {INPUT}")

    # ---------------------------
    # 1) Exact dedup by content_hash
    # ---------------------------
    exact_map: Dict[str, Dict] = {}
    exact_duplicates_removed = 0

    for obj in data:
        content = _content(obj)
        h = _as_str(obj.get("content_hash")).strip() or _hash(content)
        obj["content_hash"] = h

        if h in exact_map:
            exact_duplicates_removed += 1
            exact_map[h] = pick_better(exact_map[h], obj)
        else:
            exact_map[h] = obj

    after_exact = list(exact_map.values())

    # ---------------------------
    # 2) Soft dedup (configured embeddings)
    # ---------------------------
    model = load_sentence_transformer(EMBEDDING_CONFIG)

    groups = defaultdict(list)
    if DEDUPE_WITHIN_CATEGORY:
        for obj in after_exact:
            groups[_category(obj)].append(obj)
    else:
        groups["ALL"].extend(after_exact)

    generic_pool: List[Dict] = []
    if CROSSCAT_GENERIC:
        for obj in after_exact:
            if _is_bad_title(_title(obj)):
                generic_pool.append(obj)

    kept_all: List[Dict] = []
    soft_duplicates_removed = 0
    soft_merge_events = 0

    top_sims_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    merge_log: List[Dict[str, Any]] = []

    def embed_text(x: Dict) -> str:
        t = _title(x)
        c = _content(x)
        txt = (t + "\n" + c).strip()
        return txt[:3000]

    def cluster(items: List[Dict], cat_name: str) -> List[Dict]:
        nonlocal soft_duplicates_removed, soft_merge_events

        if len(items) <= 1:
            return items

        texts = [embed_text(x) for x in items]
        emb = model.encode(texts, normalize_embeddings=EMBEDDING_CONFIG.normalize)
        emb = np.asarray(emb, dtype=np.float32)

        reps: List[int] = []
        clusters: List[List[int]] = []

        for i in range(len(items)):
            if not reps:
                reps.append(i)
                clusters.append([i])
                continue

            best_j = -1
            best_sim = -1.0
            for r_idx, rep_i in enumerate(reps):
                sim = cosine(emb[i], emb[rep_i])
                if sim > best_sim:
                    best_sim = sim
                    best_j = r_idx

            title_i = _title(items[i])
            thr = GENERIC_THRESHOLD if _is_bad_title(title_i) else STRICT_THRESHOLD

            if len(top_sims_by_cat[cat_name]) < 20:
                top_sims_by_cat[cat_name].append({
                    "sim": round(best_sim, 4),
                    "title_a": _title(items[i])[:80],
                    "title_b": _title(items[reps[best_j]])[:80] if best_j >= 0 else "",
                    "threshold_used": thr,
                })

            if best_sim >= thr:
                clusters[best_j].append(i)
                soft_merge_events += 1
            else:
                reps.append(i)
                clusters.append([i])

        kept: List[Dict] = []
        for cl in clusters:
            if len(cl) == 1:
                kept.append(items[cl[0]])
                continue

            best = items[cl[0]]
            merged_ids = [items[cl[0]].get("ako_id")]
            for j in cl[1:]:
                soft_duplicates_removed += 1
                merged_ids.append(items[j].get("ako_id"))
                best = pick_better(best, items[j])

            kept.append(best)
            merge_log.append({
                "category": cat_name,
                "kept_ako_id": best.get("ako_id"),
                "merged_ako_ids": merged_ids,
                "kept_title": _title(best),
            })

        return kept

    for cat, items in groups.items():
        kept_all.extend(cluster(items, cat))

    if CROSSCAT_GENERIC and len(generic_pool) >= 2:
        generic_kept = cluster(generic_pool, "GENERIC_CROSSCAT")
        generic_ids = {x.get("ako_id") for x in generic_pool if x.get("ako_id")}
        kept_all = [x for x in kept_all if x.get("ako_id") not in generic_ids]
        kept_all.extend(generic_kept)

    kept_all.sort(key=lambda x: (_category(x), _title(x).lower()))

    report = {
        "input_file": INPUT,
        "output_file": OUTPUT,
        "model": MODEL_NAME,
        "thresholds": {
            "strict_threshold": STRICT_THRESHOLD,
            "generic_threshold": GENERIC_THRESHOLD,
            "dedupe_within_category": DEDUPE_WITHIN_CATEGORY,
            "crosscat_generic": CROSSCAT_GENERIC,
        },
        "counts": {
            "original": len(data),
            "after_exact": len(after_exact),
            "after_soft": len(kept_all),
            "exact_duplicates_removed": exact_duplicates_removed,
            "soft_duplicates_removed": soft_duplicates_removed,
            "soft_merge_events": soft_merge_events,
        },
        "top_similarities_sample": dict(top_sims_by_cat),
        "merge_log_sample": merge_log[:50],
        "notes": [
            "Exact dedup uses content_hash.",
            "Soft dedup embeds (title + content) using the configured common embedding model.",
            "MiniLM is intentionally not loaded here; set config/embedding.yaml explicitly.",
            "Adaptive thresholds: stricter for good titles, lower for generic titles.",
            "Optional cross-category merges only for generic/bad titles to avoid accidental merges.",
            "Input is institutional_titled.json (post LLM-title stage), not institutional_clean.json.",
        ],
    }

    ensure_parent(OUTPUT)
    ensure_parent(REPORT)
    json.dump(kept_all, open(OUTPUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(report, open(REPORT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("Saved:", OUTPUT)
    print("Saved:", REPORT)
    print("Counts:", report["counts"])

    record_stage(
        "dedup",
        input_file=INPUT,
        output_file=OUTPUT,
        input_count=len(data),
        output_count=len(kept_all),
        extra={
            "exact_duplicates_removed": exact_duplicates_removed,
            "soft_duplicates_removed": soft_duplicates_removed,
        },
    )


if __name__ == "__main__":
    main()
