import os
import json
import faiss
import yaml
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROD_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"

SYLLABUS_INDEX = PROD_ROOT / "data" / "vector_db" / "syllabus_faiss" / "faiss.index"
ADMIN_INDEX = PROD_ROOT / "data" / "vector_db" / "admin_faiss" / "faiss.index"
INSTITUTIONAL_INDEX = PROD_ROOT / "data" / "vector_db" / "institutional_faiss" / "faiss.index"

SYLLABUS_META = PROD_ROOT / "data" / "vector_db" / "syllabus_faiss" / "metadata.json"
ADMIN_META = PROD_ROOT / "data" / "vector_db" / "admin_faiss" / "metadata.json"
INSTITUTIONAL_META = PROD_ROOT / "data" / "vector_db" / "institutional_faiss" / "metadata.json"

TOP_K = 5
WEAK_MARGIN_THRESHOLD = 0.05
STRONG_MARGIN_THRESHOLD = 0.10


def load_embedding_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    emb = config["embedding"]
    return {
        "model_name": emb["model_name"],
        "normalize": emb.get("normalize_embeddings", True),
    }


def load_index(path: Path):
    if not path.exists():
        print(f"Index not found: {path}")
        return None
    return faiss.read_index(str(path))


def load_metadata(path: Path):
    if not path.exists():
        print(f"Metadata not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_preview(meta: dict, db_name: str, max_len: int = 220) -> str:
    if not meta:
        return ""

    if db_name == "syllabus":
        text = meta.get("ako_text", "")
    else:
        text = meta.get("content", "")

    text = " ".join(str(text).split())
    return text[:max_len]


def safe_title(meta: dict, db_name: str) -> str:
    if not meta:
        return ""

    if db_name == "syllabus":
        return meta.get("course_name", "") or meta.get("ako_title", "")
    return meta.get("section_title", "") or meta.get("category", "")


def search_db(index, metadata, embedding, db_name: str):
    if index is None:
        return None

    distances, indices = index.search(embedding, TOP_K)
    top_idx = int(indices[0][0])
    top_score = float(distances[0][0])

    top_meta = metadata[top_idx] if 0 <= top_idx < len(metadata) else {}
    return {
        "db": db_name,
        "score": top_score,
        "top_index": top_idx,
        "top_title": safe_title(top_meta, db_name),
        "top_preview": safe_preview(top_meta, db_name),
    }


def classify_case(predicted_db: str, true_label: str, margin: float) -> str:
    if predicted_db != true_label:
        return "WRONG_DB"
    if margin < WEAK_MARGIN_THRESHOLD:
        return "WEAK_SEPARATION"
    if margin < STRONG_MARGIN_THRESHOLD:
        return "MODERATE_MATCH"
    return "GOOD_MATCH"


def build_test_queries():
    return [
        # syllabus
        {"query": "What are the units in Data Structures?", "label": "syllabus"},
        {"query": "What topics are covered in DBMS?", "label": "syllabus"},
        {"query": "What is the syllabus for Operating Systems?", "label": "syllabus"},
        {"query": "What are the course outcomes of Computer Networks?", "label": "syllabus"},
        {"query": "What textbooks are recommended for DBMS?", "label": "syllabus"},
        {"query": "How many credits does Data Structures carry?", "label": "syllabus"},

        # institutional
        {"query": "How to apply for hostel?", "label": "institutional"},
        {"query": "What is tuition fee?", "label": "institutional"},
        {"query": "What are the library timings?", "label": "institutional"},
        {"query": "Where is the placement cell?", "label": "institutional"},
        {"query": "What student clubs are available?", "label": "institutional"},
        {"query": "How to contact the transport department?", "label": "institutional"},

        # admin / regulations
        {"query": "Explain grading system", "label": "admin"},
        {"query": "What is examination passing criteria?", "label": "admin"},
        {"query": "What is the minimum attendance required to write semester exam?", "label": "admin"},
        {"query": "What happens if attendance is below 60 percent?", "label": "admin"},
        {"query": "Can a student withdraw from semester examination?", "label": "admin"},
        {"query": "What is the condonation rule for attendance shortage?", "label": "admin"},
        {"query": "How is GPA calculated?", "label": "admin"},
        {"query": "How is CGPA calculated?", "label": "admin"},
        {"query": "What is the difference between regular examination and arrear examination?", "label": "admin"},
        {"query": "Can a student drop an elective course?", "label": "admin"},
        {"query": "What is required for award of degree?", "label": "admin"},
        {"query": "Who can appeal continuous assessment marks?", "label": "admin"},
        {"query": "What are the rules for movement to higher semester?", "label": "admin"},
        {"query": "What is the classification for First Class with Distinction?", "label": "admin"},
    ]


def main():
    embed_config = load_embedding_config()
    model = SentenceTransformer(embed_config["model_name"], trust_remote_code=True)

    syllabus_index = load_index(SYLLABUS_INDEX)
    admin_index = load_index(ADMIN_INDEX)
    institutional_index = load_index(INSTITUTIONAL_INDEX)

    syllabus_meta = load_metadata(SYLLABUS_META)
    admin_meta = load_metadata(ADMIN_META)
    institutional_meta = load_metadata(INSTITUTIONAL_META)

    print("Loaded indexes:")
    print(f"  syllabus:      {syllabus_index.d if syllabus_index else None}")
    print(f"  admin:         {admin_index.d if admin_index else None}")
    print(f"  institutional: {institutional_index.d if institutional_index else None}")

    test_queries = build_test_queries()

    results = []
    correct_count = 0
    status_counts = {
        "GOOD_MATCH": 0,
        "MODERATE_MATCH": 0,
        "WEAK_SEPARATION": 0,
        "WRONG_DB": 0,
    }

    for item in test_queries:
        query = item["query"]
        true_label = item["label"]

        embedding = model.encode(
            [query],
            normalize_embeddings=embed_config["normalize"],
            convert_to_numpy=True
        ).astype("float32")

        db_hits = []

        syllabus_hit = search_db(syllabus_index, syllabus_meta, embedding, "syllabus")
        admin_hit = search_db(admin_index, admin_meta, embedding, "admin")
        institutional_hit = search_db(institutional_index, institutional_meta, embedding, "institutional")

        for hit in [syllabus_hit, admin_hit, institutional_hit]:
            if hit is not None:
                db_hits.append(hit)

        score_map = {hit["db"]: hit["score"] for hit in db_hits}
        sorted_hits = sorted(db_hits, key=lambda x: x["score"], reverse=True)

        predicted_db = sorted_hits[0]["db"]
        winner_score = sorted_hits[0]["score"]
        runner_up_score = sorted_hits[1]["score"] if len(sorted_hits) > 1 else 0.0
        margin = winner_score - runner_up_score

        is_correct = predicted_db == true_label
        if is_correct:
            correct_count += 1

        status = classify_case(predicted_db, true_label, margin)
        status_counts[status] += 1

        results.append({
            "query": query,
            "true_label": true_label,
            "predicted_db": predicted_db,
            "correct": is_correct,
            "status": status,
            "winner_score": round(winner_score, 6),
            "runner_up_score": round(runner_up_score, 6),
            "margin": round(margin, 6),
            "scores": {k: round(v, 6) for k, v in score_map.items()},
            "top_hits": {
                hit["db"]: {
                    "score": round(hit["score"], 6),
                    "title": hit["top_title"],
                    "preview": hit["top_preview"],
                }
                for hit in db_hits
            }
        })

    accuracy = (correct_count / len(test_queries)) * 100 if test_queries else 0.0

    report = {
        "total_queries": len(test_queries),
        "correct_predictions": correct_count,
        "accuracy_percent": round(accuracy, 2),
        "weak_margin_threshold": WEAK_MARGIN_THRESHOLD,
        "strong_margin_threshold": STRONG_MARGIN_THRESHOLD,
        "status_summary": status_counts,
        "details": results
    }

    reports_dir = PROD_ROOT / "data" / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / "cross_db_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\nRouting Accuracy: {accuracy:.2f}%")
    print("Status Summary:")
    for key, value in status_counts.items():
        print(f"  {key}: {value}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()