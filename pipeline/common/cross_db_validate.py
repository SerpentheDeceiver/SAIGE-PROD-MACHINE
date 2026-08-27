import faiss
from pathlib import Path

# Anchor paths to project root regardless of CWD
PROD_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_DB = PROD_ROOT / "data" / "vector_db"

admin         = faiss.read_index(str(VECTOR_DB / "admin_faiss"        / "faiss.index"))
syllabus      = faiss.read_index(str(VECTOR_DB / "syllabus_faiss"     / "faiss.index"))
institutional = faiss.read_index(str(VECTOR_DB / "institutional_faiss" / "faiss.index"))

print("Admin:        ", admin.d,         "dims |", admin.ntotal,         "vectors")
print("Syllabus:     ", syllabus.d,      "dims |", syllabus.ntotal,      "vectors")
print("Institutional:", institutional.d, "dims |", institutional.ntotal, "vectors")