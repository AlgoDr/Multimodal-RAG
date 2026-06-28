"""
server.py — FastAPI REST API for Multimodal Identity RAG.

Endpoints:
    GET  /          — API info
    GET  /health    — server + index status
    POST /query     — ask a question, get cited answer
    POST /ingest    — add a new document to the index (no rebuild needed)

POC note: uses Gemini free tier (20 req/day, 5 req/min).
Quota resets daily at midnight Pacific Time (IST: 1:30 PM next day).
"""
import sys
import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# ── paths — resolved relative to THIS file ─────────────────────────────
# works regardless of where uvicorn is launched from
_API_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../api/
_ROOT_DIR = os.path.dirname(_API_DIR)                     # .../Shipment(V2)/

INDEX_PATH = os.path.join(_ROOT_DIR, "saved_index")
CHROMA_DB  = os.path.join(_ROOT_DIR, "chroma_db")
DATA_DIR   = os.path.join(_ROOT_DIR, "data")

# tell engine.py where ChromaDB lives (symlink: Shipment(V2)/chroma_db →
# notebooks/chroma_db)
os.environ["CHROMA_PATH_OVERRIDE"] = CHROMA_DB

load_dotenv(os.path.join(_ROOT_DIR, ".env"))
sys.path.insert(0, _ROOT_DIR)

# ── imports AFTER path + env setup ─────────────────────────────────────
from src.config import setup_rag_settings
from src.engine import load_index, ask_query
from src.loader import load_documents

# ── global state — set once in lifespan, reused for all requests ───────
index  = None
mm_llm = None

# ── quota error detection ───────────────────────────────────────────────
_QUOTA_PHRASES = [
    "quota", "429", "resource_exhausted",
    "rate limit", "too many requests", "exhausted"
]
_QUOTA_MESSAGE = (
    "⚠️ Gemini API quota exhausted (free tier: 20 requests/day, "
    "5 requests/minute).\n\n"
    "Quota resets daily at midnight Pacific Time (IST: 1:30 PM next day).\n\n"
    "Options:\n"
    "1. Retry tomorrow after quota resets\n"
    "2. Add your own GOOGLE_API_KEY to .env\n"
    "3. Enable GCP billing (~$0.10/month at personal scale)"
)

def _is_quota_error(e: Exception) -> bool:
    return any(p in str(e).lower() for p in _QUOTA_PHRASES)


# ── lifespan: runs ONCE at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load index at startup — not per request.

    Pattern: index loaded once into memory, reused for all requests.
    Loading inside the endpoint would reconnect ChromaDB on every call
    (~2s overhead). Lifespan guarantees one 2s startup, then instant queries.
    """
    global index, mm_llm
    print("🚀 Server starting — loading RAG index...")
    print(f"   Index path : {INDEX_PATH}")
    print(f"   ChromaDB   : {CHROMA_DB}")
    mm_llm = setup_rag_settings()   # sets Settings.embed_model (BGE) + Settings.llm (Gemini)
    index  = load_index(INDEX_PATH)
    print("✅ Index ready. Server accepting requests.")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="Multimodal Identity RAG",
    description=(
        "Query personal documents (certificates, IDs) via hybrid PDF+image RAG.\n\n"
        "**Text questions** → ChromaDB (BGE 384-dim) + Gemini LLM synthesis\n\n"
        "**Visual questions** → ChromaDB (CLIP 512-dim) similarity search\n\n"
        "**Ingest** → add new documents without rebuilding the index\n\n"
        "_POC on Gemini free tier: 20 requests/day, 5/minute._"
    ),
    version="2.0",
    lifespan=lifespan
)


# ── request / response models ───────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str


# ── endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Multimodal RAG API is running",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "GET  /health": "server + index status",
            "POST /query" : "ask a question about your documents",
            "POST /ingest": "add a new document to the index",
        }
    }


@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "index_loaded": index is not None,
        "index_path"  : INDEX_PATH,
        "chroma_path" : CHROMA_DB,
    }


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question about your personal documents.

    - **Text questions** → BGE embeds query → ChromaDB finds top-7 chunks
      → Gemini synthesizes cited answer
    - **Visual questions** (contains: *show me*, *looks like*, *display*,
      *find image*) → CLIP embeds query → ChromaDB image store → ranked files

    **POC note:** uses Gemini free tier (20 req/day, 5 req/min).
    Quota resets daily at midnight Pacific Time (IST: 1:30 PM next day).
    """
    if index is None:
        return QueryResponse(
            question=request.question,
            answer="Index not loaded. Restart the server and check startup logs."
        )

    try:
        result = ask_query(index, request.question)

    except Exception as e:
        if _is_quota_error(e):
            return QueryResponse(question=request.question, answer=_QUOTA_MESSAGE)
        return QueryResponse(
            question=request.question,
            answer=f"❌ Unexpected error: {type(e).__name__}: {str(e)[:200]}"
        )

    # normalize return types:
    # _text_query  → str  (Gemini cited answer)
    # _visual_query → list of (file_name, similarity) tuples
    if isinstance(result, list):
        if not result:
            answer = "No visually similar images found for your query."
        else:
            lines = [
                f"{i+1}. {fn} (CLIP similarity: {sim:.3f})"
                for i, (fn, sim) in enumerate(result)
            ]
            answer = "Visually similar images:\n" + "\n".join(lines)
    else:
        answer = str(result)

    return QueryResponse(question=request.question, answer=answer)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Add a new document to the index — **no rebuild required**.

    ChromaDB supports incremental inserts: new vectors are added to the
    existing HNSW graph without touching other documents. The updated index
    is immediately available for `/query` with no restart needed.

    **PDF** (.pdf)
    - PyMuPDF extracts text locally (+ Tesseract OCR for scanned PDFs)
    - BGE embeds locally → ChromaDB text store
    - ✅ Works even when Gemini quota is exhausted

    **Image** (.png .jpg .jpeg)
    - Gemini Vision generates caption → BGE embeds → ChromaDB text store
    - CLIP embeds image locally → ChromaDB image store
    - ⚠️ Requires Gemini quota. Returns clear error if quota is exhausted.

    Supported: `.pdf` `.png` `.jpg` `.jpeg`
    """
    if index is None:
        return {"status": "error", "message": "Index not loaded."}

    # validate file type
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg"}
    fname = file.filename or ""
    ext   = os.path.splitext(fname)[1].lower()

    if ext not in allowed_exts:
        return {
            "status" : "error",
            "message": (
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(allowed_exts))}"
            )
        }

    is_image = ext in {".png", ".jpg", ".jpeg"}

    # save uploaded file to data/
    save_path = os.path.join(DATA_DIR, fname)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {e}"}

    # process single file — loader raises on quota error for images
    # so we can return a specific message to the user
    try:
        new_text, new_images = load_documents(
            DATA_DIR,
            mm_llm=mm_llm,
            single_file=fname
        )

    except Exception as e:
        # clean up saved file so data/ stays in sync with the index
        if os.path.exists(save_path):
            os.remove(save_path)

        if _is_quota_error(e) and is_image:
            return {
                "status": "quota_error",
                "file"  : fname,
                "message": (
                    "⚠️ Cannot ingest image — Gemini Vision quota exhausted.\n\n"
                    "Image captioning requires Gemini API "
                    "(free tier: 20 req/day, 5 req/min).\n"
                    "Quota resets at midnight Pacific Time (IST: 1:30 PM next day).\n\n"
                    "✅ PDFs can still be ingested — they use local BGE only, "
                    "no Gemini quota needed."
                )
            }

        return {
            "status" : "error",
            "file"   : fname,
            "message": f"{type(e).__name__}: {str(e)[:300]}"
        }

    # insert into existing in-memory index
    # ChromaDB writes to disk automatically — no manual persist() needed
    # the updated index is immediately queryable via /query
    inserted_text   = 0
    inserted_images = 0

    try:
        if new_text:
            for doc in new_text:
                index.insert(doc)
            inserted_text = len(new_text)

        if new_images:
            index.insert_nodes(new_images)
            inserted_images = len(new_images)

    except Exception as e:
        return {
            "status" : "partial_error",
            "file"   : fname,
            "message": f"File saved but indexing failed: {type(e).__name__}: {str(e)[:200]}"
        }

    return {
        "status"              : "success",
        "file"                : fname,
        "file_type"           : "image" if is_image else "pdf",
        "text_chunks_added"   : inserted_text,
        "image_vectors_added" : inserted_images,
        "index_updated"       : True,
        "rebuild_required"    : False,
        "note": (
            "Image ingested: Gemini Vision caption → BGE → ChromaDB text store "
            "+ CLIP → ChromaDB image store."
            if is_image else
            "PDF ingested: PyMuPDF → BGE → ChromaDB text store. "
            "No Gemini quota used."
        )
    }