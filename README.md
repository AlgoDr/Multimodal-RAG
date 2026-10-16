# 🧠 Multimodal Identity RAG — V3

> A hybrid multimodal Retrieval-Augmented Generation system over personal documents (certificates, degrees, IDs) — persistent ChromaDB vector storage, hybrid PDF+image retrieval, FastAPI REST endpoint, Gradio chat GUI, incremental document ingestion, and Docker packaging.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-MultiModal-orange.svg)](https://www.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/vectordb-ChromaDB-blue.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://www.gradio.app/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Local Embeddings](https://img.shields.io/badge/embeddings-local-green.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)

> **V1** (notebook-only, SimpleVectorStore) → [see v1.0.0 release](https://github.com/AlgoDr/Multimodal-RAG/releases/tag/v1.0.0)
> **V2** (ChromaDB + FastAPI REST API + incremental ingest) → V2 commits on `v2` branch
> **V3** (this release) → Gradio chat GUI + Docker packaging + fresh `rag-env-v3`

---

## ✨ What This Does

Ask natural language questions about your personal documents and get cited answers — whether the content lives in a PDF text layer or inside an image. Chat via Gradio GUI, query via notebook OR via REST API, run everything in Docker. Add new documents at runtime without rebuilding the index.

**Real queries that work:**

| Query | Retrieval Path | Answer |
|---|---|---|
| *"What is the date on my AWS certificate?"* | PDF text → BGE → ChromaDB | "20-Dec-2020 (Source: Coursera_aws.pdf)" |
| *"Find the certificate signed by Andrew Ng"* | Image caption → BGE → ChromaDB | "Two certificates — Stanford ML + Coursera Structuring ML" |
| *"List all my certificates with dates"* | Cross-document aggregation | Lists all certificates with dates + sources |
| *"Show me images that look like certificates"* | CLIP → ChromaDB image store | Ranked images with similarity scores |
| *Upload new cert via GUI or /ingest* | Incremental ChromaDB insert | Immediately queryable — no rebuild, no restart |
| *`docker compose up` → open /ui* | CPU-torch container + volumes | Same RAG, portable — healthchecked, persistent |

---

## 🏗️ Architecture — Hybrid Multimodal RAG

```
PDFs    → PyMuPDF (+ Tesseract OCR fallback) → text   → BGE local embeddings  ─┐
Images  → Gemini Vision caption              → text   → BGE local embeddings  ─┤→ ChromaDB 'rag_text_store'  (384-dim)
Images  → CLIP ViT-B/32 (parallel)           → 512-d  → CLIP image embedding   ─→ ChromaDB 'rag_image_store' (512-dim)
                                                                                   │
                                                   Query → intent router → ────────┘
                                                           ↓ text      ↓ visual
                                                       Gemini LLM    CLIP retrieval
                                                           ↓              ↓
                                                       Cited answer   Ranked images
                                                           ↓
                                                   FastAPI POST /query → JSON response

New document → POST /ingest → single-file extraction → ChromaDB incremental insert
                                                     → immediately queryable
```

**Design decisions:**

| Component | Choice | Rationale |
|---|---|---|
| Text extraction | PyMuPDF + Tesseract OCR | Fast, local, handles text-layer + scanned PDFs |
| Image content extraction | Gemini Vision captioning | Captures text inside images — solves CLIP's weakness on document images |
| Image visual matching | CLIP ViT-B/32 | Visual similarity for "looks like" queries |
| Text embeddings | `BAAI/bge-small-en-v1.5` (local) | No API quota during indexing, 384-dim, top MTEB rank |
| Image embeddings | CLIP ViT-B/32 (local) | 512-dim joint vision-language space |
| Vector store | ChromaDB (2 collections, cosine metric) | Persistent, incremental inserts, proper DB semantics |
| Query LLM | Gemini 2.5 Flash | Free tier sufficient, called only at query time |
| API layer | FastAPI + uvicorn | Auto-docs, Pydantic validation, lifespan pattern |
| Chat GUI | Gradio 6 (`ui/app.py`, mounted at `/ui`) | Chat + CLIP gallery + upload; shares the lifespan index, no second backend |
| Packaging | Docker + compose (CPU torch, volumes) | `identity-rag:v3` — portable, healthchecked, persistent ingest |
| Query router | Keyword-based intent detection | Visual queries → CLIP, content queries → Gemini |

---

## 📁 Project Structure

```
Shipment(V3)/
├── api/
│   └── server.py                      # FastAPI — POST /query, POST /ingest, GET /health; mounts Gradio at /ui
├── ui/
│   ├── app.py                         # Gradio chat + CLIP gallery + upload (standalone :7860 or mounted /ui)
│   └── __init__.py
├── Dockerfile                         # python:3.13-slim, CPU torch, non-root, single uvicorn process
├── docker-compose.yml                 # volumes: data/, chroma_data/; env: GOOGLE_API_KEY + paths
├── requirements.docker.txt            # Linux/Docker variant (torch from CPU index, + chromadb/uvicorn)
├── data/                              # Personal documents (certificates)
│   ├── Coursera_aws.pdf
│   ├── Coursera_Structuring_ML_Projects.pdf
│   ├── GCP.pdf                        # Scanned PDF — tests OCR fallback
│   ├── STANFORD ONLINE.jpg            # Image cert — tests Gemini Vision captioning
│   ├── Google Technical Fundamentals.png
│   ├── AZURE -ML STUDIO.png
│   └── GEN_AI-COURSERA.jpeg
├── experiments/                       # Embedding exploration utilities
│   ├── exp_01_raw_embeddings.py       # what embeddings look like
│   ├── exp_02_cosine_similarity.py    # cosine similarity + 3D visualization
│   └── exp_03_chunking_feel.py        # chunking exploration
├── notebooks/
│   ├── RAG(V2+chromadb).ipynb         # V2 demo — ChromaDB integration walkthrough
│   ├── identity_rag_demo.ipynb        # V1 demo — full pipeline walkthrough
│   └── test_notebook.ipynb            # debugging + diagnostics (dev reference)
├── src/
│   ├── config.py                      # BGE + CLIP + Gemini setup
│   ├── loader.py                      # PDF + image extraction (retry + single-file mode)
│   ├── engine.py                      # ChromaDB index + retrieval + query router (env-var paths)
│   └── __init__.py
├── chroma_data/                       # local container volume (gitignored; seeded from notebooks/chroma_db)
├── .env.example                       # API key template
├── requirements.txt                   # local venv pins (+ gradio, llama-index-vector-stores-chroma)
└── README.md
```

---

## 🚀 Setup

### 1. Clone and install

```bash
git clone https://github.com/AlgoDr/Multimodal-RAG.git
cd Multimodal-RAG
git checkout v3
python3.13 -m venv rag-env-v3
source rag-env-v3/bin/activate      # Windows: rag-env-v3\Scripts\activate
pip install -r requirements.txt
# needs git on PATH (CLIP installs from a git URL) + chroma store pin included
```

### 2. Install Tesseract (for OCR fallback on scanned PDFs)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required — Gemini for image captioning + query LLM
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — only for LlamaParse failure demo in Section 2 of notebook
LLAMA_CLOUD_API_KEY=optional
```

Get a free Gemini API key: https://aistudio.google.com/apikey

### 4. Add your documents

Drop PDFs, JPEGs, and PNGs into `data/`. The system auto-routes by file type — no config needed.

### 5. Build the index (first run only)

```bash
cd notebooks
jupyter notebook RAG\(V2+chromadb\).ipynb
# run all cells top to bottom — creates chroma_db/ and saved_index/
```

Or from Python:

```python
from src.config import setup_rag_settings
from src.loader import load_documents
from src.engine import build_index

mm_llm = setup_rag_settings()
textual_data, vision_data = load_documents("../data/", mm_llm)
index = build_index(textual_data, vision_data, "../saved_index")
```

---

## 💻 Usage — Four Ways

### Option A: Gradio chat GUI (recommended)

```bash
# from repo root, venv active
python ui/app.py
# → http://localhost:7860 — chat + visual gallery + upload box
```

Or via the API server (same GUI, shared backend): `uvicorn api.server:app --port 8000` → `http://localhost:8000/ui/`.

### Option B: Notebook

```python
from src.engine import load_index, ask_query

index = load_index("../saved_index")   # instant — no re-embedding

print(ask_query(index, "What is the date on my AWS certificate?"))
# → "20-Dec-2020 (Source: Coursera_aws.pdf)"

print(ask_query(index, "Find the certificate signed by Andrew Ng"))
# → "Stanford ML certificate + Coursera Structuring ML (both cited)"

ask_query(index, "Show me images that look like certificates")
# → displays ranked images inline with CLIP similarity scores
```

### Option C: FastAPI REST endpoint

```bash
uvicorn api.server:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for interactive Swagger UI — no code needed to test any endpoint.

### Option D: Docker (portable)

```bash
export GOOGLE_API_KEY=your_gemini_api_key_here
docker compose up --build
# → GUI http://localhost:8000/ui/ · API http://localhost:8000/docs
# volumes persist data/, chroma vectors and index jsons across restarts
```

---

## 🔌 API Endpoints

### `GET /health`
Server status and index load confirmation.

```json
{
  "status": "ok",
  "index_loaded": true,
  "index_path": "...saved_index",
  "chroma_path": "...chroma_db"
}
```

### `POST /query`
Ask a question about your documents. Auto-routes by intent.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the date on my AWS certificate?"}'
```

```json
{
  "question": "What is the date on my AWS certificate?",
  "answer": "The date on your AWS certificate is 20-Dec-2020. (Source: Coursera_aws.pdf)"
}
```

Returns a quota-aware error message when Gemini free tier is exhausted — never a raw stack trace.

### `POST /ingest`
Add a new document to the index **without rebuilding**.

ChromaDB supports incremental inserts — the new vector is added to the existing HNSW graph without touching other documents. The updated index is immediately available for `/query` with no restart.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/new_certificate.pdf"
```

```json
{
  "status": "success",
  "file": "new_certificate.pdf",
  "file_type": "pdf",
  "text_chunks_added": 1,
  "image_vectors_added": 0,
  "index_updated": true,
  "rebuild_required": false,
  "note": "PDF ingested: PyMuPDF → BGE → ChromaDB text store. No Gemini quota used."
}
```

**PDF vs Image ingestion:**

| File type | Gemini needed? | Works when quota exhausted? |
|---|---|---|
| `.pdf` | ❌ No — PyMuPDF + BGE only | ✅ Yes |
| `.png` `.jpg` `.jpeg` | ✅ Yes — Gemini Vision caption | ❌ Returns clear quota error |

When an image is uploaded and Gemini quota is exhausted, the API returns:
```json
{
  "status": "quota_error",
  "message": "⚠️ Cannot ingest image — Gemini Vision quota exhausted. PDFs can still be ingested..."
}
```

---

## 🧪 Engineering Decisions

### 1. Why ChromaDB over SimpleVectorStore?

V1 used LlamaIndex's built-in `SimpleVectorStore` — an in-memory Python dict serialized to JSON. It works for demos but has real limitations:

| | SimpleVectorStore (V1) | ChromaDB (V2) |
|---|---|---|
| Persistence | Manual `.persist()` → JSON | Automatic — SQLite + HNSW on disk |
| Restart cost | Rebuild from scratch | Reconnect in milliseconds |
| Incremental insert | Not supported — full rebuild | `collection.add()` — HNSW updates in place |
| Metadata filtering | Not supported | `where={"source_type": "pdf"}` |
| Concurrent access | Single process only | Multi-client safe |
| Search algorithm | Brute force O(n) | HNSW graph O(log n) |

ChromaDB uses SQLite for metadata storage and an HNSW graph for vector search — O(log n) vs O(n). At 1M vectors, brute force takes seconds; HNSW takes milliseconds.

### 2. Why two ChromaDB collections?

```
rag_text_store   (BGE 384-dim)  ← PDFs + image captions
rag_image_store  (CLIP 512-dim) ← raw images for visual search
```

ChromaDB enforces dimension consistency within a collection. BGE produces 384-dim vectors, CLIP produces 512-dim vectors — they can't share a collection. Beyond that, they represent different embedding spaces (semantic text vs joint vision-language) — mixing them produces meaningless similarity scores.

### 3. Why bypass LlamaIndex's RetrieverQueryEngine?

`MultiModalVectorIndexRetriever.retrieve()` returns 0 nodes on hybrid builds (`from_documents` + `insert_nodes`), despite embeddings existing in the underlying store. Verified empirically by querying the ChromaDB collection directly — returned correct results. The bug is in LlamaIndex's wrapper dispatch, not the data.

Workaround: `VectorStoreQuery` directly on the ChromaDB store, bypassing the broken retriever.

### 4. Why read text from ChromaDB, not LlamaIndex's docstore?

When using an external vector store (ChromaDB), LlamaIndex stores node text INSIDE ChromaDB's `documents` field — the separate docstore stays empty. First discovered when `docstore.get_node()` raised `ValueError: doc_id not found` for every node.

Diagnosed by inspecting both stores: `docstore.docs` → 0 nodes, `collection.count()` → 7 vectors. Fix: read text from `collection.get(ids=..., include=['documents', 'metadatas'])` directly.

### 5. Why cosine metric, not L2?

ChromaDB defaults to L2 (Euclidean) distance. CLIP vectors are not unit-normalized (norm ~8–10), so L2 produces near-zero similarity values (`4e-57`).

Fix: `metadata={"hnsw:space": "cosine"}` at collection creation. Verified: manual cosine calculation returned 0.28–0.31; ChromaDB returned `4e-57` with L2. After fix, scores match.

**Important:** existing collections can't change their distance metric. Must wipe and rebuild after changing this setting.

### 6. Why exponential backoff on image captioning?

Gemini Vision returns `503 UNAVAILABLE` during high demand, causing inconsistent indexing. Added retry with exponential backoff: 2s → 4s → 8s → 16s, up to 4 attempts. Non-503 errors fail immediately (no point retrying a real bug).

In single-file ingest mode, caption failures re-raise so the API can detect quota errors and return a user-friendly message. In full build mode, failures are logged and the build continues.

### 7. Why FastAPI lifespan pattern?

The index loads ONCE at server startup via `@asynccontextmanager async def lifespan()`, not per request. Loading inside the endpoint would reconnect to ChromaDB on every call (~2s overhead). Lifespan guarantees one startup cost, then instant requests.

### 8. Why PyMuPDF over LlamaParse or Docling?

Evaluated three parsers empirically on Apple Silicon M4:

| Parser | Outcome |
|---|---|
| **PyMuPDF + Tesseract OCR** | ✅ Selected — fast, local, handles text-layer + scanned PDFs |
| **LlamaParse** | ❌ Async event-loop conflict in Jupyter — randomly drops 1-2 files per batch |
| **Docling** | ❌ Layout model (RT-DETR v2) uses `float64`, incompatible with Apple Silicon MPS backend |

Section 2 of the notebook runs each failing parser live — real error messages, no fake `print()` simulations.

---

## ⚠️ Known Limitations

- **Gemini free tier:** 20 requests/day, 5 requests/minute. Quota resets daily at midnight Pacific Time (IST: 1:30 PM next day). API returns a clear message on exhaustion — never a raw error. Production use requires GCP billing (~$0.10/month at personal scale).
- **Image ingest requires Gemini:** uploading images via `/ingest` when quota is exhausted returns a `quota_error` response. PDFs always work — they use local BGE only.
- **CLIP on document images:** CLIP was trained on photo captions, not certificates. Mitigated by parallel Gemini Vision captioning path.
- **Keyword-based visual router:** "show me", "looks like" trigger CLIP mode. Ambiguous queries like "show me proof" may route incorrectly. A future version would use LLM-based intent classification.
- **Named entity retrieval:** BGE-small produces tightly clustered scores on semantically similar documents. Named entity queries (e.g. "Andrew Ng") may rank low with small `top_k`. Fixed by setting `top_k=7` for this dataset.
- **LlamaIndex retriever bug:** documented above. Workaround in `engine.py`.

---

## 🛣️ Roadmap

- [x] ~~Replace SimpleVectorStore with ChromaDB~~ (V2)
- [x] ~~FastAPI REST endpoint~~ (V2)
- [x] ~~Incremental document ingestion via /ingest~~ (V2)
- [x] ~~Quota-aware error handling~~ (V2)
- [x] ~~Gradio web UI — chat + CLIP gallery + upload, mounted at /ui~~ (V3)
- [x] ~~Docker packaging — CPU-torch image + compose volumes~~ (V3)
- [x] ~~Fresh `rag-env-v3` (python3.13) + missing chroma-store pin~~ (V3)
- [ ] BGE-reranker — cross-encoder reranking for named entity precision
- [ ] ColPali or JinaCLIP v2 — document-aware image embeddings
- [ ] Add `.pptx`, `.docx`, `.mp3` (Whisper STT) extractors
- [ ] Deploy to AWS (EC2 + ECR) — replaces Railway/Render target

---

## 📊 Tech Stack

| Layer | Library |
|---|---|
| Vector database | ChromaDB (SQLite + HNSW, cosine metric) |
| Text embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (local) |
| Image embeddings | CLIP ViT-B/32 (local, `llama-index-embeddings-clip`) |
| PDF parsing | PyMuPDF (`fitz`) + Tesseract OCR |
| Image captioning | Google Gemini Vision (`google-genai` SDK) |
| Query LLM | Gemini 2.5 Flash |
| RAG framework | LlamaIndex `MultiModalVectorStoreIndex` |
| REST API | FastAPI + uvicorn |
| Chat GUI | Gradio 6 |
| Packaging | Docker (python:3.13-slim, CPU torch) + compose |
| File upload | python-multipart |
| Request validation | Pydantic v2 |
| Local runtime | Python 3.13 venv `rag-env-v3`, Apple Silicon M4 (CPU + MPS) |

---

## 🆕 What Changed in V3 (vs V2)

| Area | V2 | V3 |
|---|---|---|
| Interface | API + notebooks only | + Gradio chat GUI (`ui/app.py`) — chat, CLIP gallery, upload box; mounted at `/ui` sharing the lifespan index |
| Portability | Local venv only | + Docker (`identity-rag:v3`, 4.16GB, CPU torch) + compose with persistent `data/` + `chroma_data/` volumes, healthcheck |
| Env | `ragenv` (broken torch `libtorch_cpu.dylib`) | Fresh `rag-env-v3` (python3.13) — torch/CLIP/chroma verified; `requirements.txt` += `gradio`, `llama-index-vector-stores-chroma==0.5.5` (was missing from V2 freeze) |
| Config | Hardcoded `saved_index`/`chroma_db`/`data` paths | `INDEX_PATH` / `CHROMA_PATH_OVERRIDE` / `DATA_DIR` env overrides in `server.py` + `ui/app.py` (container-ready) |
| Retrieval | Same dual-path BGE + CLIP, `top_k=7` | Unchanged — V3 adds interface + packaging only, no retrieval logic touched |

```bash
# V3 quickstart (local)
git checkout v3 && source rag-env-v3/bin/activate
python ui/app.py                      # GUI :7860
uvicorn api.server:app --port 8000    # API + /ui on :8000

# V3 quickstart (docker)
export GOOGLE_API_KEY=... && docker compose up --build   # → :8000/ui
```

---

## 🎓 What This Project Demonstrates

1. **Vector database selection and integration** — migrated from in-memory SimpleVectorStore to ChromaDB; understands HNSW vs brute force, L2 vs cosine, collection design, incremental inserts
2. **Production debugging** — diagnosed 4 real bugs (docstore empty, CLIP 0.000 similarity, retriever 0 nodes, top_k miss) by inspecting each pipeline layer independently
3. **REST API design** — FastAPI with lifespan startup pattern, Pydantic models, file upload endpoint, quota-aware error handling
4. **Hybrid multimodal architecture** — text + visual retrieval paths with intent-based routing, two embedding models, two ChromaDB collections
5. **Incremental ingestion** — `/ingest` endpoint adds documents to ChromaDB without rebuilding; PDFs work without Gemini quota; images return clear quota error when exhausted
6. **Resilient ingestion** — exponential backoff retry on transient 503s; single-file mode re-raises for API quota detection; full build mode logs and continues
7. **Local + cloud cost-aware design** — local BGE + CLIP for indexing (no quota), Gemini only at query time and image captioning
8. **Systematic parser evaluation** — LlamaParse async bug + Docling MPS float64 incompatibility reproduced live; PyMuPDF selected on evidence
9. **Honest documentation** — limitations stated with root causes, workarounds documented, known bugs listed with fixes
10. **Interface + packaging** — Gradio GUI sharing one backend (no duplicate index load), Dockerized CPU-torch image with persistent volumes, container-tested end to end

---

## 🔒 Privacy Note

A personal degree certificate was used during development to validate the pipeline on real identity documents (bilingual Hindi/English text, official seals, embedded photographs). It has been removed from the public repo. The included Coursera, Google Cloud, Stanford, Azure, and GEN_AI certificates are already public on the author's LinkedIn.

`saved_index/` and `chroma_db/` are gitignored — they contain indexed text in plain JSON/SQLite. Anyone cloning the repo runs `build_index()` to generate their own index from their own documents.

---

## 👤 Author

**Deepak Rathore** — AI/ML Engineer at Bengaluru
- GitHub: [@AlgoDr](https://github.com/AlgoDr)
- LinkedIn: [linkedin.com/in/deepak-rathore-7b3b4718b](https://www.linkedin.com/in/deepak-rathore-7b3b4718b/)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.