# 🧠 Multimodal Identity RAG

> A hybrid multimodal Retrieval-Augmented Generation system over personal documents (certificates, degrees, IDs) — searches across PDFs and images in a unified index, with both text content retrieval and CLIP-based visual similarity search.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-MultiModal-orange.svg)](https://www.llamaindex.ai/)
[![Local Embeddings](https://img.shields.io/badge/embeddings-local-green.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)

---

## ✨ What This Does

Ask natural language questions about your personal documents and get cited answers — whether the content lives in a PDF or inside an image.

**Example queries it handles:**
- *"What is the date on my AWS certificate?"* → finds Coursera_aws.pdf → "20-Dec-2020"
- *"Which university issued my degree?"* → finds DEGREE.jpeg caption → "Rajasthan Technical University Kota"
- *"Find the certificate signed by Andrew Ng"* → cross-document search → "Coursera_Structuring_ML_Projects.pdf"
- *"Show me images that look like certificates"* → CLIP visual similarity → displays matching images inline

---

## 🏗️ Architecture — Hybrid Multimodal RAG

```
PDFs    → PyMuPDF (+ Tesseract OCR fallback) → text  → BGE local embeddings  ─┐
Images  → Gemini Vision caption              → text  → BGE local embeddings  ─┤→ Text vector store
Images  → CLIP ViT-B/32 (parallel)           → 512-d → CLIP image embedding   ─→ Image vector store
                                                                                │
                                                  Query → intent router → ──────┘
                                                          ↓ text    ↓ visual
                                                      Gemini LLM    CLIP retrieval
                                                          ↓             ↓
                                                      Cited answer  Ranked images
```

**Why this works:**

| Component | Choice | Rationale |
|---|---|---|
| Text extraction | PyMuPDF + Tesseract OCR | Fast, local, handles both text-layer and image-only PDFs |
| Image content extraction | Gemini Vision captioning | Captures text inside images + entities + layout — solves CLIP's weakness on document images |
| Image visual matching | CLIP ViT-B/32 | Visual similarity for "looks like" queries |
| Text embeddings | BAAI/bge-small-en-v1.5 (local) | No API quota, 384-dim, top MTEB rank |
| Query LLM | Gemini 2.5 Flash | Free tier sufficient, called only at query time |
| Vector store | LlamaIndex MultiModalVectorStoreIndex | One DB, two namespaces (text + image) |
| Query router | Keyword-based intent detection | Visual queries → CLIP, content queries → Gemini |

---

## 📁 Project Structure

```
Shipment(GitHub)/
├── data/                              # Your personal documents
│   ├── Coursera_aws.pdf
│   ├── Coursera_Structuring_ML_Projects.pdf
│   ├── GCP.pdf
│   ├── DEGREE.jpeg
│   └── DL-Front.png
├── experiments/                       # Embedding exploration code
│   ├── exp_01_raw_embeddings.py       # how embeddings look
│   ├── exp_02_cosine_similarity.py    # cosine similarity + visualization
│   └── exp_03_chunking_feel.py        # chunking experiments
├── notebooks/
│   └── identity_rag_demo.ipynb        # Main demo notebook
├── src/
│   ├── config.py                      # LLM + embedding model setup
│   ├── loader.py                      # PDF + image extraction
│   ├── engine.py                      # Index build + retrieval + query
│   └── __init__.py
├── saved_index/                       # Persisted index (auto-generated)
├── .env.example                       # API key template
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/identity-rag.git
cd identity-rag/Shipment\(GitHub\)
python3 -m venv ragenv
source ragenv/bin/activate
pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR fallback)

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
LLAMA_CLOUD_API_KEY=optional_only_for_parser_demo_in_section_2
```

Get free Gemini API key: https://aistudio.google.com/apikey

### 4. Add your documents

Drop PDFs, JPEGs, and PNGs into `data/`. The system will auto-route by file type.

### 5. Run the notebook

```bash
cd notebooks
jupyter notebook identity_rag_demo.ipynb
```

Run cells top-to-bottom. First run builds the index (~30 seconds for 5 documents). Subsequent runs load from disk instantly.

---

## 💻 Usage

### From the notebook

```python
from src.config import setup_rag_settings
from src.loader import load_documents
from src.engine import build_index, ask_query

# 1. Setup models
mm_llm = setup_rag_settings()

# 2. Extract content from data/ folder
textual_data, vision_data = load_documents("../data/", mm_llm)

# 3. Build index (first run only)
index = build_index(textual_data, vision_data, "../saved_index")

# 4. Query
print(ask_query(index, "What is the date on my AWS certificate?"))
# → "20-Dec-2020 (Source: Coursera_aws.pdf)"

print(ask_query(index, "Which university issued my degree?"))
# → "Rajasthan Technical University Kota (Source: DEGREE.jpeg)"

ask_query(index, "Show me images that look like certificates")
# → displays matching images inline with CLIP similarity scores
```

### Subsequent runs (skip rebuild)

```python
from src.engine import load_index
index = load_index("../saved_index")  # instant, no re-embedding
```

---

## 🧪 Engineering Decisions

### Why PyMuPDF over LlamaParse or Docling?

Evaluated three parsers empirically on Apple Silicon M4:

| Parser | Outcome |
|---|---|
| **PyMuPDF + Tesseract OCR** | ✅ Selected — fast, local, CPU-based, handles text-layer + scanned PDFs |
| **LlamaParse** | ❌ Known async event-loop conflict in Jupyter — randomly drops files per batch |
| **Docling** | ❌ Layout model uses float64 dtype, incompatible with Apple Silicon MPS backend |

Section 2 of the notebook demonstrates each parser failing live with real error messages — no fake `print()` simulations.

### Why hybrid image processing (caption + CLIP)?

CLIP was trained on internet photo captions, not text-heavy documents. It captures visual style but not OCR-style text content. For certificates and IDs (where TEXT inside the image matters), pure CLIP retrieval misses queries like "find AWS certificate."

Solution: each image gets BOTH a Gemini Vision caption (rich text → BGE → text vector store) AND a CLIP embedding (visual → image vector store). Same image, two retrieval paths.

### Why bypass LlamaIndex's RetrieverQueryEngine?

Discovered empirically that `MultiModalVectorIndexRetriever.retrieve()` returns 0 nodes on hybrid indexes (built via `from_documents()` + `insert_nodes()`), despite embeddings existing correctly in the underlying SimpleVectorStore.

**Verified by querying the SimpleVectorStore directly** — returned 5 nodes with similarity scores 0.55–0.75. The bug is in LlamaIndex's wrapper, not our data.

Workaround: bypass the retriever, query the vector store directly via `VectorStoreQuery`, fetch nodes from docstore by ID, call `Settings.llm.complete()` with assembled context. Same end result, every step observable.

### Why local embeddings + cloud LLM?

- **BGE (text) + CLIP (image)** run locally on M4 GPU via MPS → no API rate limits during indexing, scales infinitely with hardware
- **Gemini 2.5 Flash** called only at query time (~5–10 calls per session) → free tier sufficient

This is the standard production pattern: cheap local indexing, cloud LLM only where it adds value.

---

## ⚠️ Known Limitations

- **Gemini free tier:** 20 requests/day for gemini-2.5-flash (Google reduced from earlier limits). Production use requires billing-enabled GCP project (~$1/month at typical personal-document RAG scale).
- **CLIP weakness on document images:** mitigated by parallel Gemini Vision captioning path, which captures text content inside images.
- **LlamaIndex retriever wrapper bug:** documented above. Workaround implemented in `engine.py`.
- **SimpleVectorStore:** in-memory + JSON persistence. Fine for ≤1000 documents. For production scale, migrate to Qdrant or Pinecone.
- **Apple Silicon MPS float64:** Docling layout model crashes on M-series Macs. Documented limitation, not a flaw of our chosen architecture.

---

## 🛣️ Roadmap (v2)

- [ ] Add `.pptx`, `.docx`, `.mp3` (Whisper STT), `.db` (SQL) extractors — same hybrid pattern, just new functions in `loader.py`
- [ ] Replace `SimpleVectorStore` with Qdrant for production-scale persistence
- [ ] Swap CLIP for ColPali or JinaCLIP v2 — document-aware image embeddings
- [ ] Add BGE-reranker before LLM synthesis for better retrieval precision
- [ ] FastAPI REST endpoint for app integration
- [ ] Streamlit UI for non-technical users

---

## 📊 Tech Stack

| Layer | Library |
|---|---|
| Vector index | LlamaIndex (MultiModalVectorStoreIndex) |
| Text embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Image embeddings | CLIP ViT-B/32 (`llama-index-embeddings-clip`) |
| PDF parsing | PyMuPDF (`fitz`) + Tesseract OCR |
| Image captioning | Google Gemini Vision (`google-genai` SDK) |
| Query LLM | Gemini 2.5 Flash |
| Agent framework | LlamaIndex FunctionAgent |
| Runtime | Python 3.13 on Apple Silicon M4 (CPU + MPS) |

---

## 🎓 What This Project Demonstrates

For interviewers — this project shows:

1. **Systematic parser evaluation** — three parsers tested empirically with real failure modes, final choice defended by data
2. **Hybrid multimodal architecture** — text + visual retrieval paths in unified vector database  
3. **Local + cloud cost-aware design** — eliminate rate limits where they hurt (indexing), use cloud where it adds value (generation)
4. **Production debugging skills** — diagnosed and bypassed a real LlamaIndex retriever bug by isolating each pipeline layer
5. **Hardware awareness** — caught Apple Silicon MPS float64 incompatibility before it caused silent failures
6. **Modular code** — clean separation between `config.py` (models), `loader.py` (extraction), `engine.py` (index + query), with reusable functions imported from `experiments/`
7. **Agentic tool use** — Gemini autonomously calls Python functions as tools (cosine similarity calculator, vector visualizer) based on natural language
8. **Honest documentation** — limitations stated openly, with workarounds documented and v2 roadmap defined

---

## 👤 Author

**Deepak Rathore** — Software / AI-ML Engineer at Bengaluru
- GitHub: [@AlgoDr](https://github.com/AlgoDr)
- LinkedIn: [linkedin.com/in/deepakrathore](https://www.linkedin.com/in/deepak-rathore-7b3b4718b/)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
