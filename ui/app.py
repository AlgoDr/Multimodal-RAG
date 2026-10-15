"""
app.py — Gradio GUI for Multimodal Identity RAG (V3), part 1: backend.

Standalone:  python ui/app.py          → http://localhost:7860
Mounted:    imported by api/server.py → served at /ui

The UI calls the SAME backend as the API (src.engine.ask_query,
src.loader.load_documents) — no retrieval logic duplicated here.
"""
import os
import sys
import shutil

import gradio as gr
from dotenv import load_dotenv

_UI_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ui/
_ROOT_DIR = os.path.dirname(_UI_DIR)                     # .../Shipment(V3)/

INDEX_PATH = os.environ.get("INDEX_PATH", os.path.join(_ROOT_DIR, "saved_index"))
CHROMA_DB = os.environ.get("CHROMA_PATH_OVERRIDE",
                           os.path.join(_ROOT_DIR, "chroma_db"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_ROOT_DIR, "data"))

# tell engine.py where ChromaDB lives (same override server.py uses)
os.environ["CHROMA_PATH_OVERRIDE"] = CHROMA_DB

load_dotenv(os.path.join(_ROOT_DIR, ".env"))
sys.path.insert(0, _ROOT_DIR)

_index = None
_mm_llm = None
_EXTERNAL_GET_INDEX = None
_EXTERNAL_GET_MM_LLM = None

_QUOTA_PHRASES = [
    "quota", "429", "resource_exhausted",
    "rate limit", "too many requests", "exhausted",
]
_QUOTA_MESSAGE = (
    "Gemini API quota exhausted (free tier: 20 requests/day, "
    "5 requests/minute). Quota resets daily at midnight Pacific Time "
    "(IST: 1:30 PM next day)."
)


def _is_quota_error(e: Exception) -> bool:
    return any(p in str(e).lower() for p in _QUOTA_PHRASES)


def attach_external(get_index, get_mm_llm=None):
    """Mounted mode: share the FastAPI lifespan index, not a second copy."""
    global _EXTERNAL_GET_INDEX, _EXTERNAL_GET_MM_LLM
    _EXTERNAL_GET_INDEX = get_index
    _EXTERNAL_GET_MM_LLM = get_mm_llm


def ensure_backend():
    """Load the RAG backend once (standalone). No-op if already loaded."""
    global _index, _mm_llm
    if _index is not None:
        return _index, _mm_llm
    from src.config import setup_rag_settings
    from src.engine import load_index
    print("Gradio UI — loading RAG backend...")
    print(f"   Index path : {INDEX_PATH}")
    print(f"   ChromaDB   : {CHROMA_DB}")
    _mm_llm = setup_rag_settings()
    _index = load_index(INDEX_PATH)
    print("Backend ready.")
    return _index, _mm_llm


def _get_backend():
    """Current (index, mm_llm) — external shared index wins when available."""
    if _EXTERNAL_GET_INDEX is not None:
        idx = _EXTERNAL_GET_INDEX()
        if idx is not None:
            mm = _EXTERNAL_GET_MM_LLM() if _EXTERNAL_GET_MM_LLM else None
            return idx, mm
    return ensure_backend()


def _resolve_image_path(file_name):
    """Map a retrieved file_name to a servable path inside DATA_DIR."""
    if not file_name or file_name == "unknown":
        return None
    candidate = os.path.join(DATA_DIR, file_name)
    return candidate if os.path.exists(candidate) else None


def answer_question(message):
    """Ask the RAG backend. Returns (reply_text, gallery_items)."""
    index, _ = _get_backend()
    if index is None:
        return "Index not loaded. Restart the app.", []

    from src.engine import ask_query
    try:
        result = ask_query(index, message)
    except Exception as e:
        if _is_quota_error(e):
            return _QUOTA_MESSAGE, []
        return f"Unexpected error: {type(e).__name__}: {str(e)[:300]}", []

    if isinstance(result, list):
        if not result:
            return "No visually similar images found.", []
        lines, gallery = [], []
        for i, (fn, sim) in enumerate(result, 1):
            lines.append(f"{i}. {fn} (CLIP similarity: {sim:.3f})")
            path = _resolve_image_path(fn)
            if path:
                gallery.append((path, f"{fn} ({sim:.3f})"))
        return "Visually similar images:\n" + "\n".join(lines), gallery

    return str(result), []


def chat_submit(message, history):
    """Chat handler — appends user + assistant turns (messages format)."""
    history = history or []
    if not (message or "").strip():
        return "", history, []
    reply, gallery = answer_question(message.strip())
    history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": reply},
    ]
    return "", history, gallery


def ingest_file(file_obj):
    """Ingest one uploaded file — same incremental path as POST /ingest."""
    if file_obj is None:
        return "Choose a .pdf, .png, .jpg or .jpeg file first."

    src_path = file_obj if isinstance(file_obj, str) else getattr(
        file_obj, "name", None)

    orig = getattr(file_obj, "orig_name", None) or (
        os.path.basename(src_path) if src_path else "")
    ext = os.path.splitext(orig)[1].lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg"}
    if ext not in allowed:
        return f"Unsupported file type '{ext}'. Allowed: pdf, png, jpg, jpeg."

    if not src_path or not os.path.exists(src_path):
        return "Upload failed — temp file not found. Try again."

    index, mm_llm = _get_backend()
    if index is None:
        return "Index not loaded. Restart the app."

    save_path = os.path.join(DATA_DIR, orig)
    try:
        if os.path.abspath(src_path) != os.path.abspath(save_path):
            shutil.copy(src_path, save_path)
    except Exception as e:
        return f"Failed to save file: {e}"

    is_image = ext in {".png", ".jpg", ".jpeg"}
    from src.loader import load_documents
    try:
        new_text, new_images = load_documents(
            DATA_DIR, mm_llm=mm_llm, single_file=orig
        )
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        if _is_quota_error(e) and is_image:
            return ("Cannot ingest image — Gemini Vision quota exhausted. "
                    "PDFs can still be ingested (local BGE only, no quota).")
        return f"Extraction failed: {type(e).__name__}: {str(e)[:300]}"

    try:
        n_text = n_img = 0
        if new_text:
            for doc in new_text:
                index.insert(doc)
            n_text = len(new_text)
        if new_images:
            index.insert_nodes(new_images)
            n_img = len(new_images)
    except Exception as e:
        return f"File saved but indexing failed: {type(e).__name__}: {str(e)[:200]}"

    kind = "image" if is_image else "pdf"
    return (f"{orig} ingested ({kind}): {n_text} text chunk(s), "
            f"{n_img} image vector(s). Ask about it right away.")


EXAMPLES = [
    "What is the date on my AWS certificate?",
    "Which certificates are signed by Andrew Ng?",
    "List all my certificates with dates",
    "Show me images that look like certificates",
]


def build_demo():
    """Build the Gradio Blocks UI (shared by standalone + mounted mode)."""
    with gr.Blocks(title="Multimodal Identity RAG") as demo:
        gr.Markdown("# Multimodal Identity RAG\n"
                    "Ask about your certificates, degrees and IDs — "
                    "content lives in PDFs and inside images.")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Answers (cited)")
                msg = gr.Textbox(label="Your question",
                                 placeholder="e.g. What is the date on my AWS certificate?")
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear chat")
                gr.Examples(examples=EXAMPLES, inputs=msg)
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Visual matches (CLIP)",
                                     columns=2, height="auto")
        with gr.Accordion("Add a document (no rebuild needed)", open=False):
            upload = gr.File(label="PDF or image (.pdf .png .jpg .jpeg)",
                             file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            ingest_btn = gr.Button("Ingest file")
            ingest_out = gr.Markdown()

        ask_btn.click(chat_submit, inputs=[msg, chatbot],
                      outputs=[msg, chatbot, gallery])
        msg.submit(chat_submit, inputs=[msg, chatbot],
                   outputs=[msg, chatbot, gallery])
        clear_btn.click(lambda: ([], []), outputs=[chatbot, gallery])
        ingest_btn.click(ingest_file, inputs=upload, outputs=ingest_out)
    return demo


if __name__ == "__main__":
    ensure_backend()
    build_demo().launch(server_name="0.0.0.0", server_port=7860,
                        allowed_paths=[DATA_DIR])

