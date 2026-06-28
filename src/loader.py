"""
loader.py — Extract content from files into LlamaIndex-ready objects.

Uses google-genai (new SDK) instead of deprecated google.generativeai.
"""
import os
import time
import fitz
from google import genai
from google.genai import types
from llama_index.core import Document
from llama_index.core.schema import ImageNode


def _caption_image(image_path, client, max_retries=4):
    """Call Gemini SDK with retry on transient 503 errors.

    Gemini returns 503 UNAVAILABLE during high demand. These are
    temporary — we retry with exponential backoff (2s, 4s, 8s, 16s)
    before giving up. This is the standard production pattern for
    handling transient API failures.

    Raises the last exception if all retries exhausted — caller decides
    whether to skip (build_index) or propagate (ingest quota detection).
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = image_path.lower().split('.')[-1]
    mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    "Describe this document image for search indexing. Include: "
                    "all visible text verbatim, document type, issuing organization, "
                    "logos, names of people, dates, and visual layout."
                ]
            )
            return response.text   # success — return immediately
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s, 16s
                print(f"    503 — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise   # non-503 error — don't retry, propagate immediately

    # all retries exhausted
    raise last_error


def load_documents(data_dir, mm_llm=None, single_file=None):
    """Walk data_dir, route each file to PDF or image extractor.

    Args:
        data_dir    : directory containing documents
        mm_llm      : unused (kept for API compatibility with older callers)
        single_file : if provided, process ONLY this filename.
                      Used by FastAPI /ingest for incremental indexing.
                      If None, process all files in data_dir (default).

    Key behaviour difference in single_file mode:
        Image caption failures RE-RAISE instead of being caught — so the
        /ingest endpoint can detect quota errors and return a clear message
        to the user rather than silently skipping the caption.
    """
    # initialize Gemini client (reads GOOGLE_API_KEY from env automatically)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    textual_data = []
    vision_data  = []

    # ── file selection ──────────────────────────────────────────────────
    # single_file mode : process one file only  (for /ingest endpoint)
    # full mode        : process everything     (for build_index)
    files_to_process = [single_file] if single_file else os.listdir(data_dir)

    for file in files_to_process:
        if file.startswith('.'):
            continue

        filepath = os.path.join(data_dir, file)

        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            continue

        # ─── PDF: extract text, OCR fallback for scanned PDFs ───────────
        if file.endswith('.pdf'):
            print(f"\n==== PDF: {file} ====")
            pages = []

            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")

                if len(text.strip()) < 20:
                    tp = page.get_textpage_ocr(language="eng", dpi=300)
                    text = page.get_text("text", textpage=tp)
                    print(f"  [OCR used on page {page_num+1}]")

                pages.append(text)
            doc.close()

            combined = "\n\n".join(pages)
            if combined.strip():
                textual_data.append(Document(
                    text=combined,
                    metadata={"file_name": file, "source_type": "pdf"}
                ))
                print(f"  ✅ {len(combined)} chars")

        # ─── Image: caption (Gemini) + ImageNode (CLIP) — dual path ─────
        elif file.endswith((".png", ".jpg", ".jpeg")):
            print(f"\n---- Image: {file} ----")

            # Path A: Gemini Vision caption → BGE text embedding
            try:
                caption = _caption_image(filepath, client)
                textual_data.append(Document(
                    text=caption,
                    metadata={"file_name": file, "source_type": "image_caption"}
                ))
                print(f"  ✅ Caption: {len(caption)} chars")
            except Exception as e:
                if single_file:
                    # ingest mode: re-raise so server detects quota error
                    # and returns a user-friendly message instead of 500
                    raise
                else:
                    # build mode: log and continue — don't stop whole build
                    # for one failed caption (Gemini 503 is transient)
                    print(f"  ⚠️  Caption failed: {e}")

            # Path B: raw image → CLIP visual embedding (always runs, no Gemini)
            vision_data.append(ImageNode(
                image_path=filepath,
                metadata={"file_name": file, "source_type": "image"}
            ))
            print(f"  ✅ ImageNode for CLIP")

    print(f"\n══ Summary: {len(textual_data)} text docs, {len(vision_data)} image nodes ══")
    return textual_data, vision_data