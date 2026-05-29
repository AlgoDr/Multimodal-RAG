"""
loader.py — Extract content from files into LlamaIndex-ready objects.

Uses google-genai (new SDK) instead of deprecated google.generativeai.
"""
import os
import fitz
from google import genai
from google.genai import types
from llama_index.core import Document
from llama_index.core.schema import ImageNode


def _caption_image(image_path, client):
    """Call Gemini SDK directly — for textual query on image (not visual query)."""
    
    # read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # infer MIME type from extension
    ext = image_path.lower().split('.')[-1]
    mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Describe this document image for search indexing. Include: "
            "all visible text verbatim, document type, issuing organization, "
            "logos, names of people, dates, and visual layout."
        ]
    )
    return response.text


def load_documents(data_dir, mm_llm=None):
    """Walk data_dir, route each file to PDF or image extractor."""
    
    # initialize Gemini client (reads GOOGLE_API_KEY from env automatically)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    textual_data = []
    vision_data = []
    
    for file in os.listdir(data_dir):
        if file.startswith('.'):
            continue
        
        filepath = os.path.join(data_dir, file)
        
        # ─── PDF: extract text, OCR fallback for scanned PDFs ───
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
        
        # ─── Image: caption + ImageNode (dual path) ───
        elif file.endswith((".png", ".jpg", ".jpeg")):
            print(f"\n---- Image: {file} ----")
            
            # Path A: Gemini Vision caption via new google-genai SDK
            try:
                caption = _caption_image(filepath, client)
                textual_data.append(Document(
                    text=caption,
                    metadata={"file_name": file, "source_type": "image_caption"}
                ))
                print(f"  ✅ Caption: {len(caption)} chars")
            except Exception as e:
                print(f"  ⚠️  Caption failed: {e}")
            
            # Path B: raw image → CLIP visual embedding
            vision_data.append(ImageNode(
                image_path=filepath,
                metadata={"file_name": file, "source_type": "image"}
            ))
            print(f"  ✅ ImageNode for CLIP")
    
    print(f"\n══ Summary: {len(textual_data)} text docs, {len(vision_data)} image nodes ══")
    return textual_data, vision_data