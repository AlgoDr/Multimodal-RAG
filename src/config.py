import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal



import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="llama_index.llms.gemini")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*google.generativeai.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")


def setup_rag_settings():
    """
    RAG configuration:
    - Local text embeddings (BGE) → no API quota, unlimited indexing
    - Local image embeddings (CLIP, via MultiModalVectorStoreIndex defaults)
    - Cloud LLM (Gemini 2.5 Flash) → only used at query time

    GOOGLE_API_KEY auto-read from environment by Gemini client.

    Two LLM wrappers maintained for future modality routing:
    - Settings.llm  → text queries (PDFs, DBs, transcribed audio)
    - mm_llm        → multimodal queries (images, slides, video frames)
    """

    # text embeddings — local BGE, no quota, runs on M4 GPU via MPS
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # text LLM — Gemini 2.5 Flash, free tier 250 req/day
    Settings.llm = Gemini(
        model="models/gemini-2.5-flash"
    )

    # multimodal LLM — same Gemini 2.5 Flash, image-aware interface
    mm_llm = GeminiMultiModal(
        model="models/gemini-2.5-flash"
    )
    return mm_llm