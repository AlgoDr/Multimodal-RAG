"""
engine.py — V2: ChromaDB for both text and image vector stores.

Architecture:
  Text vectors  → ChromaDB 'rag_text_store'   (BGE 384-dim, PDFs + captions)
  Image vectors → ChromaDB 'rag_image_store'  (CLIP 512-dim, raw images)

  Two collections, one database, both persist automatically to chroma_db/.
  No manual .persist() call for vectors. No re-embedding on restart.

Why ChromaDB for both (vs V1 where images stayed in SimpleVectorStore):
  - Single storage layer — one place to inspect, backup, wipe
  - Proper persistence — CLIP vectors survive kernel restarts
  - Metadata filtering — can filter by source_type before vector search
  - Same VectorStoreQuery interface — our proven V1 bypass still works unchanged

Upstream bugs carried forward from V1 (still present in LlamaIndex):
  1. from_documents() strips ImageDocuments — fix: insert_nodes() separately
  2. MultiModalVectorIndexRetriever.retrieve() returns 0 — fix: direct store query
"""
import os
import chromadb
from llama_index.core import Settings, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.clip import ClipEmbedding


VISUAL_KEYWORDS = {
    "show me", "display", "looks like", "similar to", "visually",
    "image of", "picture of", "appearance", "what does this look",
    "find image", "find images", "view"
}


# ChromaDB persists here on disk
CHROMA_PATH = os.environ.get("CHROMA_PATH_OVERRIDE", "./chroma_db")

TEXT_COLLECTION  = "rag_text_store"        # BGE 384-dim — PDFs + image captions
IMAGE_COLLECTION = "rag_image_store"       # CLIP 512-dim — raw images


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _get_chroma_stores(path=CHROMA_PATH):
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=abs_path)

    text_col  = client.get_or_create_collection(
        TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"}   # ← BGE needs cosine
    )
    image_col = client.get_or_create_collection(
        IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"}   # ← CLIP needs cosine
    )

    return (
        ChromaVectorStore(chroma_collection=text_col),
        ChromaVectorStore(chroma_collection=image_col),
    )

def _is_visual_query(query):
    """True if query asks for visual matching rather than text content."""
    return any(kw in query.lower() for kw in VISUAL_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def build_index(textual_data, vision_data, index_dest_path):
    """
    Build multimodal index with ChromaDB for both vector stores.

    Step 1: from_documents() — embeds PDFs + image captions (textual_data)
            → BGE 384-dim vectors → ChromaDB 'rag_text_store'

    Step 2: insert_nodes() — embeds raw images (vision_data)
            → CLIP 512-dim vectors → ChromaDB 'rag_image_store'
            (insert_nodes bypasses run_transformations() which strips ImageDocuments)

    Step 3: persist() — saves docstore + index metadata to index_dest_path/
            (ChromaDB vectors are already persisted automatically)

    Args:
        textual_data    : list of Document objects from loader.load_documents()
        vision_data     : list of ImageNode objects from loader.load_documents()
        index_dest_path : directory to persist docstore + index metadata
    """
    text_store, image_store = _get_chroma_stores()

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,       # 'default' namespace → ChromaDB text
        image_store=image_store,       # 'image' namespace   → ChromaDB image
    )

    # Step 1 — index PDFs + captions (BGE → text ChromaDB collection)
    index = MultiModalVectorStoreIndex.from_documents(
        textual_data,
        storage_context=storage_context,
        image_embed_model=ClipEmbedding(),
        show_progress=True,
    )

    # Step 2 — index raw images (CLIP → image ChromaDB collection)
    if vision_data:
        index.insert_nodes(vision_data)

    # Step 3 — persist docstore + index metadata (not vectors — chroma has those)
    index.storage_context.persist(persist_dir=index_dest_path)

    print(f"\n✅ Index built")
    print(f"   ChromaDB path   : {CHROMA_PATH}/")
    print(f"   Text vectors    : {text_store._collection.count()} (BGE 384-dim)")
    print(f"   Image vectors   : {image_store._collection.count()} (CLIP 512-dim)")
    print(f"   Docstore        : {index_dest_path}/")
    return index


def load_index(index_dest_path):
    """
    Load previously built index from disk.

    ChromaDB already has all vectors — just reconnects to existing collections.
    Docstore + index metadata loaded from index_dest_path/.
    No re-embedding. Instant startup.
    """
    text_store, image_store = _get_chroma_stores()

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store,
        persist_dir=index_dest_path,
    )

    index = load_index_from_storage(storage_context)

    print(f"✅ Index loaded")
    print(f"   Text vectors  : {text_store._collection.count()}")
    print(f"   Image vectors : {image_store._collection.count()}")
    return index



def see_chunks(index):
    """
    Inspect both ChromaDB collections.

    With an external vector store, content lives in ChromaDB (not the docstore),
    so we read everything from the ChromaDB collections directly.
    """
    import json
    from IPython.display import Image, display

    text_store  = index.storage_context.vector_stores['default']
    image_store = index.storage_context.vector_stores['image']

    # ── Text nodes — read from ChromaDB ───────────────────────────────
    print("=" * 60)
    print(f"TEXT NODES — ChromaDB '{TEXT_COLLECTION}'")
    print(f"Vectors in collection: {text_store._collection.count()}")
    print("=" * 60)

    text_data = text_store._collection.get(include=['documents', 'metadatas'])
    for i, (doc, meta) in enumerate(zip(text_data['documents'], text_data['metadatas']), 1):
        meta = meta or {}
        file_name = meta.get('file_name', 'unknown')
        source    = meta.get('source_type', 'unknown')
        preview   = (doc or "")[:200].replace('\n', ' ')
        print(f"\n--- TEXT NODE {i} ---")
        print(f"  File   : {file_name}")
        print(f"  Source : {source}")
        print(f"  Content: {preview}...")

    print(f"\nTotal text nodes: {len(text_data['ids'])}")

    # ── Image nodes — read from ChromaDB ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"IMAGE NODES — ChromaDB '{IMAGE_COLLECTION}'")
    print(f"Vectors in collection: {image_store._collection.count()}")
    print("=" * 60)

    image_data = image_store._collection.get(include=['metadatas'])
    for i, meta in enumerate(image_data['metadatas'], 1):
        meta = meta or {}
        file_name = meta.get('file_name', 'unknown')

        # image_path from serialized _node_content
        img_path = None
        nc = meta.get('_node_content')
        if nc:
            try:
                img_path = json.loads(nc).get('image_path')
            except (json.JSONDecodeError, TypeError):
                pass
        if not img_path and file_name != 'unknown':
            cand = os.path.join("../data", file_name)
            if os.path.exists(cand):
                img_path = cand

        print(f"\n--- IMAGE NODE {i} ---")
        print(f"  File       : {file_name}")
        print(f"  Stored in  : ChromaDB '{IMAGE_COLLECTION}' (CLIP 512-dim)")
        if img_path and os.path.exists(img_path):
            display(Image(filename=img_path, width=300))
        else:
            print(f"  (image not found: {img_path})")

def ask_query(index, query):
    """
    Unified query entry point — auto-routes by intent.

    Visual keywords → ChromaDB image collection → CLIP similarity → inline images
    Otherwise       → ChromaDB text collection  → BGE retrieval  → Gemini answer
    """
    if _is_visual_query(query):
        return _visual_query(index, query)
    else:
        return _text_query(index, query)

def _text_query(index, query, top_k=7):
    """
    Text retrieval from ChromaDB 'rag_text_store'.

    Uses direct VectorStoreQuery on the ChromaDB store — bypasses the broken
    MultiModalVectorIndexRetriever.retrieve() (returns 0 on hybrid builds).

    IMPORTANT — text location with external vector stores:
    When LlamaIndex uses an external vector store (ChromaDB), it stores node
    text INSIDE ChromaDB's 'documents' field, NOT in the separate docstore
    (which stays empty). So we read text from ChromaDB directly. We keep a
    docstore fallback for robustness across LlamaIndex versions.

    Steps:
    1. Embed query with BGE (384-dim, same model used at index time)
    2. VectorStoreQuery directly on ChromaDB 'default' store
    3. Fetch text from ChromaDB documents field (fallback: docstore)
    4. Assemble context + call Gemini 2.5 Flash
    """
    print("🔍 Text retrieval mode (ChromaDB)\n")

    # 1. Embed query
    query_emb = Settings.embed_model.get_query_embedding(query)

    # 2. Direct store query
    text_store  = index.storage_context.vector_stores['default']
    store_query = VectorStoreQuery(
        query_embedding=query_emb,
        similarity_top_k=top_k,
    )
    result = text_store.query(store_query)

    if not result.ids:
        return "No relevant context found in the index."

    # 3. Fetch text from ChromaDB (text lives here with external vector store)
    chroma_data = text_store._collection.get(
        ids=result.ids,
        include=['documents', 'metadatas'],
    )

    docstore      = index.storage_context.docstore
    context_parts = []
    similarities  = result.similarities or [0.0] * len(result.ids)

    for node_id, similarity in zip(result.ids, similarities):
        # primary: read text + metadata from ChromaDB
        content, meta = "", {}
        if node_id in chroma_data['ids']:
            i = chroma_data['ids'].index(node_id)
            content = chroma_data['documents'][i] or ""
            meta    = chroma_data['metadatas'][i] or {}
        else:
            # fallback: try docstore (older LlamaIndex layouts)
            node = docstore.get_node(node_id, raise_error=False)
            if node is not None:
                content = node.get_content()
                meta    = node.metadata

        if not content.strip():
            continue

        file_name = meta.get('file_name', 'unknown')
        source    = meta.get('source_type', 'unknown')
        context_parts.append(
            f"[Source: {file_name} ({source}) | similarity: {similarity:.3f}]\n{content}"
        )

    if not context_parts:
        return "No text context retrieved."

    context = "\n\n---\n\n".join(context_parts)

    # 4. Call Gemini directly
    prompt = (
        "You are answering questions about personal documents "
        "(certificates, degrees, IDs).\n\n"
        f"Context from retrieved documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Instructions: Answer based on the context above. "
        "When the question asks who 'signed' a certificate, also consider "
        "people listed as instructor, professor, authorizer, or signatory — "
        "these are equivalent roles on certificates. "
        "Cite which document (file name) the answer came from. "
        "If the answer is truly not present, say so."
    )

    return str(Settings.llm.complete(prompt))





def _visual_query(index, query, top_k=2):
    """
    CLIP visual similarity from ChromaDB 'rag_image_store'.

    CLIP embeds the text query into the same 512-dim space as stored images.
    Direct VectorStoreQuery on ChromaDB image store.

    Metadata location: with an external vector store, the full ImageNode is
    serialized into ChromaDB's '_node_content' field (a JSON string). We parse
    it to recover image_path. file_name is also available as a flat field.
    """
    import json
    from IPython.display import Image, display

    print("🖼️  Visual similarity mode (ChromaDB + CLIP)\n")

    # CLIP embeds query text into 512-dim image space
    clip_emb  = ClipEmbedding()
    query_emb = clip_emb.get_query_embedding(query)

    # direct query to image ChromaDB collection
    image_store = index.storage_context.vector_stores['image']
    store_query = VectorStoreQuery(
        query_embedding=query_emb,
        similarity_top_k=top_k,
    )
    result = image_store.query(store_query)

    if not result.ids:
        print("No visually similar images found.")
        return []

    # fetch image metadata from ChromaDB (docstore is empty with external store)
    chroma_data = image_store._collection.get(
        ids=result.ids,
        include=['metadatas'],
    )

    similarities = result.similarities or [0.0] * len(result.ids)
    results_list = []

    print(f"Top {len(result.ids)} visually similar images:\n")

    for i, (node_id, similarity) in enumerate(zip(result.ids, similarities), 1):
        meta = {}
        if node_id in chroma_data['ids']:
            idx  = chroma_data['ids'].index(node_id)
            meta = chroma_data['metadatas'][idx] or {}

        file_name = meta.get('file_name', 'unknown')

        # image_path is inside the serialized _node_content JSON string
        img_path = None
        node_content = meta.get('_node_content')
        if node_content:
            try:
                node_json = json.loads(node_content)
                img_path  = node_json.get('image_path')
            except (json.JSONDecodeError, TypeError):
                pass

        # fallback: reconstruct from file_name
        if not img_path and file_name != 'unknown':
            candidate = os.path.join("../data", file_name)
            if os.path.exists(candidate):
                img_path = candidate

        print(f"--- MATCH {i} (CLIP similarity: {similarity:.3f}) ---")
        print(f"  File: {file_name}")
        if img_path and os.path.exists(img_path):
            display(Image(filename=img_path, width=300))
        else:
            print(f"  (image file not found: {img_path})")
        print()
        results_list.append((file_name, similarity))

    return results_list