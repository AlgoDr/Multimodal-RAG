"""
engine.py — Build multimodal index, retrieve, query (unified router).

Architecture:
- Text retrieval: direct VectorStoreQuery on 'default' store → docstore lookup
  → manual context assembly → Settings.llm.complete()
  
  Why direct store query: MultiModalVectorIndexRetriever.retrieve() returns 0
  nodes on hybrid builds (from_documents + insert_nodes) despite embeddings
  existing in the underlying SimpleVectorStore. Verified empirically.

- Visual retrieval: direct VectorStoreQuery on 'image' store using CLIP-embedded
  query text. Same workaround — retriever wrapper has the same bug.

This pattern (bypass broken wrapper, call underlying API directly) is the
standard production workaround when high-level abstractions have dispatch bugs
you cannot patch upstream.
"""
import os
from llama_index.core import Settings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.clip import ClipEmbedding


# query intent detection — visual keywords route to CLIP image search
VISUAL_KEYWORDS = {
    "show me", "display", "looks like", "similar to", "visually",
    "image of", "picture of", "appearance", "what does this look",
    "find image", "find images", "view"
}


def _is_visual_query(query):
    """True if query asks for visual matching/display rather than text content."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in VISUAL_KEYWORDS)


def build_index(textual_data, vision_data, index_dest_path):
    """Build multimodal index from extracted documents and image nodes.
    
    Two-step build:
    1. from_documents() with explicit CLIP — text gets BGE, images get CLIP
    2. insert_nodes() for ImageNodes — bypasses run_transformations() which
       strips ImageDocuments (LlamaIndex quirk discovered during dev)
    """
    index = MultiModalVectorStoreIndex.from_documents(
        textual_data,
        image_embed_model=ClipEmbedding()
    )
    
    if vision_data:
        index.insert_nodes(vision_data)
    
    index.storage_context.persist(persist_dir=index_dest_path)
    print(f"\nIndex built and saved to '{index_dest_path}/'")
    return index


def load_index(index_dest_path):
    """Load previously-built index from disk into memory."""
    storage_context = StorageContext.from_defaults(persist_dir=index_dest_path)
    index = load_index_from_storage(storage_context)
    print(f"Index loaded from '{index_dest_path}/'")
    return index


def see_chunks(index):
    """Inspect both vector stores — text nodes (PDFs + captions) and image nodes (CLIP)."""
    from IPython.display import Image, display
    
    print("=" * 60)
    print("TEXT NODES (PDFs + image captions via BGE)")
    print("=" * 60)
    
    docstore = index.storage_context.docstore
    ref_info = docstore.get_all_ref_doc_info() or {}
    
    text_count = 0
    for doc_id, info in ref_info.items():
        node_ids = info.node_ids if hasattr(info, 'node_ids') else []
        meta = info.metadata if hasattr(info, 'metadata') else {}
        file_name = meta.get('file_name', 'unknown')
        source = meta.get('source_type', 'unknown')
        
        for node_id in node_ids:
            node = docstore.get_node(node_id)
            text_count += 1
            print(f"\n--- TEXT NODE {text_count} ---")
            print(f"  File   : {file_name}")
            print(f"  Source : {source}")
            print(f"  Content: {node.get_content()[:250]}...")
    
    print(f"\nTotal text nodes: {text_count}\n")
    
    print("=" * 60)
    print("IMAGE NODES (raw images via CLIP)")
    print("=" * 60)
    
    image_store = index.storage_context.vector_stores.get('image')
    if image_store is None or not hasattr(image_store, '_data'):
        print("⚠️  No image vector store found")
        return
    
    image_ids = list(image_store._data.embedding_dict.keys())
    print(f"\nFound {len(image_ids)} image nodes\n")
    
    for i, node_id in enumerate(image_ids):
        node = docstore.get_node(node_id)
        meta = node.metadata if hasattr(node, 'metadata') else {}
        emb = image_store._data.embedding_dict[node_id]
        
        print(f"--- IMAGE NODE {i+1} ---")
        print(f"  File   : {meta.get('file_name', 'embedded')}")
        print(f"  CLIP vector: {len(emb)} dims (preview: {[round(x,3) for x in emb[:5]]}...)")
        
        img_path = getattr(node, 'image_path', None)
        if img_path and os.path.exists(img_path):
            display(Image(filename=img_path, width=300))
        print()


def ask_query(index, query):
    """
    Unified query entry point — auto-routes by intent.
    
    - Visual keywords ("show me", "looks like", "similar to") → CLIP visual search
    - Otherwise → text retrieval across PDFs + image captions → Gemini answer
    """
    if _is_visual_query(query):
        return _visual_query(index, query)
    else:
        return _text_query(index, query)


def _text_query(index, query, top_k=5):
    """Text retrieval via direct vector store query (bypasses broken retriever).
    
    Steps:
    1. Embed query with BGE
    2. Query 'default' (text) vector store directly → get node IDs + similarities
    3. Fetch full nodes from docstore by ID
    4. Build context + call Settings.llm.complete() directly
    """
    print(f"🔍 Text retrieval mode\n")
    
    # ── 1. Embed query with BGE ──
    query_emb = Settings.embed_model.get_query_embedding(query)
    
    # ── 2. Direct query to text vector store (bypasses broken retriever wrapper) ──
    text_store = index.storage_context.vector_stores['default']
    store_query = VectorStoreQuery(query_embedding=query_emb, similarity_top_k=top_k)
    result = text_store.query(store_query)
    
    if not result.ids:
        return "No relevant context found in the index."
    
    # ── 3. Fetch nodes from docstore by ID ──
    docstore = index.storage_context.docstore
    context_parts = []
    
    similarities = result.similarities or [0.0] * len(result.ids)
    for node_id, similarity in zip(result.ids, similarities):
        node = docstore.get_node(node_id)
        content = node.get_content()
        if not content.strip():
            continue
        file_name = node.metadata.get('file_name', 'unknown')
        source = node.metadata.get('source_type', 'unknown')
        context_parts.append(
            f"[Source: {file_name} ({source}) | similarity: {similarity:.3f}]\n{content}"
        )
    
    if not context_parts:
        return "No text context retrieved."
    
    context = "\n\n---\n\n".join(context_parts)
    
    # ── 4. Build prompt + call LLM directly ──
    prompt = (
        f"You are answering questions about personal documents (certificates, degrees, IDs).\n\n"
        f"Context from retrieved documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based ONLY on the context above. If the answer is not in the context, "
        f"say so honestly. Cite which document (file name) the answer came from."
    )
    
    response = Settings.llm.complete(prompt)
    return str(response)


def _visual_query(index, query, top_k=2):
    """CLIP visual similarity via direct image store query.
    
    Same workaround as _text_query — retriever wrapper returns 0 on hybrid
    indexes, so we query the image store directly with CLIP-embedded query.
    """
    from IPython.display import Image, display
    
    print(f"🖼️  Visual similarity mode (CLIP)\n")
    
    # CLIP embeds the text query into the same 512-dim space as stored images
    clip_emb = ClipEmbedding()
    query_emb = clip_emb.get_query_embedding(query)
    
    # Direct query to image vector store
    image_store = index.storage_context.vector_stores['image']
    store_query = VectorStoreQuery(query_embedding=query_emb, similarity_top_k=top_k)
    result = image_store.query(store_query)
    
    if not result.ids:
        print("No visually similar images found.")
        return []
    
    docstore = index.storage_context.docstore
    similarities = result.similarities or [0.0] * len(result.ids)
    
    print(f"Top {len(result.ids)} visually similar images:\n")
    results_list = []
    
    for i, (node_id, similarity) in enumerate(zip(result.ids, similarities), 1):
        node = docstore.get_node(node_id)
        file_name = node.metadata.get('file_name', 'unknown')
        img_path = getattr(node, 'image_path', None)
        
        print(f"--- MATCH {i} (CLIP similarity: {similarity:.3f}) ---")
        print(f"  File: {file_name}")
        if img_path and os.path.exists(img_path):
            display(Image(filename=img_path, width=300))
        print()
        results_list.append((node, similarity))
    
    return results_list