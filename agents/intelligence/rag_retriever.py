from agents.processing.embedder import embed_texts
from agents.processing.vector_store import query_similar
from config import RAG_TOP_K

def retrieve_context(query_text: str, top_k: int = RAG_TOP_K) -> str:
    embedding    = embed_texts([query_text])[0]
    similar_docs = query_similar(embedding, top_k=top_k)
    context      = "\n---\n".join(similar_docs)
    return context
