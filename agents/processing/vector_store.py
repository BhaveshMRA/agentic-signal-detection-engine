import chromadb
from config import CHROMA_PATH

_client = None
_collection = None

def get_collection(name: str = "signals"):
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def upsert_posts(posts: list[dict], embeddings: list[list[float]]):
    collection = get_collection()
    collection.upsert(
        ids=[str(p["id"]) for p in posts],
        embeddings=embeddings,
        documents=[p["clean_text"] for p in posts],
        metadatas=[{
            "source":      p.get("source", ""),
            "keyword":     p.get("keyword", ""),
            "created_utc": str(p.get("created_utc", "")),
            "score":       str(p.get("score", 0)),
            "title":       p.get("title", "")[:100]
        } for p in posts]
    )
    print(f"  Upserted {len(posts)} posts into ChromaDB")

def query_similar(embedding: list[float], top_k: int = 5) -> list[str]:
    collection = get_collection()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]

def get_collection_size() -> int:
    return get_collection().count()

if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.processing.preprocessor import preprocess_batch
    from agents.processing.embedder import embed_texts

    raw        = scrape_hackernews("bitcoin", limit=10)
    posts      = preprocess_batch(raw)
    texts      = [p["clean_text"] for p in posts]
    embeddings = embed_texts(texts)

    upsert_posts(posts, embeddings)
    print(f"  Total in ChromaDB : {get_collection_size()} documents")

    print("\n--- RAG RETRIEVAL TEST ---")
    query_embedding = embed_texts(["federal reserve interest rate decision"])[0]
    similar = query_similar(query_embedding, top_k=3)
    for i, doc in enumerate(similar):
        print(f"  [{i+1}] {doc[:80]}")
