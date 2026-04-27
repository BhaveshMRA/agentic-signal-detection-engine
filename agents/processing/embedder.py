from sentence_transformers import SentenceTransformer

# Load once at module level so it doesn't reload every call
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model (first time only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.processing.preprocessor import preprocess_batch

    raw   = scrape_hackernews("bitcoin", limit=5)
    posts = preprocess_batch(raw)
    texts = [p["clean_text"] for p in posts]

    embeddings = embed_texts(texts)

    print(f"\n--- EMBEDDINGS ---")
    print(f"  Posts embedded : {len(embeddings)}")
    print(f"  Vector size    : {len(embeddings[0])} dimensions")
    print(f"  First 5 values : {[round(v, 4) for v in embeddings[0][:5]]}")
