import numpy as np
import ruptures as rpt

def detect_changepoints(embeddings: list[list[float]], penalty: int = 3) -> list[int]:
    if len(embeddings) < 4:
        return []
    matrix = np.array(embeddings)
    algo = rpt.Pelt(model="rbf").fit(matrix)
    changepoints = algo.predict(pen=penalty)
    return changepoints[:-1]

def compute_drift_score(embeddings: list[list[float]]) -> float:
    if len(embeddings) < 2:
        return 0.0
    mid = len(embeddings) // 2
    first  = np.mean(embeddings[:mid], axis=0)
    second = np.mean(embeddings[mid:], axis=0)
    cosine_sim = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(1 - cosine_sim)

if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.processing.preprocessor import preprocess_batch
    from agents.processing.embedder import embed_texts

    keywords = ["bitcoin", "election", "fed rate"]
    for kw in keywords:
        raw        = scrape_hackernews(kw, limit=20)
        posts      = preprocess_batch(raw)
        texts      = [p["clean_text"] for p in posts]
        embeddings = embed_texts(texts)

        drift      = compute_drift_score(embeddings)
        changes    = detect_changepoints(embeddings)

        print(f"\n--- {kw.upper()} ---")
        print(f"  Posts analysed  : {len(embeddings)}")
        print(f"  Drift score     : {drift:.4f}  {'🚨 HIGH' if drift > 0.1 else '✅ LOW'}")
        print(f"  Changepoints at : {changes if changes else 'none detected'}")
