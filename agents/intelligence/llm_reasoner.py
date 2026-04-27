import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=os.getenv("OLLAMA_API_KEY")
        )
    return _client

PROMPT_TEMPLATE = """You are a financial signal analyst monitoring prediction markets.
Given a recent social media post, similar historical posts, and a market snapshot,
determine if the current post shows early signs of a narrative shift that could
precede a prediction market price movement.

Current post:
{current_post}

Similar historical posts:
{historical_context}

Market snapshot:
{market_info}

Respond in exactly this format:
VERDICT: SIGNAL or NO_SIGNAL
REASON: one sentence explanation
CONFIDENCE: a number between 0.0 and 1.0"""

def reason_over_context(current_post: str, historical_context: str, market_info: str) -> dict:
    client = get_client()
    prompt = PROMPT_TEMPLATE.format(
        current_post=current_post,
        historical_context=historical_context,
        market_info=market_info
    )
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL", "gemma4:27b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1000
    )
    raw = response.choices[0].message.content.strip()
    is_signal  = "VERDICT: SIGNAL" in raw and "NO_SIGNAL" not in raw
    confidence = 0.5
    reason     = raw
    for line in raw.split("\n"):
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":")[1].strip())
            except:
                pass
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return {
        "is_signal":  is_signal,
        "confidence": confidence,
        "reasoning":  reason,
        "raw":        raw
    }

if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.processing.preprocessor import preprocess_batch
    from agents.processing.embedder import embed_texts
    from agents.processing.vector_store import upsert_posts, query_similar

    raw   = scrape_hackernews("bitcoin", limit=20)
    posts = preprocess_batch(raw)
    texts = [p["clean_text"] for p in posts]
    embs  = embed_texts(texts)
    upsert_posts(posts, embs)

    latest       = posts[-1]
    latest_emb   = embs[-1]
    similar_docs = query_similar(latest_emb, top_k=3)
    context      = "\n---\n".join(similar_docs)
    market_info  = "Bitcoin market: YES 67% on Polymarket"

    print(f"\nTesting on: {latest['title']}")
    print("-" * 50)

    result = reason_over_context(latest["clean_text"], context, market_info)
    print(f"Signal    : {result['is_signal']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason    : {result['reasoning']}")
    print(f"\nFull output:\n{result['raw']}")
