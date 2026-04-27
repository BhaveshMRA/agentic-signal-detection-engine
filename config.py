import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
CHROMA_PATH = "./data/chroma_db"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Change Detection ---
CHANGE_DETECTION_PENALTY = 3
CHANGE_DETECTION_WINDOW  = 20

# --- RAG ---
RAG_TOP_K = 5

# --- Bayesian ---
BAYESIAN_THRESHOLD = 0.70

# --- Polling ---
POLLING_INTERVAL_SECONDS = 300

# --- LLM (Ollama Cloud) ---
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

# --- Fallback keywords ---
_DEFAULT_KEYWORDS = ["bitcoin", "election", "fed rate", "trump", "crypto"]

# --- Tag slugs we care about from Polymarket ---
# These come directly from Polymarket's own tag system
_RELEVANT_TAG_SLUGS = {
    "crypto", "finance", "economy", "business", "politics",
    "stocks", "ipos", "geopolitics", "world", "tech",
    "science", "climate", "fed", "rates", "war", "election"
}

def get_dynamic_keywords(limit: int = 5) -> list[str]:
    """
    Fetches Polymarket EVENTS (not markets) — which have proper tags.
    Filters to financially/politically relevant events only using tag slugs.
    Uses event title directly as keyword — no hardcoding, no NLP.
    Sorted by volume — most actively traded first.
    """
    try:
        import requests

        url = "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false"
        response = requests.get(url)
        response.raise_for_status()
        events = response.json()

        # sort by volume descending
        events.sort(key=lambda x: float(x.get("volume", 0) or 0), reverse=True)

        keywords  = []
        used_titles = set()

        for event in events:
            # extract tag slugs for this event
            tags     = event.get("tags", [])
            tag_slugs = {t.get("slug", "").lower() for t in tags}

            # only keep events with at least one relevant tag
            if not tag_slugs.intersection(_RELEVANT_TAG_SLUGS):
                continue

            title = event.get("title", "").strip()
            if not title or title in used_titles:
                continue

            # clean up the title — remove trailing punctuation
            title = title.rstrip("?.,!")
            used_titles.add(title)
            keywords.append({
                "keyword": title,
                "tags":    [t.get("label") for t in tags],
                "volume":  float(event.get("volume", 0) or 0)
            })

            if len(keywords) >= limit:
                break

        if keywords:
            print(f"\n  🎯 Trending Polymarket topics:")
            for k in keywords:
                print(f"     → [{k['tags'][0] if k['tags'] else 'N/A'}] "
                      f"{k['keyword']} (vol: ${k['volume']:,.0f})")
            return [k["keyword"] for k in keywords]

        return _DEFAULT_KEYWORDS

    except Exception as e:
        print(f"  ⚠️  Falling back to defaults: {e}")
        return _DEFAULT_KEYWORDS


TRACKED_KEYWORDS = _DEFAULT_KEYWORDS