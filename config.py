import os
import re
import json
import requests
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
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

OLLAMA_API_KEY  = _get_secret("OLLAMA_API_KEY")
OLLAMA_BASE_URL = _get_secret("OLLAMA_BASE_URL") or "https://ollama.com/v1"
OLLAMA_MODEL    = _get_secret("OLLAMA_MODEL")    or "gemma4:31b-cloud"

TWITTER_CONSUMER_KEY    = _get_secret("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = _get_secret("TWITTER_CONSUMER_SECRET")


# --- Fallback keywords ---
_DEFAULT_KEYWORDS = ["bitcoin", "election", "fed rate", "trump", "crypto"]

# --- Relevant tag slugs from Polymarket ---
_RELEVANT_TAG_SLUGS = {
    "crypto", "finance", "economy", "business", "politics",
    "stocks", "ipos", "geopolitics", "world", "tech",
    "science", "climate", "fed", "rates", "war", "election",
    "world-elections", "middle-east", "asia", "europe"
}

def get_dynamic_keywords(limit: int = 8) -> list[str]:
    try:
        url = "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false"
        response = requests.get(url)
        response.raise_for_status()
        events = response.json()
        events.sort(key=lambda x: float(x.get("volume", 0) or 0), reverse=True)

        keywords    = []
        used_titles = set()

        for event in events:
            tags      = event.get("tags", [])
            tag_slugs = {t.get("slug", "").lower() for t in tags}
            if not tag_slugs.intersection(_RELEVANT_TAG_SLUGS):
                continue
            title = event.get("title", "").strip().rstrip("?.,!")
            if not title or title in used_titles:
                continue
            used_titles.add(title)
            keywords.append(title)
            if len(keywords) >= limit:
                break

        final = keywords if keywords else _DEFAULT_KEYWORDS
        print(f"  Trending Polymarket topics: {final}")
        return final

    except Exception as e:
        print(f"  Falling back to defaults: {e}")
        return _DEFAULT_KEYWORDS


def get_keyword_market_map(limit: int = 8) -> dict:
    """
    Returns dict mapping keyword to its Polymarket market data.
    Keyword and market are always in sync.
    """
    try:
        url = "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false"
        response = requests.get(url)
        response.raise_for_status()
        events = response.json()
        events.sort(key=lambda x: float(x.get("volume", 0) or 0), reverse=True)

        result      = {}
        used_titles = set()

        for event in events:
            tags      = event.get("tags", [])
            tag_slugs = {t.get("slug", "").lower() for t in tags}
            if not tag_slugs.intersection(_RELEVANT_TAG_SLUGS):
                continue

            title = event.get("title", "").strip().rstrip("?.,!")
            if not title or title in used_titles:
                continue
            used_titles.add(title)

            event_markets = event.get("markets", [])
            yes_price     = None
            question      = title

            if event_markets:
                try:
                    first  = event_markets[0]
                    prices = first.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    yes_price = float(prices[0]) if prices else None
                    question  = first.get("question", title)
                except Exception:
                    pass

            result[title] = {
                "question":  question,
                "yes_price": yes_price,
                "volume":    float(event.get("volume", 0) or 0),
                "tags":      [t.get("label") for t in tags]
            }

            if len(result) >= limit:
                break

        return result

    except Exception as e:
        print(f"  Could not build keyword-market map: {e}")
        return {}


# default fallback
TRACKED_KEYWORDS = _DEFAULT_KEYWORDS
