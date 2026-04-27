import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#\w+", "", text)             # remove hashtags
    text = re.sub(r"[^a-z0-9\s]", "", text)     # remove special chars
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace
    return text

def preprocess_batch(posts: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []
    for post in posts:
        raw = post.get("title", "") + " " + post.get("text", "")
        text = clean_text(raw)
        # skip empty or duplicate posts
        if not text or text in seen:
            continue
        seen.add(text)
        post["clean_text"] = text
        cleaned.append(post)
    return cleaned


if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews

    raw_posts = scrape_hackernews("bitcoin", limit=5)
    cleaned   = preprocess_batch(raw_posts)

    print(f"\n--- PREPROCESSED ({len(cleaned)} posts) ---")
    for p in cleaned:
        print(f"\n  RAW   : {p['title'][:60]}")
        print(f"  CLEAN : {p['clean_text'][:60]}")
