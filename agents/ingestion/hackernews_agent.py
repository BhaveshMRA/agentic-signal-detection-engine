import requests

HN_BASE = "https://hn.algolia.com/api/v1"

def scrape_hackernews(keyword: str, limit: int = 30) -> list[dict]:
    # Use 'search' instead of 'search_by_date' for relevance ranking
    url = f"{HN_BASE}/search?query={keyword}&tags=story&hitsPerPage={limit}"
    
    response = requests.get(url)
    response.raise_for_status()
    hits = response.json().get("hits", [])

    posts = []
    for h in hits:
        title = h.get("title", "")
        text  = h.get("story_text") or ""
        if not title:
            continue
        posts.append({
            "id":          h.get("objectID", ""),
            "title":       title,
            "text":        text,
            "url":         h.get("url", ""),
            "score":       h.get("points", 0),
            "created_utc": h.get("created_at_i", 0),
            "source":      "hackernews",
            "keyword":     keyword
        })
    return posts


if __name__ == "__main__":
    from config import TRACKED_KEYWORDS
    for kw in TRACKED_KEYWORDS[:3]:
        posts = scrape_hackernews(kw, limit=5)
        print(f"\n--- {kw.upper()} ({len(posts)} posts) ---")
        for p in posts:
            print(f"  [{p['score']}pts] {p['title'][:80]}")
