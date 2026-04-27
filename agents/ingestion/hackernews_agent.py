import requests

HN_BASE = "https://hn.algolia.com/api/v1"

def _extract_search_term(keyword: str) -> str:
    """
    Extracts a short, effective search term from a long keyword.
    'Republican Presidential Nominee 2028' -> 'Republican Presidential'
    'Netanyahu out by' -> 'Netanyahu'
    'MicroStrategy sells any Bitcoin' -> 'MicroStrategy Bitcoin'
    """
    SKIP = {
        "will", "the", "a", "an", "in", "on", "at", "to", "for",
        "of", "and", "or", "is", "be", "before", "after", "by",
        "any", "out", "next", "new", "first", "last", "winner",
        "nominee", "election", "presidential", "prime", "minister",
        "sells", "invade", "invades", "called", "end", "win", "wins"
    }
    words = [w for w in keyword.lower().split() if w not in SKIP and len(w) > 2]
    return " ".join(words[:2]) if words else keyword.split()[0]


def scrape_hackernews(keyword: str, limit: int = 30) -> list[dict]:
    """
    Fetches HackerNews stories matching a keyword.
    Automatically shortens long keywords for better search results.
    """
    search_term = _extract_search_term(keyword)
    url = f"{HN_BASE}/search?query={search_term}&tags=story&hitsPerPage={limit}"

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
            "keyword":     keyword,
            "search_term": search_term
        })
    return posts


if __name__ == "__main__":
    test_keywords = [
        "Republican Presidential Nominee 2028",
        "Netanyahu out by",
        "MicroStrategy sells any Bitcoin",
        "Democratic Presidential Nominee 2028"
    ]
    for kw in test_keywords:
        posts = scrape_hackernews(kw, limit=5)
        term  = _extract_search_term(kw)
        print(f"\n--- {kw[:50]}")
        print(f"    Search term used: '{term}'")
        print(f"    Posts found: {len(posts)}")
        for p in posts[:3]:
            print(f"    [{p['score']}pts] {p['title'][:70]}")
