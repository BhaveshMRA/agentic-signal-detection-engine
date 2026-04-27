import requests
import json

POLYMARKET_BASE = "https://gamma-api.polymarket.com"

def get_active_markets(limit: int = 10, min_volume: float = 1000.0) -> list[dict]:
    url = f"{POLYMARKET_BASE}/markets?limit=50&active=true&closed=false"
    response = requests.get(url)
    response.raise_for_status()
    markets = response.json()

    results = []
    for m in markets:
        try:
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            volume = float(m.get("volume", 0) or 0)
            if volume < min_volume:
                continue
            results.append({
                "id":             m.get("id", ""),
                "question":       m.get("question", ""),
                "group_title":    m.get("groupItemTitle", ""),
                "description":    m.get("description", "")[:200],
                "yes_price":      float(prices[0]) if prices else None,
                "no_price":       float(prices[-1]) if len(prices) > 1 else None,
                "volume":         volume,
                "volume_24hr":    float(m.get("volume24hr", 0) or 0),
                "end_date":       m.get("endDate", ""),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:limit]


if __name__ == "__main__":
    markets = get_active_markets(limit=10)
    print(f"\n--- TOP {len(markets)} MARKETS ---")
    for m in markets:
        yes = f"{float(m['yes_price'])*100:.0f}%" if m['yes_price'] else "N/A"
        print(f"  [YES {yes}] {m['question'][:60]}")
        print(f"           group_title: {m['group_title']}")
        print()
