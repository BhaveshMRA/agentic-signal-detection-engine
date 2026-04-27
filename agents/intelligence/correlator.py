import time
import json
import requests
from datetime import datetime, timezone

POLYMARKET_BASE = "https://gamma-api.polymarket.com"

_signal_log   = []
_market_log   = []
_correlations = []


def snapshot_markets(keywords: list = None) -> dict:
    """
    Take a fresh snapshot of active markets.
    If keywords provided, tries to match markets to keywords.
    """
    try:
        url      = f"{POLYMARKET_BASE}/markets?limit=50&active=true&closed=false"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        markets   = response.json()
        timestamp = time.time()

        snapshot = {
            "timestamp": timestamp,
            "datetime":  datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
            "markets":   {}
        }

        for m in markets:
            try:
                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                yes_price = float(prices[0]) if prices else None
                question  = m.get("question", "")
                snapshot["markets"][question] = {
                    "yes_price": yes_price,
                    "volume":    float(m.get("volume", 0) or 0),
                    "volume24h": float(m.get("volume24hr", 0) or 0)
                }
            except Exception:
                continue

        _market_log.append(snapshot)
        print(f"  📸 Snapshot at {snapshot['datetime']} — {len(snapshot['markets'])} markets")
        return snapshot

    except Exception as e:
        print(f"  ⚠️  Snapshot failed: {e}")
        return {"timestamp": time.time(), "datetime": "unknown", "markets": {}}


def log_signal(keyword: str, verdict: str, confidence: float, reasoning: str) -> dict:
    """
    Record a signal the pipeline fired.
    """
    timestamp = time.time()
    entry = {
        "timestamp":  timestamp,
        "datetime":   datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
        "keyword":    keyword,
        "verdict":    verdict,
        "confidence": confidence,
        "reasoning":  reasoning,
        "validated":  None
    }
    _signal_log.append(entry)
    print(f"  �� Signal logged: {keyword.upper()} at {entry['datetime']}")
    return entry


def compare_snapshots(before: dict, after: dict, threshold: float = 0.02) -> list:
    """
    Compare two market snapshots.
    Returns markets where YES price moved by more than threshold.
    threshold=0.02 means 2 percentage points movement.
    """
    moves = []
    for question, before_data in before.get("markets", {}).items():
        if question not in after.get("markets", {}):
            continue
        after_data    = after["markets"][question]
        before_price  = before_data.get("yes_price") or 0
        after_price   = after_data.get("yes_price") or 0

        if before_price and after_price:
            change = after_price - before_price
            if abs(change) >= threshold:
                moves.append({
                    "question":  question,
                    "before":    round(before_price * 100, 1),
                    "after":     round(after_price  * 100, 1),
                    "change":    round(change * 100, 2),
                    "direction": "UP ↑" if change > 0 else "DOWN ↓"
                })

    return sorted(moves, key=lambda x: abs(x["change"]), reverse=True)


def validate_signal(before_snapshot: dict, after_snapshot: dict, keyword: str) -> dict:
    """
    Check if a market related to keyword moved between two snapshots.
    Returns validation result with confirmed True/False.
    """
    moves = compare_snapshots(before_snapshot, after_snapshot)

    # check if any move is related to our keyword
    kw_lower    = keyword.lower()
    kw_words    = set(kw_lower.split())
    related     = []

    for move in moves:
        q_words = set(move["question"].lower().split())
        overlap = kw_words.intersection(q_words)
        if overlap:
            move["relevance"] = list(overlap)
            related.append(move)

    confirmed = len(related) > 0

    result = {
        "keyword":          keyword,
        "confirmed":        confirmed,
        "related_moves":    related,
        "all_moves":        moves,
        "before_time":      before_snapshot.get("datetime"),
        "after_time":       after_snapshot.get("datetime"),
        "lag_seconds":      round(after_snapshot["timestamp"] - before_snapshot["timestamp"])
    }

    if confirmed:
        print(f"  ✅ CONFIRMED: {keyword.upper()} signal preceded market move!")
        for m in related:
            print(f"     → {m['question'][:60]} {m['direction']} {m['change']:+.1f}%")
    else:
        print(f"  ❌ No related market move detected for {keyword.upper()}")

    return result


def get_summary() -> dict:
    confirmed = [s for s in _signal_log if s["validated"] is True]
    rejected  = [s for s in _signal_log if s["validated"] is False]
    total     = len(confirmed) + len(rejected)
    accuracy  = (len(confirmed) / total * 100) if total > 0 else 0

    return {
        "total_signals":     len(_signal_log),
        "confirmed_correct": len(confirmed),
        "confirmed_wrong":   len(rejected),
        "accuracy_pct":      round(accuracy, 1),
        "market_snapshots":  len(_market_log),
        "correlations":      _correlations
    }
