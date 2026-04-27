import time
from datetime import datetime, timezone
from agents.ingestion.polymarket_agent import get_active_markets

# in-memory store of signals and market snapshots
# structure: list of dicts
_signal_log    = []   # signals we fired + timestamp
_market_log    = []   # market snapshots over time
_correlations  = []   # confirmed matches

def snapshot_markets():
    """
    Take a fresh snapshot of all active markets right now.
    Call this BEFORE running the pipeline so we have a baseline.
    """
    markets   = get_active_markets(limit=20)
    timestamp = time.time()
    entry = {
        "timestamp": timestamp,
        "datetime":  datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
        "markets":   {
            m["question"]: {
                "yes_price": m["yes_price"],
                "volume":    m["volume"]
            }
            for m in markets
        }
    }
    _market_log.append(entry)
    print(f"  📸 Market snapshot taken at {entry['datetime']} — {len(entry['markets'])} markets")
    return entry


def log_signal(keyword: str, verdict: str, confidence: float, reasoning: str):
    """
    Record a signal the pipeline fired, with timestamp.
    Call this when LLM says SIGNAL.
    """
    timestamp = time.time()
    entry = {
        "timestamp":  timestamp,
        "datetime":   datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
        "keyword":    keyword,
        "verdict":    verdict,
        "confidence": confidence,
        "reasoning":  reasoning,
        "validated":  None   # None = not yet checked
    }
    _signal_log.append(entry)
    print(f"  📝 Signal logged: {keyword.upper()} at {entry['datetime']}")
    return entry


def compare_snapshots(before: dict, after: dict, threshold: float = 0.03) -> list[dict]:
    """
    Compare two market snapshots.
    Returns list of markets where YES price moved by more than threshold.
    threshold=0.03 means 3 percentage points movement.
    """
    moves = []
    for question, before_data in before["markets"].items():
        if question not in after["markets"]:
            continue
        after_data  = after["markets"][question]
        before_price = before_data.get("yes_price") or 0
        after_price  = after_data.get("yes_price") or 0

        if before_price and after_price:
            change = after_price - before_price
            if abs(change) >= threshold:
                moves.append({
                    "question":     question,
                    "before":       round(before_price * 100, 1),
                    "after":        round(after_price  * 100, 1),
                    "change":       round(change * 100, 1),
                    "direction":    "UP ↑" if change > 0 else "DOWN ↓"
                })
    return sorted(moves, key=lambda x: abs(x["change"]), reverse=True)


def validate_signals(lag_seconds: int = 300) -> list[dict]:
    """
    For each unvalidated signal, check if any market moved
    within lag_seconds after the signal was fired.
    Default lag = 5 minutes.
    """
    if len(_market_log) < 2:
        print("  ⚠️  Need at least 2 market snapshots to validate signals")
        return []

    results = []
    for signal in _signal_log:
        if signal["validated"] is not None:
            continue   # already checked

        signal_time = signal["timestamp"]

        # find market snapshots bracketing this signal
        before_snaps = [s for s in _market_log if s["timestamp"] <= signal_time]
        after_snaps  = [s for s in _market_log
                        if s["timestamp"] > signal_time
                        and s["timestamp"] <= signal_time + lag_seconds]

        if not before_snaps or not after_snaps:
            continue   # not enough data yet

        before = before_snaps[-1]   # most recent snapshot before signal
        after  = after_snaps[-1]    # most recent snapshot after signal

        moves = compare_snapshots(before, after)

        if moves:
            signal["validated"] = True
            result = {
                "signal_time":  signal["datetime"],
                "keyword":      signal["keyword"],
                "market_moves": moves,
                "confirmed":    True,
                "lag_seconds":  round(after["timestamp"] - signal["timestamp"])
            }
            _correlations.append(result)
            results.append(result)
            print(f"  ✅ CONFIRMED: {signal['keyword'].upper()} signal preceded market move!")
            for m in moves:
                print(f"     → {m['question'][:60]} {m['direction']} {m['change']:+.1f}%")
        else:
            signal["validated"] = False
            print(f"  ❌ No market move detected after {signal['keyword'].upper()} signal")

    return results


def get_summary() -> dict:
    """
    Returns a summary of signal accuracy so far.
    """
    total     = len([s for s in _signal_log if s["validated"] is not None])
    confirmed = len([s for s in _signal_log if s["validated"] is True])
    accuracy  = (confirmed / total * 100) if total > 0 else 0

    return {
        "total_signals":     len(_signal_log),
        "validated":         total,
        "confirmed_correct": confirmed,
        "accuracy_pct":      round(accuracy, 1),
        "market_snapshots":  len(_market_log),
        "correlations":      _correlations
    }


if __name__ == "__main__":
    import time

    print("\n🔗 CORRELATOR TEST — Simulating signal + market validation")
    print("=" * 55)

    # Step 1 — take baseline snapshot
    print("\n[1] Taking baseline market snapshot...")
    snap1 = snapshot_markets()

    # Step 2 — simulate a signal firing
    print("\n[2] Simulating a SIGNAL detection...")
    log_signal(
        keyword    = "bitcoin",
        verdict    = "SIGNAL",
        confidence = 0.82,
        reasoning  = "Unusual narrative shift detected around Bitcoin regulation"
    )

    # Step 3 — wait and take another snapshot
    print("\n[3] Waiting 5 seconds and taking follow-up snapshot...")
    time.sleep(5)
    snap2 = snapshot_markets()

    # Step 4 — compare snapshots directly
    print("\n[4] Comparing market snapshots...")
    moves = compare_snapshots(snap1, snap2, threshold=0.001)  # low threshold for test
    if moves:
        print(f"  Found {len(moves)} market(s) that moved:")
        for m in moves:
            print(f"  → {m['question'][:65]}")
            print(f"    Before: {m['before']}% → After: {m['after']}% ({m['direction']} {m['change']:+.1f}%)")
    else:
        print("  No significant market moves detected (normal over 5 seconds)")

    # Step 5 — validate signals
    print("\n[5] Validating signals against market moves...")
    validate_signals(lag_seconds=30)

    # Step 6 — summary
    print("\n[6] Summary:")
    s = get_summary()
    print(f"  Total signals logged    : {s['total_signals']}")
    print(f"  Market snapshots taken  : {s['market_snapshots']}")
    print(f"  Signals validated       : {s['validated']}")
    print(f"  Confirmed correct       : {s['confirmed_correct']}")
    print(f"  Accuracy so far         : {s['accuracy_pct']}%")
