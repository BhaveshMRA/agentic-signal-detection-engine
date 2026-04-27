from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.processing.preprocessor import preprocess_batch
from agents.processing.embedder import embed_texts
from agents.processing.vector_store import upsert_posts
from agents.intelligence.change_detector import compute_drift_score, detect_changepoints
from agents.intelligence.rag_retriever import retrieve_context
from agents.intelligence.llm_reasoner import reason_over_context
from agents.intelligence.bayesian_model import BayesianSignalModel
from agents.intelligence.correlator import (
    snapshot_markets, log_signal, validate_signal
)
from config import BAYESIAN_THRESHOLD

bayes = BayesianSignalModel()

class PipelineState(TypedDict):
    raw_posts:          list
    clean_posts:        list
    embeddings:         list
    drift_score:        float
    changepoints:       list
    rag_context:        str
    llm_output:         dict
    market_snapshot:    dict
    before_snapshot:    dict
    after_snapshot:     dict
    bayes_probability:  float
    alert_fired:        bool
    reasoning:          Optional[str]
    keyword:            str
    correlation_result: Optional[dict]


def preprocess_node(state: PipelineState) -> PipelineState:
    print("  [1/8] Preprocessing...")
    state["clean_posts"] = preprocess_batch(state["raw_posts"])
    print(f"         {len(state['clean_posts'])} posts after cleaning")
    return state


def embed_node(state: PipelineState) -> PipelineState:
    print("  [2/8] Embedding...")
    if not state["clean_posts"]:
        print("         ⚠️  No posts — skipping")
        state["embeddings"] = []
        return state
    texts = [p["clean_text"] for p in state["clean_posts"]]
    state["embeddings"] = embed_texts(texts)
    upsert_posts(state["clean_posts"], state["embeddings"])
    return state


def detect_node(state: PipelineState) -> PipelineState:
    print("  [3/8] Detecting drift...")
    if not state["embeddings"]:
        state["drift_score"]  = 0.0
        state["changepoints"] = []
        return state
    state["drift_score"]  = compute_drift_score(state["embeddings"])
    state["changepoints"] = detect_changepoints(state["embeddings"])
    print(f"         Drift score: {state['drift_score']:.4f}")
    return state


def rag_node(state: PipelineState) -> PipelineState:
    print("  [4/8] RAG retrieval...")
    if not state["clean_posts"]:
        state["rag_context"] = ""
        return state
    state["rag_context"] = retrieve_context(state["clean_posts"][-1]["clean_text"])
    return state


def llm_node(state: PipelineState) -> PipelineState:
    print("  [5/8] LLM reasoning (Gemma 4 31B)...")
    if not state["clean_posts"]:
        state["llm_output"] = {
            "is_signal":  False,
            "confidence": 0.0,
            "reasoning":  "No posts found"
        }
        return state
    latest      = state["clean_posts"][-1]["clean_text"]
    market_info = str(state["market_snapshot"])
    state["llm_output"] = reason_over_context(
        latest, state["rag_context"], market_info
    )
    verdict = "SIGNAL" if state["llm_output"]["is_signal"] else "NO_SIGNAL"
    print(f"         Verdict: {verdict}")

    # log signal for correlator if detected
    if state["llm_output"]["is_signal"]:
        log_signal(
            keyword    = state.get("keyword", "unknown"),
            verdict    = verdict,
            confidence = state["llm_output"].get("confidence", 0.5),
            reasoning  = state["llm_output"].get("reasoning", "")
        )
    return state


def bayes_node(state: PipelineState) -> PipelineState:
    print("  [6/8] Updating Bayesian model...")
    bayes.update(state["llm_output"]["is_signal"])
    state["bayes_probability"] = bayes.probability()
    print(f"         P(shift) = {state['bayes_probability']:.4f}")
    return state


def alert_node(state: PipelineState) -> PipelineState:
    print("  [7/8] 🚨 ALERT FIRED!")
    state["alert_fired"] = True
    state["reasoning"]   = state["llm_output"]["reasoning"]
    print(f"         P(shift)  = {state['bayes_probability']:.4f}")
    print(f"         Reasoning = {state['reasoning'][:80]}...")
    return state


def skip_node(state: PipelineState) -> PipelineState:
    print("  [7/8] ✅ No signal — monitoring next window")
    state["alert_fired"] = False
    return state


def correlator_node(state: PipelineState) -> PipelineState:
    """
    Takes a follow-up market snapshot and validates whether
    the signal (if any) preceded an actual market move.
    Feeds result back into Bayesian model as ground truth.
    """
    print("  [8/8] Correlator — validating signal against market...")

    before = state.get("before_snapshot", {})
    if not before or not before.get("markets"):
        print("         ⚠️  No before snapshot — skipping correlation")
        state["correlation_result"] = None
        return state

    # take after snapshot now
    after = snapshot_markets()
    state["after_snapshot"] = after

    # validate
    result = validate_signal(
        before_snapshot = before,
        after_snapshot  = after,
        keyword         = state.get("keyword", "")
    )
    state["correlation_result"] = result

    # update Bayesian with ground truth
    if result["confirmed"]:
        print("         📈 Market moved! Bayesian updated with confirmed signal")
        bayes.update(True)
    else:
        print("         📉 No market move. Bayesian updated with miss")
        bayes.update(False)

    state["bayes_probability"] = bayes.probability()
    print(f"         Updated P(shift) = {state['bayes_probability']:.4f}")
    return state


def route_after_bayes(state: PipelineState) -> str:
    return "alert" if state["bayes_probability"] > BAYESIAN_THRESHOLD else "skip"


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("preprocess",  preprocess_node)
    graph.add_node("embed",       embed_node)
    graph.add_node("detect",      detect_node)
    graph.add_node("rag",         rag_node)
    graph.add_node("llm",         llm_node)
    graph.add_node("bayes",       bayes_node)
    graph.add_node("alert",       alert_node)
    graph.add_node("skip",        skip_node)
    graph.add_node("correlator",  correlator_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "embed")
    graph.add_edge("embed",      "detect")
    graph.add_edge("detect",     "rag")
    graph.add_edge("rag",        "llm")
    graph.add_edge("llm",        "bayes")
    graph.add_conditional_edges("bayes", route_after_bayes, {
        "alert": "alert",
        "skip":  "skip"
    })
    # both alert and skip feed into correlator
    graph.add_edge("alert",      "correlator")
    graph.add_edge("skip",       "correlator")
    graph.add_edge("correlator", END)

    return graph.compile()


if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.ingestion.polymarket_agent import get_active_markets
    from config import get_dynamic_keywords

    print("\n🧠 AGENTIC SIGNAL DETECTION ENGINE")
    print("=" * 45)

    keywords = get_dynamic_keywords(limit=3)
    markets  = get_active_markets(limit=5)
    pipeline = build_pipeline()

    for kw in keywords:
        print(f"\n📡 Running pipeline for: {kw.upper()}")
        print("-" * 45)

        # take before snapshot BEFORE scraping
        before = snapshot_markets()
        raw_posts = scrape_hackernews(kw, limit=20)

        if not raw_posts:
            print(f"  ⚠️  No posts found for '{kw}' — skipping")
            continue

        result = pipeline.invoke({
            "raw_posts":          raw_posts,
            "clean_posts":        [],
            "embeddings":         [],
            "drift_score":        0.0,
            "changepoints":       [],
            "rag_context":        "",
            "llm_output":         {},
            "market_snapshot":    markets[0] if markets else {},
            "before_snapshot":    before,
            "after_snapshot":     {},
            "bayes_probability":  0.0,
            "alert_fired":        False,
            "reasoning":          None,
            "keyword":            kw,
            "correlation_result": None
        })

        corr = result.get("correlation_result")
        print(f"\n  Result:")
        print(f"  Alert fired : {result['alert_fired']}")
        print(f"  P(shift)    : {result['bayes_probability']:.4f}")
        print(f"  Drift       : {result['drift_score']:.4f}")
        if corr:
            print(f"  Confirmed   : {corr['confirmed']}")
            if corr['related_moves']:
                for m in corr['related_moves']:
                    print(f"  Market move : {m['question'][:50]} {m['direction']} {m['change']:+.1f}%")
