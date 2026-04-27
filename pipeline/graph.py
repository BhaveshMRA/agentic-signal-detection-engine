from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.processing.preprocessor import preprocess_batch
from agents.processing.embedder import embed_texts
from agents.processing.vector_store import upsert_posts, query_similar
from agents.intelligence.change_detector import compute_drift_score, detect_changepoints
from agents.intelligence.rag_retriever import retrieve_context
from agents.intelligence.llm_reasoner import reason_over_context
from agents.intelligence.bayesian_model import BayesianSignalModel
from config import BAYESIAN_THRESHOLD

bayes = BayesianSignalModel()

class PipelineState(TypedDict):
    raw_posts:         list[dict]
    clean_posts:       list[dict]
    embeddings:        list[list[float]]
    drift_score:       float
    changepoints:      list[int]
    rag_context:       str
    llm_output:        dict
    market_snapshot:   dict
    bayes_probability: float
    alert_fired:       bool
    reasoning:         Optional[str]

def preprocess_node(state: PipelineState) -> PipelineState:
    print("  [1/7] Preprocessing...")
    state["clean_posts"] = preprocess_batch(state["raw_posts"])
    print(f"         {len(state['clean_posts'])} posts after cleaning")
    return state

def embed_node(state: PipelineState) -> PipelineState:
    print("  [2/7] Embedding...")
    if not state["clean_posts"]:
        print("         ⚠️ No posts to embed — skipping")
        state["embeddings"] = []
        return state
    texts = [p["clean_text"] for p in state["clean_posts"]]
    state["embeddings"] = embed_texts(texts)
    upsert_posts(state["clean_posts"], state["embeddings"])
    return state

def detect_node(state: PipelineState) -> PipelineState:
    print("  [3/7] Detecting narrative drift...")
    if not state["embeddings"]:
        print("         ⚠️ No embeddings — skipping")
        state["drift_score"]  = 0.0
        state["changepoints"] = []
        return state
    state["drift_score"]  = compute_drift_score(state["embeddings"])
    state["changepoints"] = detect_changepoints(state["embeddings"])
    print(f"         Drift score: {state['drift_score']:.4f}")
    return state

def rag_node(state: PipelineState) -> PipelineState:
    print("  [4/7] Retrieving RAG context...")
    if not state["clean_posts"]:
        print("         ⚠️ No posts — skipping RAG")
        state["rag_context"] = ""
        return state
    latest_text      = state["clean_posts"][-1]["clean_text"]
    state["rag_context"] = retrieve_context(latest_text)
    return state

def llm_node(state: PipelineState) -> PipelineState:
    print("  [5/7] LLM reasoning (Gemma 4 31B)...")
    if not state["clean_posts"]:
        print("         ⚠️ No posts — skipping LLM")
        state["llm_output"] = {"is_signal": False, "confidence": 0.0, "reasoning": "No posts found"}
        return state
    latest_text  = state["clean_posts"][-1]["clean_text"]
    market_info  = str(state["market_snapshot"])
    state["llm_output"] = reason_over_context(
        latest_text,
        state["rag_context"],
        market_info
    )
    print(f"         Verdict: {'SIGNAL' if state['llm_output']['is_signal'] else 'NO_SIGNAL'}")
    return state

def bayes_node(state: PipelineState) -> PipelineState:
    print("  [6/7] Updating Bayesian model...")
    bayes.update(state["llm_output"]["is_signal"])
    state["bayes_probability"] = bayes.probability()
    print(f"         P(shift) = {state['bayes_probability']:.4f}")
    return state

def alert_node(state: PipelineState) -> PipelineState:
    print("  [7/7] 🚨 ALERT FIRED!")
    state["alert_fired"] = True
    state["reasoning"]   = state["llm_output"]["reasoning"]
    print(f"         P(shift)  = {state['bayes_probability']:.4f}")
    print(f"         Reasoning = {state['reasoning']}")
    return state

def skip_node(state: PipelineState) -> PipelineState:
    print("  [7/7] ✅ No signal — monitoring next window")
    state["alert_fired"] = False
    return state

def route_after_bayes(state: PipelineState) -> str:
    return "alert" if state["bayes_probability"] > BAYESIAN_THRESHOLD else "skip"

def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("embed",      embed_node)
    graph.add_node("detect",     detect_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("llm",        llm_node)
    graph.add_node("bayes",      bayes_node)
    graph.add_node("alert",      alert_node)
    graph.add_node("skip",       skip_node)

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
    graph.add_edge("alert", END)
    graph.add_edge("skip",  END)

    return graph.compile()


if __name__ == "__main__":
    from agents.ingestion.hackernews_agent import scrape_hackernews
    from agents.ingestion.polymarket_agent import get_active_markets
    from config import get_dynamic_keywords

    print("\n🧠 AGENTIC SIGNAL DETECTION ENGINE")
    print("=" * 45)

    keywords = get_dynamic_keywords(limit=5)   
    markets  = get_active_markets(limit=3)
    pipeline = build_pipeline()

    for kw in keywords:
        print(f"\n📡 Running pipeline for: {kw.upper()}")
        print("-" * 45)

        raw_posts = scrape_hackernews(kw, limit=20)

        result = pipeline.invoke({
            "raw_posts":       raw_posts,
            "clean_posts":     [],
            "embeddings":      [],
            "drift_score":     0.0,
            "changepoints":    [],
            "rag_context":     "",
            "llm_output":      {},
            "market_snapshot": markets[0] if markets else {},
            "bayes_probability": 0.0,
            "alert_fired":     False,
            "reasoning":       None
        })

        print(f"\n  Result → Alert: {result['alert_fired']} | P: {result['bayes_probability']:.4f} | Drift: {result['drift_score']:.4f}")
