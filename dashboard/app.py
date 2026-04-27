import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingestion.hackernews_agent import scrape_hackernews
from agents.ingestion.polymarket_agent import get_active_markets
from agents.processing.preprocessor import preprocess_batch
from agents.processing.embedder import embed_texts
from agents.processing.vector_store import upsert_posts, get_collection_size
from agents.intelligence.change_detector import compute_drift_score
from agents.intelligence.rag_retriever import retrieve_context
from agents.intelligence.llm_reasoner import reason_over_context
from agents.intelligence.bayesian_model import BayesianSignalModel
from config import TRACKED_KEYWORDS, BAYESIAN_THRESHOLD

st.set_page_config(page_title="Signal Detection Engine", page_icon="🧠", layout="wide")

if "bayes"         not in st.session_state: st.session_state.bayes         = BayesianSignalModel()
if "alerts"        not in st.session_state: st.session_state.alerts        = []
if "drift_history" not in st.session_state: st.session_state.drift_history = []
if "run_count"     not in st.session_state: st.session_state.run_count     = 0
if "log"           not in st.session_state: st.session_state.log           = []

# ── HEADER ────────────────────────────────────────────────
st.title("🧠 Agentic Insider Signal Detection Engine")
st.caption("We monitor online discussions and detect unusual narrative shifts that may predict market movements — before prices change.")
st.divider()

# ── WHAT IS THIS? ─────────────────────────────────────────
with st.expander("👋 What is this system? (Click to learn)"):
    st.markdown("""
**In simple terms:** People on the internet often *talk* about something before it affects markets.
This system watches online discussions 24/7 and uses AI to detect when the conversation around a topic
suddenly shifts — which can be an early warning sign of a market price movement.

**How it works step by step:**
1. 🔍 **Scrape** — Pulls latest posts from HackerNews about topics like Bitcoin, Elections, Fed Rate
2. 🧹 **Clean** — Removes noise, duplicates, irrelevant text
3. 🔢 **Understand** — Converts each post into numbers that capture its *meaning*
4. 📊 **Watch for shifts** — Detects when the meaning of posts suddenly changes
5. 🧠 **AI Reasoning** — Gemma 4 AI reads the posts and decides: is this a real signal or just noise?
6. 🎲 **Probability** — Updates a running score of how likely a market shift is
7. 🚨 **Alert** — Fires an alert when confidence crosses 70%
""")

# ── METRICS ───────────────────────────────────────────────
st.subheader("📈 System Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Topics Watched",        len(TRACKED_KEYWORDS),
          help="Keywords we're monitoring on HackerNews")
m2.metric("Posts in Memory",       get_collection_size(),
          help="Total posts stored and remembered by the system")
m3.metric("Alerts Fired",          len(st.session_state.alerts),
          help="Times the system was confident enough to raise an alarm")
m4.metric("Times Run",             st.session_state.run_count,
          help="How many times you've run the detection pipeline")
m5.metric("Alert Confidence Bar",  f"{BAYESIAN_THRESHOLD*100:.0f}%",
          help="System needs to be this confident before firing an alert")
st.divider()

# ── CONTROLS ──────────────────────────────────────────────
st.subheader("⚙️ What Should We Watch?")
col_kw, col_posts, col_markets = st.columns([2, 1, 2])

with col_kw:
    selected_kw = st.multiselect(
        "Topics to monitor right now",
        TRACKED_KEYWORDS,
        default=TRACKED_KEYWORDS[:2],
        help="Select which topics you want the AI to analyse"
    )

with col_posts:
    n_posts = st.slider("How many posts to analyse per topic",
                        5, 30, 10,
                        help="More posts = more accurate but slower")

with col_markets:
    st.markdown("**🏦 What Polymarket is betting on right now**")
    st.caption("These are real prediction markets people are actively trading")
    try:
        markets = get_active_markets(limit=3)
        for m in markets:
            yes = f"{float(m['yes_price'])*100:.0f}%" if m['yes_price'] else "N/A"
            st.markdown(f"- **{yes} chance** · {m['question'][:55]}...")
    except:
        st.warning("Could not fetch markets")
        markets = []

st.divider()

# ── RUN BUTTON ────────────────────────────────────────────
run = st.button("🚀 Run the Detection Pipeline Now", type="primary", use_container_width=True)

if run:
    st.session_state.run_count += 1
    run_id      = st.session_state.run_count
    market_snap = markets[0] if markets else {"question": "Unknown"}
    progress    = st.progress(0, text="Starting...")

    for idx, kw in enumerate(selected_kw):
        progress.progress(idx / len(selected_kw),
                          text=f"Step {idx+1} of {len(selected_kw)}: Analysing '{kw}' posts...")

        with st.spinner(f"Scraping and understanding '{kw}' posts..."):
            raw   = scrape_hackernews(kw, limit=n_posts)
            posts = preprocess_batch(raw)
            if not posts:
                continue
            texts = [p["clean_text"] for p in posts]
            embs  = embed_texts(texts)
            upsert_posts(posts, embs)
            drift = compute_drift_score(embs)
            st.session_state.drift_history.append({
                "run": run_id, "keyword": kw, "drift": round(drift, 4),
                "label": f"Run {run_id}"
            })

        with st.spinner(f"Asking Gemma 4 AI to reason about '{kw}'..."):
            latest     = posts[-1]
            context    = retrieve_context(latest["clean_text"])
            llm_result = reason_over_context(latest["clean_text"], context, str(market_snap))
            st.session_state.bayes.update(llm_result["is_signal"])
            p       = st.session_state.bayes.probability()
            verdict = "🚨 SIGNAL DETECTED" if llm_result["is_signal"] else "✅ No Signal"

            st.session_state.log.append({
                "run_id":   run_id,
                "keyword":  kw.upper(),
                "drift":    round(drift, 4),
                "verdict":  verdict,
                "p":        round(p, 4),
                "reason":   llm_result["reasoning"],
                "post":     latest["title"]
            })

            if p > BAYESIAN_THRESHOLD:
                st.session_state.alerts.append({
                    "time":    pd.Timestamp.now().strftime("%H:%M:%S"),
                    "keyword": kw.upper(),
                    "p":       round(p, 4),
                    "drift":   round(drift, 4),
                    "verdict": verdict,
                    "reason":  llm_result["reasoning"]
                })

    progress.progress(1.0, text="✅ Done!")
    st.rerun()

st.divider()

# ── MAIN LAYOUT ───────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    # ── DRIFT CHART ───────────────────────────────────────
    st.subheader("📊 How Much Is the Conversation Shifting?")
    st.caption("""
**What you're looking at:** Each line tracks how much the meaning of online posts about a topic 
is changing over time. A low score means people are saying the same things as usual. 
A high score means the narrative is suddenly shifting — which could mean something is happening.
**The red dashed line** is the warning zone — if a topic crosses it, our AI investigates further.
    """)

    if st.session_state.drift_history:
        df = pd.DataFrame(st.session_state.drift_history)
        fig = go.Figure()
        colors = {"trump": "#F06292", "bitcoin": "#FFB74D",
                  "crypto": "#4FC3F7", "election": "#81C784", "fed rate": "#CE93D8"}

        for kw in df["keyword"].unique():
            kw_df = df[df["keyword"] == kw]
            fig.add_trace(go.Scatter(
                x=kw_df["run"].astype(int),
                y=kw_df["drift"],
                name=kw.upper(),
                mode="lines+markers",
                line=dict(color=colors.get(kw, "#ffffff"), width=3),
                marker=dict(size=10),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Run #%{x}<br>"
                    "Shift Score: %{y:.4f}<br>"
                    "<extra></extra>"
                )
            ))

        # signal zone annotation
        fig.add_hrect(y0=0.15, y1=0.5, fillcolor="red",
                      opacity=0.08, line_width=0)
        fig.add_hline(y=0.15, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="⚠️ Warning Zone — AI investigates above this line",
                      annotation_position="top right",
                      annotation_font_color="red")

        fig.update_layout(
            xaxis_title="Detection Run Number",
            yaxis_title="Narrative Shift Score",
            xaxis=dict(tickmode="linear", dtick=1,
                       title_font=dict(size=12)),
            yaxis=dict(title_font=dict(size=12)),
            height=320,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(
                orientation="h", y=-0.25,
                title_text="Topics: "
            ),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # plain english summary below chart
        if len(df) > 0:
            latest_drifts = df.groupby("keyword")["drift"].last()
            st.markdown("**Latest readings:**")
            cols = st.columns(len(latest_drifts))
            for i, (kw, val) in enumerate(latest_drifts.items()):
                status = "🔴 High shift" if val > 0.15 else "🟢 Normal"
                cols[i].metric(kw.upper(), f"{val:.4f}", status)
    else:
        st.info("👆 Hit **Run the Detection Pipeline** to start seeing data here")

    st.divider()

    # ── RUN LOG ───────────────────────────────────────────
    st.subheader("🔍 What Did the AI Find?")
    st.caption("Every time you run the pipeline, the AI reads posts and explains what it found.")

    if st.session_state.log:
        for entry in reversed(st.session_state.log):
            is_signal = "SIGNAL" in entry["verdict"]
            bg        = "#1a3a1a" if not is_signal else "#3a1a1a"
            border    = "#2d5a2d" if not is_signal else "#5a2d2d"
            st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:10px;
            padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:15px;font-weight:700;">
      Run #{entry['run_id']} · {entry['keyword']}
    </span>
    <span style="font-size:14px;font-weight:600;">{entry['verdict']}</span>
  </div>
  <div style="font-size:12px;color:#aaa;margin-bottom:8px;">
    📰 Post analysed: <i>{entry['post'][:90]}...</i>
  </div>
  <div style="display:flex;gap:24px;font-size:13px;margin-bottom:8px;">
    <span>📊 Shift Score: <b>{entry['drift']}</b></span>
    <span>🎲 Market Shift Probability: <b>{entry['p']*100:.0f}%</b></span>
  </div>
  <div style="font-size:13px;color:#ddd;line-height:1.6;
              background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 12px;">
    🧠 <b>AI Reasoning:</b> {entry['reason']}
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("AI analysis results will appear here after you run the pipeline")

with right:
    # ── PROBABILITY GAUGE ─────────────────────────────────
    st.subheader("🎲 How Likely Is a Market Shift?")
    st.caption("""
This gauge shows the system's current confidence that a market price shift is coming.
It starts at 50% (no idea) and moves up or down as the AI detects signals.
**Above 70% = Alert fires.**
    """)

    p_val   = st.session_state.bayes.probability()
    summary = st.session_state.bayes.summary()

    # colour based on value
    bar_color = "#4CAF50" if p_val < 0.4 else "#FF9800" if p_val < 0.7 else "#F44336"

    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(p_val * 100, 1),
        title = {"text": "Confidence a Market Move is Coming",
                 "font": {"size": 14}},
        number= {"suffix": "%", "font": {"size": 40}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickvals": [0, 25, 50, 70, 100],
                     "ticktext": ["0%", "25%\nVery Low", "50%\nUncertain",
                                  "70%\n⚠️ Alert", "100%\nCertain"]},
            "bar":   {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  50], "color": "#1B5E20", "name": "Low"},
                {"range": [50, 70], "color": "#E65100", "name": "Medium"},
                {"range": [70, 100],"color": "#B71C1C", "name": "High"}
            ],
            "threshold": {
                "line":      {"color": "white", "width": 4},
                "thickness": 0.85,
                "value":     70
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # plain english status
    if p_val < 0.4:
        st.success("🟢 **Status: Calm** — Nothing unusual detected. Markets appear stable.")
    elif p_val < 0.7:
        st.warning("🟡 **Status: Watching** — Some unusual signals. Monitoring closely.")
    else:
        st.error("🔴 **Status: ALERT** — High confidence of incoming market movement!")

    st.markdown("---")
    st.markdown("**What's behind this number:**")
    c1, c2 = st.columns(2)
    c1.metric("Times signal was right", int(summary["alpha"] - 1),
              help="How many times AI said SIGNAL and was correct")
    c2.metric("Times signal was wrong", int(summary["beta"] - 1),
              help="How many times AI said SIGNAL but nothing happened")
    c1.metric("Total analyses run",    int(summary["alpha"] + summary["beta"] - 2))
    c2.metric("Current confidence",    summary["confidence"])

    st.divider()

    # ── ALERTS ────────────────────────────────────────────
    st.subheader("🚨 Alerts")
    st.caption("Fires when the system is more than 70% confident a market move is coming.")

    if st.session_state.alerts:
        for a in reversed(st.session_state.alerts[-5:]):
            with st.expander(f"🚨 {a['time']} · {a['keyword']} · {a['p']*100:.0f}% confident"):
                st.error(f"**{a['verdict']}**")
                st.markdown(f"**Shift score at time of alert:** {a['drift']}")
                st.markdown(f"**Market shift probability:** {a['p']*100:.0f}%")
                st.markdown("**Why the AI flagged this:**")
                st.info(a['reason'])
    else:
        st.markdown("""
<div style="background:#1a2a3a;border:1px solid #2a4a6a;border-radius:10px;
            padding:20px;text-align:center;color:#aaa;">
  <div style="font-size:32px;margin-bottom:8px;">👁️</div>
  <div style="font-size:14px;">Watching and waiting...</div>
  <div style="font-size:12px;margin-top:4px;">
    No alerts yet. The system needs to reach 70% confidence before firing.
    Keep running the pipeline to build up signal history.
  </div>
</div>
""", unsafe_allow_html=True)
