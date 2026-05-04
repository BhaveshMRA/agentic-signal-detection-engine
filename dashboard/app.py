import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingestion.hackernews_agent import scrape_hackernews
from agents.processing.preprocessor import preprocess_batch
from agents.processing.embedder import embed_texts
from agents.processing.vector_store import upsert_posts, get_collection_size
from agents.intelligence.change_detector import compute_drift_score
from agents.intelligence.rag_retriever import retrieve_context
from agents.intelligence.llm_reasoner import reason_over_context, generate_search_queries
from agents.intelligence.bayesian_model import BayesianSignalModel
from agents.intelligence.correlator import snapshot_markets
from config import get_keyword_market_map, BAYESIAN_THRESHOLD

st.set_page_config(page_title="Signal Detection Engine", page_icon="🧠", layout="wide")

if "bayes"         not in st.session_state: st.session_state.bayes         = BayesianSignalModel()
if "alerts"        not in st.session_state: st.session_state.alerts        = []
if "drift_history" not in st.session_state: st.session_state.drift_history = []
if "run_count"     not in st.session_state: st.session_state.run_count     = 0
if "log"           not in st.session_state: st.session_state.log           = []
if "correlations"  not in st.session_state: st.session_state.correlations  = []

KEYWORD_MARKET_MAP = get_keyword_market_map(limit=8)
LIVE_KEYWORDS      = list(KEYWORD_MARKET_MAP.keys())

st.title("🧠 Agentic Insider Signal Detection Engine")
st.caption("Monitors HackerNews + Polymarket · Detects narrative shifts before market prices move")
st.divider()

with st.expander("👋 What is this system? (Click to learn)"):
    st.markdown("""
**In simple terms:** People on the internet often *talk* about something before it affects markets.
This system watches online discussions 24/7 and uses AI to detect when the conversation around a topic
suddenly shifts — which can be an early warning sign of a market price movement.

**How it works:**
1. 🎯 **Topics** — Auto-fetched from Polymarket's most actively traded markets right now
2. 🔍 **Scrape** — Pulls latest posts from HackerNews about those topics
3. 🧹 **Clean** — Removes noise, duplicates, irrelevant text
4. 🔢 **Understand** — Converts each post into numbers that capture its meaning
5. 📊 **Watch for shifts** — Detects when the meaning of posts suddenly changes
6. 🧠 **AI Reasoning** — Gemma 4 AI reads posts and decides: real signal or noise?
7. 🎲 **Probability** — Updates a running score of how likely a market shift is
8. 🔗 **Correlator** — Checks if Polymarket prices actually moved after the signal
8. 🚨 **Alert** — Fires when confidence crosses 70%

**Important distinction:**
- 📊 **High Drift** = the conversation is changing a lot
- 🚨 **Signal** = the AI believes this change could affect market prices
- 🔗 **Confirmed** = the market actually moved after the signal fired
""")

st.subheader("📈 System Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Topics Watched",   len(LIVE_KEYWORDS))
m2.metric("Posts in Memory",  get_collection_size())
m3.metric("Alerts Fired",     len(st.session_state.alerts))
m4.metric("Times Run",        st.session_state.run_count)
m5.metric("Alert Threshold",  f"{BAYESIAN_THRESHOLD*100:.0f}%")
st.divider()

st.subheader("⚙️ What Should We Watch?")
col_kw, col_posts, col_markets = st.columns([2, 1, 2])

if "generated_options" not in st.session_state:
    st.session_state.generated_options = []
if "selected_market_topic" not in st.session_state:
    st.session_state.selected_market_topic = LIVE_KEYWORDS[0] if LIVE_KEYWORDS else ""

with col_kw:
    selected_kw = st.selectbox(
        "1. Select Polymarket Topic",
        LIVE_KEYWORDS,
        index=0,
        help="Auto-fetched live from Polymarket's most actively traded markets"
    )
    
    if selected_kw != st.session_state.selected_market_topic:
        st.session_state.selected_market_topic = selected_kw
        st.session_state.generated_options = [] # reset on change
        
    generate = st.button("✨ 2. Generate Search Options (LLM)", use_container_width=True)
    if generate:
        with st.spinner("Generating best search queries..."):
            st.session_state.generated_options = generate_search_queries(selected_kw)

    search_query = None
    if st.session_state.generated_options:
        search_query = st.radio("3. Select exact search query to monitor:", st.session_state.generated_options)
        
with col_posts:
    n_posts = st.slider("Posts to analyse per topic", 5, 30, 10)

with col_markets:
    st.markdown("**🏦 Linked Polymarket Market**")
    st.caption("Market linked to your selected topic")
    if selected_kw:
        market = KEYWORD_MARKET_MAP.get(selected_kw, {})
        if market and market.get("yes_price"):
            yes  = f"{float(market['yes_price'])*100:.0f}%"
            tags = ", ".join(market.get("tags", [])[:2])
            st.markdown(f"- **{yes} chance** · {market['question'][:55]}...")
            st.caption(f"  Tags: {tags} · Vol: ${market['volume']:,.0f}")
    else:
        st.info("Select a topic on the left to see linked markets")

st.divider()

run = st.button("🚀 4. Run the Detection Pipeline Now", type="primary", use_container_width=True, disabled=(not search_query))

if run and search_query:
    kw = selected_kw  # keep the original kw for correlation/display
    
    st.session_state.run_count += 1
    run_id   = st.session_state.run_count
    
    with st.spinner(f"Taking market snapshot for '{kw}'..."):
        before = snapshot_markets()

    with st.spinner(f"Scraping HackerNews & Twitter for '{search_query}'..."):
        from agents.ingestion.twitter_agent import scrape_twitter
        raw_hn = scrape_hackernews(search_query, limit=n_posts)
        raw_tw = scrape_twitter(search_query, limit=n_posts)
        raw = raw_hn + raw_tw
        
        posts = preprocess_batch(raw)
        
        if not posts:
            st.warning("No posts found for this query!")
        else:
            texts = [p["clean_text"] for p in posts]
            embs  = embed_texts(texts)
            upsert_posts(posts, embs)
            drift = compute_drift_score(embs)

            with st.spinner(f"Asking Gemma 4 AI about '{kw}'..."):
                latest      = posts[-1]
                context     = retrieve_context(latest["clean_text"])
                market_info = str(KEYWORD_MARKET_MAP.get(kw, {}))
                llm_result  = reason_over_context(latest["clean_text"], context, market_info)
                st.session_state.bayes.update(llm_result["is_signal"])
                p       = st.session_state.bayes.probability()
                verdict = "🚨 SIGNAL" if llm_result["is_signal"] else "✅ No Signal"

            with st.spinner(f"Correlator validating '{kw}' against market..."):
                from agents.intelligence.correlator import validate_signal
                after  = snapshot_markets()
                corr   = validate_signal(before, after, kw)
                if llm_result["is_signal"]:
                    st.session_state.bayes.update(corr["confirmed"])
                    p = st.session_state.bayes.probability()
                st.session_state.correlations.append({
                    "run_id":    run_id,
                    "keyword":   kw,
                    "confirmed": corr["confirmed"],
                    "moves":     corr["related_moves"]
                })

            st.session_state.drift_history.append({
                "run":       run_id,
                "keyword":   kw,
                "drift":     round(drift, 4),
                "is_signal": llm_result["is_signal"],
                "verdict":   verdict,
                "confirmed": corr["confirmed"],
                "p":         round(p, 4)
            })

            st.session_state.log.append({
                "run_id":    run_id,
                "keyword":   kw,
                "drift":     round(drift, 4),
                "verdict":   verdict,
                "confirmed": corr["confirmed"],
                "p":         round(p, 4),
                "reason":    llm_result["reasoning"],
                "posts":     [{"title": p["title"], "source": p["source"], "text": p.get("text", "")} for p in posts[-10:]]
            })

            if p > BAYESIAN_THRESHOLD:
                st.session_state.alerts.append({
                    "time":      pd.Timestamp.now().strftime("%H:%M:%S"),
                    "keyword":   kw,
                    "p":         round(p, 4),
                    "drift":     round(drift, 4),
                    "verdict":   verdict,
                    "confirmed": corr["confirmed"],
                    "reason":    llm_result["reasoning"]
                })

            st.rerun()

st.divider()

st.subheader("📊 Signal Analysis Dashboard")
view = st.radio(
    "👁️ Choose your view:",
    ["🟢 Layman — Show me what's happening simply",
     "📈 Analyst — Show me the trends over time",
     "🔬 Technical — Show me the full data"],
    horizontal=True
)
st.divider()

df = pd.DataFrame(st.session_state.drift_history) if st.session_state.drift_history else pd.DataFrame()

if "Layman" in view:
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 🚦 Drift Score + AI Verdict + Market Confirmation")
        st.caption("""
**Drift Score** = how much the conversation is changing (bar length).
**AI Verdict** = whether the AI thinks this is a real market signal.
**Confirmed** = whether Polymarket prices actually moved after the signal.
        """)

        if not df.empty:
            latest = df.groupby("keyword").last().reset_index()
            latest = latest.sort_values("drift", ascending=True)

            bar_colors = []
            bar_labels = []
            for _, row in latest.iterrows():
                if row.get("is_signal") and row.get("confirmed"):
                    bar_colors.append("#F44336")
                    bar_labels.append("🚨 SIGNAL + ✅ Market moved!")
                elif row.get("is_signal") and not row.get("confirmed"):
                    bar_colors.append("#FF9800")
                    bar_labels.append("🚨 SIGNAL — market move not confirmed yet")
                elif row["drift"] > 0.20:
                    bar_colors.append("#FFC107")
                    bar_labels.append("🟡 High drift — AI says not a signal")
                else:
                    bar_colors.append("#4CAF50")
                    bar_labels.append("🟢 Normal — nothing unusual")

            fig = go.Figure(go.Bar(
                x=latest["drift"],
                y=latest["keyword"].str[:35],
                orientation="h",
                marker_color=bar_colors,
                text=bar_labels,
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Drift: %{x:.4f}<extra></extra>"
            ))
            fig.add_vline(x=0.15, line_dash="dash", line_color="orange",
                          annotation_text="⚠️ Watch Zone")
            fig.add_vline(x=0.20, line_dash="dash", line_color="red",
                          annotation_text="🚨 High Drift")
            fig.update_layout(
                xaxis_title="Drift Score (0 = no change, 0.5 = big shift)",
                yaxis_title="",
                height=max(280, len(latest) * 70),
                margin=dict(l=0, r=350, t=20, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**In plain English:**")
            for _, row in latest.sort_values("drift", ascending=False).iterrows():
                is_sig    = row.get("is_signal", False)
                confirmed = row.get("confirmed", False)
                drift     = row["drift"]
                kw        = row["keyword"]
                if is_sig and confirmed:
                    st.error(f"🚨 **{kw}** — Signal detected AND market confirmed it moved! Drift: {drift:.3f}")
                elif is_sig:
                    st.warning(f"🟡 **{kw}** — AI detected signal (drift: {drift:.3f}) but market hasn't moved yet. Watching.")
                elif drift > 0.20:
                    st.warning(f"🟡 **{kw}** — Lots of chatter (drift: {drift:.3f}) but AI says not a market signal.")
                else:
                    st.success(f"🟢 **{kw}** — All normal. Nothing unusual. (drift: {drift:.3f})")

            st.markdown("""
<div style="background:#1a2a3a;border:1px solid #2a4a6a;border-radius:10px;
            padding:14px;margin-top:12px;">
<b>💡 How to read this:</b><br>
<span style="font-size:13px;color:#ccc;line-height:1.8;">
🟢 <b>Green bar</b> = conversation is normal, nothing to worry about<br>
🟡 <b>Yellow bar</b> = conversation shifting but AI says not a market signal yet<br>
🚨 <b>Red bar</b> = AI detected a signal AND Polymarket prices confirmed a move<br>
📊 <b>Bar length</b> = how much the narrative is shifting (longer = more change)<br>
🔗 <b>Confirmed</b> = Polymarket price actually moved within minutes of the signal
</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("👆 Run the pipeline first to see results here")

    with right:
        st.markdown("### 🎲 Should We Be Worried?")
        p_val = st.session_state.bayes.probability()

        if p_val < 0.4:
            st.markdown("""
<div style="background:#1B5E20;border-radius:16px;padding:30px;text-align:center;">
  <div style="font-size:60px;">😴</div>
  <div style="font-size:24px;font-weight:700;color:white;margin:10px 0;">ALL CALM</div>
  <div style="font-size:14px;color:#A5D6A7;">Nothing unusual detected.<br>Markets appear stable.</div>
</div>""", unsafe_allow_html=True)
        elif p_val < 0.7:
            st.markdown("""
<div style="background:#E65100;border-radius:16px;padding:30px;text-align:center;">
  <div style="font-size:60px;">👀</div>
  <div style="font-size:24px;font-weight:700;color:white;margin:10px 0;">WATCHING</div>
  <div style="font-size:14px;color:#FFE0B2;">Some signals detected.<br>Monitoring closely.</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#B71C1C;border-radius:16px;padding:30px;text-align:center;">
  <div style="font-size:60px;">🚨</div>
  <div style="font-size:24px;font-weight:700;color:white;margin:10px 0;">ALERT!</div>
  <div style="font-size:14px;color:#FFCDD2;">High confidence of incoming<br>market movement!</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="margin-top:16px;background:#1a2a3a;border-radius:12px;
            padding:16px;text-align:center;">
  <div style="font-size:13px;color:#aaa;">System confidence</div>
  <div style="font-size:48px;font-weight:700;color:white;">{p_val*100:.0f}%</div>
  <div style="font-size:12px;color:#aaa;">Alert fires above 70%</div>
</div>""", unsafe_allow_html=True)

        if st.session_state.log:
            st.markdown("### 🧠 Latest AI Finding")
            last = st.session_state.log[-1]
            confirmed_txt = "✅ Market confirmed" if last.get("confirmed") else "⏳ Not confirmed yet"
            st.markdown(f"""
<div style="background:#1a2a3a;border-radius:10px;padding:14px;margin-top:8px;">
  <div style="font-weight:600;margin-bottom:4px;">{last['keyword'][:40]} · {last['verdict']}</div>
  <div style="font-size:12px;color:#4FC3F7;margin-bottom:6px;">{confirmed_txt}</div>
  <div style="font-size:13px;color:#ccc;line-height:1.5;">{last['reason']}</div>
</div>""", unsafe_allow_html=True)

elif "Analyst" in view:
    st.markdown("### 📈 Narrative Drift Over Time")

    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["📉 Trend Lines", "🌡️ Heatmap", "🎲 Bayesian"])

        with tab1:
            fig    = go.Figure()
            colors = ["#4FC3F7","#81C784","#FFB74D","#F06292",
                      "#CE93D8","#80DEEA","#FFCC02","#FF7043"]
            for i, kw in enumerate(df["keyword"].unique()):
                kw_df = df[df["keyword"] == kw]
                fig.add_trace(go.Scatter(
                    x=kw_df["run"].astype(int),
                    y=kw_df["drift"],
                    name=kw[:30],
                    mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(
                        size=12,
                        symbol=["star" if s else "circle"
                                for s in kw_df.get("is_signal", [False]*len(kw_df))],
                        color=["#F44336" if s else colors[i % len(colors)]
                               for s in kw_df.get("is_signal", [False]*len(kw_df))]
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>Run #%{x}<br>Drift: %{y:.4f}<extra></extra>"
                ))
            fig.add_hrect(y0=0.15, y1=0.6, fillcolor="red", opacity=0.05, line_width=0)
            fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                          annotation_text="Watch Zone (0.15)")
            fig.add_hline(y=0.20, line_dash="dash", line_color="red",
                          annotation_text="High Drift (0.20)")
            fig.update_layout(
                xaxis_title="Pipeline Run #",
                yaxis_title="Narrative Drift Score",
                xaxis=dict(tickmode="linear", dtick=1),
                height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=-0.3),
                hovermode="x unified"
            )
            st.caption("⭐ Star = AI detected SIGNAL · 🔴 Red star = confirmed by market move")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            pivot = df.pivot_table(index="keyword", columns="run", values="drift")
            pivot.index = [k[:35] for k in pivot.index]
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale=[[0,"#1B5E20"],[0.4,"#F9A825"],[1,"#B71C1C"]],
                aspect="auto",
                title="Drift Score Heatmap"
            )
            fig_heat.update_layout(
                height=300, xaxis_title="Run #", yaxis_title="",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Each cell = one topic x one run. Red = high shift, Green = calm.")

        with tab3:
            p_val   = st.session_state.bayes.probability()
            summary = st.session_state.bayes.summary()
            col_g, col_s = st.columns([1, 1])
            with col_g:
                fig_gauge = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = round(p_val * 100, 1),
                    delta = {"reference": 50, "suffix": "%"},
                    title = {"text": "P(Market Shift) %"},
                    number= {"suffix": "%"},
                    gauge = {
                        "axis":  {"range": [0, 100]},
                        "bar":   {"color": "#4FC3F7"},
                        "steps": [
                            {"range": [0,  50], "color": "#1B5E20"},
                            {"range": [50, 70], "color": "#E65100"},
                            {"range": [70, 100],"color": "#B71C1C"}
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.85, "value": 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col_s:
                st.metric("P(shift)",         f"{p_val*100:.1f}%")
                st.metric("Confidence",        summary["confidence"])
                st.metric("Correct signals",   int(summary["alpha"] - 1))
                st.metric("Incorrect signals", int(summary["beta"] - 1))
                st.metric("Total runs",        int(summary["alpha"] + summary["beta"] - 2))
    else:
        st.info("👆 Run the pipeline first to see trend data")

elif "Technical" in view:
    st.markdown("### 🔬 Full Technical View")
    t1, t2, t3, t4, t5 = st.tabs([
        "📋 Run Log", "📊 Raw Data", "🔗 Correlations", "🚨 Alerts", "🎲 Bayesian"
    ])

    with t1:
        if st.session_state.log:
            for entry in reversed(st.session_state.log):
                is_signal = "SIGNAL" in entry["verdict"]
                confirmed = entry.get("confirmed", False)
                bg     = "#1a3a1a" if not is_signal else ("#1a3a3a" if confirmed else "#3a1a1a")
                border = "#2d5a2d" if not is_signal else ("#2d5a5a" if confirmed else "#5a2d2d")
                conf_badge = "✅ Market confirmed" if confirmed else "⏳ Not confirmed"
                st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:10px;
            padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="font-size:15px;font-weight:700;">Run #{entry['run_id']} · {entry['keyword']}</span>
    <span style="font-size:14px;font-weight:600;">{entry['verdict']} · {conf_badge}</span>
  </div>
  <div style="font-size:12px;color:#aaa;margin-bottom:6px;max-height:200px;overflow-y:auto;background:rgba(0,0,0,0.2);padding:8px;border-radius:4px;">
    {"".join(f"<div style='margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:6px;'><div>{'🐦' if p.get('source') == 'twitter' else '👽'} <b>{p.get('title', '')}</b></div><div style='color:#ccc;margin-top:4px;line-height:1.4;'>{p.get('text', '')}</div></div>" for p in entry.get('posts', [{'title': entry.get('post', ''), 'source': 'hackernews', 'text': ''}]))}
  </div>
  <div style="display:flex;gap:24px;font-size:13px;margin-bottom:6px;">
    <span>Drift: <b>{entry['drift']}</b></span>
    <span>P(shift): <b>{entry['p']*100:.0f}%</b></span>
  </div>
  <div style="font-size:13px;color:#ddd;background:rgba(255,255,255,0.05);
              border-radius:6px;padding:8px 12px;line-height:1.5;">
    🧠 {entry['reason']}
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No runs yet")

    with t2:
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download CSV",
                               df.to_csv(index=False),
                               "drift_data.csv", "text/csv")
        else:
            st.info("No data yet")

    with t3:
        st.markdown("**Correlator Results** — Did signals precede actual market moves?")
        if st.session_state.correlations:
            for c in reversed(st.session_state.correlations):
                icon = "✅" if c["confirmed"] else "❌"
                with st.expander(f"{icon} Run #{c['run_id']} · {c['keyword'][:50]}"):
                    if c["confirmed"] and c["moves"]:
                        st.success("Market moved after signal!")
                        for m in c["moves"]:
                            st.write(f"**{m['question'][:60]}**")
                            st.write(f"Before: {m['before']}% → After: {m['after']}% "
                                     f"({m['direction']} {m['change']:+.1f}%)")
                    else:
                        st.info("No related market movement detected within the window.")
        else:
            st.info("No correlation data yet")

    with t4:
        if st.session_state.alerts:
            for a in reversed(st.session_state.alerts):
                conf = "✅ Confirmed" if a.get("confirmed") else "⏳ Unconfirmed"
                with st.expander(f"🚨 {a['time']} · {a['keyword'][:40]} · {a['p']*100:.0f}% · {conf}"):
                    st.error(f"**{a['verdict']}**")
                    st.write(f"**Drift:** {a['drift']}")
                    st.write(f"**Confidence:** {a['p']*100:.0f}%")
                    st.info(a['reason'])
        else:
            st.info("No alerts fired yet")

    with t5:
        summary = st.session_state.bayes.summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("P(shift)",      f"{summary['probability']*100:.1f}%")
        c2.metric("Alpha (hits)",  summary["alpha"])
        c3.metric("Beta (misses)", summary["beta"])
        c1.metric("CI Low",        f"{summary['ci_low']*100:.1f}%")
        c2.metric("CI High",       f"{summary['ci_high']*100:.1f}%")
        c3.metric("Confidence",    summary["confidence"])

        if st.session_state.log:
            st.markdown("**Bayesian Probability Over Time:**")
            runs   = list(range(1, len(st.session_state.log) + 1))
            p_vals = []
            tmp    = BayesianSignalModel()
            for entry in st.session_state.log:
                tmp.update("SIGNAL" in entry["verdict"])
                p_vals.append(tmp.probability())

            fig_b = go.Figure(go.Scatter(
                x=runs, y=[p * 100 for p in p_vals],
                mode="lines+markers",
                line=dict(color="#4FC3F7", width=2),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(79,195,247,0.1)"
            ))
            fig_b.add_hline(y=70, line_dash="dash", line_color="red",
                            annotation_text="Alert Threshold 70%")
            fig_b.update_layout(
                xaxis_title="Pipeline Run #",
                yaxis_title="P(Market Shift) %",
                yaxis=dict(range=[0, 100]),
                height=280,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_b, use_container_width=True)
