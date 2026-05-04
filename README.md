# 🧠 Agentic Insider Signal Detection Engine

> Detects early narrative shifts in online discussions that may signal information leakage **before** prediction market prices move.

---

## 📌 What Is This?

Prediction markets (like **Polymarket**) let people bet on real-world outcomes — elections, geopolitical events, crypto prices, etc. The prices reflect the crowd's collective belief about what will happen.

**The core hypothesis:** Before a market price moves, there are early warning signals in online conversations that hint something is changing. People with inside knowledge start discussing things subtly. Rumors begin circulating. Narrative framing quietly shifts.

This system detects those shifts **before** prices move.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           LAYER 1 — DATA COLLECTION             │
│   HackerNews & Twitter Scrapers · Polymarket    │
└────────────────────────┬────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────┐
│          LAYER 2 — PROCESSING PIPELINE          │
│   Preprocessor  →  Embedder  →  ChromaDB        │
└────────────────────────┬────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────┐
│        LAYER 3 — RAG + SIGNAL INTELLIGENCE      │
│   Change Detector  →  RAG Retriever             │
│   →  LLM Reasoner  →  Bayesian Model            │
│   →  Correlator                                 │
└────────────────────────┬────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────┐
│              LAYER 4 — OUTPUT                   │
│       Alert Engine  ·  Streamlit Dashboard      │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Data — Social Media | Algolia HN API, Tweepy (Twitter) |
| Data — Prediction Markets | Polymarket REST API |
| Text Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store + RAG | `chromadb` |
| Change Detection | `ruptures` (PELT algorithm) |
| LLM Reasoning | Gemma 4 31B via Ollama Cloud |
| Bayesian Model | `scipy.stats` Beta-Binomial |
| Orchestration | `LangGraph` |
| Dashboard | `streamlit` + `plotly` |

---

## 📁 Project Structure

```
signal-detection-engine/
├── agents/
│   ├── ingestion/          # Data collection agents
│   ├── processing/         # Embedding + vector store
│   └── intelligence/       # Change detection, RAG, LLM, Bayesian
├── pipeline/
│   └── graph.py            # LangGraph orchestration
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── fine_tuning/            # Optional QLoRA fine-tuning
├── data/
│   └── chroma_db/          # Persisted vector store
├── config.py               # Settings + dynamic keyword fetching
└── requirements.txt
```

---

## ⚙️ Setup

### 1 — Clone and install
```bash
git clone https://github.com/BhaveshMRA/agentic-signal-detection-engine.git
cd agentic-signal-detection-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Create `.env`
```
OLLAMA_API_KEY=your_ollama_key
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=gemma4:31b-cloud
```

### 3 — Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 4 — Or run the pipeline directly
```bash
python3 -m pipeline.graph
```

---

## 🎛️ Dashboard Views

The dashboard has three views for different audiences:

**🟢 Layman View**
Traffic light bar chart + emoji status card. Plain English explanations of what the AI found. No jargon.

**📈 Analyst View**
Trend lines over time, heatmap grid, Bayesian probability gauge. For someone who understands charts.

**🔬 Technical View**
Full run log with AI reasoning, raw data table with CSV export, Bayesian update history chart.

---

## 🔑 API Keys Needed

| Service | Where to get | Cost |
|---|---|---|
| Ollama Cloud | ollama.com/settings/keys | Free tier available |
| Polymarket | No key needed | Public API |
| HackerNews | No key needed | Public API |

---

## 🧠 How It Works

1. **Keywords auto-fetched** from Polymarket Events API — sorted by 24hr trading volume, filtered by relevant tags (crypto, finance, politics, geopolitics). Zero hardcoding.

2. **HackerNews posts scraped** for each keyword using smart term extraction — long market titles are shortened to effective search terms automatically.

3. **Posts embedded** using sentence-BERT into 384-dimensional vectors and stored in ChromaDB with timestamps.

4. **PELT algorithm** detects changepoints in the embedding time series — flags when narrative meaning suddenly shifts.

5. **RAG retrieval** fetches top-K historically similar posts from ChromaDB to give the LLM context.

6. **Gemma 4 31B** reasons over the current post + historical context + market snapshot. Returns SIGNAL / NO_SIGNAL with confidence and reasoning.

7. **Bayesian model** updates P(market shift) after each run. Starts at 50%, rises on confirmed signals, drops on misses.

8. **Alert fires** when P(market shift) > 70%.

---

## 📊 Key Concepts

**Drift Score** — how much the meaning of posts about a topic is changing. Measured as cosine distance between the first half and second half of the embedding window.

**SIGNAL vs High Drift** — high drift means the conversation is changing a lot, but the AI may still say NO_SIGNAL if the posts are not relevant to market movements.

**Bayesian Calibration** — over time, the model learns how trustworthy its signals are by tracking how often signals preceded actual market moves.

**Correlator** — compares timestamps of detected signals against Polymarket price snapshots to measure how many minutes ahead the social signal appeared.

---

## 📍 Where We Left Off

### Last Session: May 4, 2026

**Done:**
- ✅ Full 8-node LangGraph pipeline end to end
- ✅ HackerNews + Twitter + Polymarket live data ingestion
- ✅ Dynamic LLM Search Query Generation for Polymarket topics
- ✅ Twitter Agent with Fallback Mock Generation
- ✅ sentence-BERT embeddings + ChromaDB
- ✅ PELT change detection + drift scoring
- ✅ RAG retrieval pipeline
- ✅ Gemma 4 31B reasoning via Ollama Cloud
- ✅ Bayesian probability model
- ✅ Correlator agent built and tested
- ✅ Correlator feedback loop wired into Bayesian model
- ✅ 3-view Streamlit dashboard (Layman / Analyst / Technical)
- ✅ Dynamic keywords from Polymarket Events API
- ✅ Smart HackerNews search term extraction
- ✅ keyword-market map (topics and markets always in sync)
- ✅ Pushed to GitHub

**Next session:**
- ⬜ Deploy to Streamlit Cloud
- ⬜ QLoRA fine-tuning on Colab Pro (optional)

### Resume Commands
```bash
cd '/Applications/Projects-Claudcode/Agentic Insider Signal Detection Engine'
source venv/bin/activate
streamlit run dashboard/app.py
```

---

## 📚 References

- [Polymarket API Docs](https://docs.polymarket.com/)
- [HackerNews Algolia API](https://hn.algolia.com/api)
- [sentence-transformers](https://www.sbert.net/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [ruptures — changepoint detection](https://centre-borelli.github.io/ruptures-docs/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Ollama Cloud](https://ollama.com)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
