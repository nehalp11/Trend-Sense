<h1 align="center">🔥 TrendSense</h1>

<p align="center">
  <b>Emerging Indian Fashion Trend Detection Using RAR on BERT</b><br>
  <i>Predicting fashion micro-trends 6–8 weeks before mainstream platforms</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/BERT-FF6F00?style=flat&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/CLIP-412991?style=flat&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-green?style=flat"/>
</p>

---

## 📌 What is TrendSense?

TrendSense is an AI-powered fashion trend detection system that identifies **emerging Indian fashion micro-trends 6–8 weeks before they appear on mainstream platforms** like Instagram and Pinterest.

It fuses signals from **Google Trends, Reddit, and Pinterest** and uses a novel NLP technique called **Recency Attention Reweighting (RAR)** on top of BERT to separate genuine emerging trends from historical references — drastically reducing false trend alerts.

---

## 🧠 How It Works

```
📥 Data Sources
├── Google Trends   → search volume signals (pytrends)
├── Reddit          → community discovery posts (PRAW)
└── Pinterest       → visual fashion signals

         ↓

🔍 RAR-BERT (Core Innovation)
├── Standard BERT processes all text equally
├── RAR modifies attention weights to prioritise present-tense posts
├── Filters out historical references ("remember when this was trendy")
└── Result: genuine emerging trend detection, fewer false alerts

         ↓

🖼️ CLIP (ViT-B/32)
└── Matches text trend signals with visual fashion images

         ↓

📊 DBSCAN Clustering
└── Groups similar micro-trends together

         ↓

🏆 Weighted Trend Score
├── Google Trends weight  → search momentum
├── Reddit weight         → community buzz
└── Pinterest weight      → visual spread

         ↓

🌐 Streamlit Dashboard
└── Visual trend report with scores and timeline
```

---

## 🚀 Key Features

- **6–8 week early detection** — spots trends before they go mainstream
- **RAR-BERT** — novel attention reweighting for temporal trend separation
- **Multi-source fusion** — combines text + search + visual signals
- **Indian fashion focused** — trained and tuned for Indian fashion context
- **Real-time dashboard** — interactive Streamlit UI for trend exploration
- **DBSCAN clustering** — groups related micro-trends automatically

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| NLP Model | BERT (bert-base-uncased) + RAR modification |
| Vision Model | CLIP (ViT-B/32) |
| Clustering | DBSCAN (scikit-learn) |
| Data Sources | Google Trends (pytrends), Reddit (PRAW) |
| Frontend | Streamlit |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |

---

## 📁 Project Structure

```
TrendSense/
│
├── data/
│   ├── scraper_google.py      # Google Trends data collection
│   ├── scraper_reddit.py      # Reddit post collection via PRAW
│   └── preprocessing.py       # Data cleaning and formatting
│
├── models/
│   ├── rar_bert.py            # RAR-BERT implementation (core)
│   ├── clip_matcher.py        # CLIP visual matching
│   └── trend_scorer.py        # Weighted trend score computation
│
├── clustering/
│   └── dbscan_cluster.py      # DBSCAN trend grouping
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/nehalp1130/TrendSense.git
cd TrendSense

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboard/app.py
```

---

## 📦 Requirements

```
torch
transformers
clip
streamlit
pytrends
praw
pandas
numpy
scikit-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔬 Core Innovation — RAR (Recency Attention Reweighting)

Standard BERT treats all text equally. A post saying *"I love how sarees were trending in 2020"* gets the same attention as *"Sarees are blowing up right now!"*

RAR solves this by **modifying BERT's attention formula** to down-weight historical references and up-weight present-tense discovery language:

```python
# Standard BERT attention
attention = softmax(QK^T / sqrt(d_k)) * V

# RAR-BERT attention (simplified)
recency_weight = compute_recency_score(post_timestamp, tense_classifier)
attention = softmax((QK^T / sqrt(d_k)) * recency_weight) * V
```

This makes TrendSense significantly better at distinguishing **genuine emerging trends** from nostalgia posts and historical references.

---

## 📊 Weighted Trend Score Formula

```
TrendScore = (w1 × GoogleTrends_score)
           + (w2 × Reddit_buzz_score)
           + (w3 × Pinterest_visual_score)

Where:
  w1 + w2 + w3 = 1.0
  Weights tuned for Indian fashion context
```

---

## 👥 Team

| Name | Role |
|------|------|
| Nehal P | NLP + RAR-BERT design |
| Team Member 2 | Data collection + scraping |
| Team Member 3 | CLIP integration |
| Team Member 4 | Dashboard + deployment |

---

## 📅 Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Research & design | Jan 2026 | ✅ Done |
| Data pipeline | Feb 2026 | ✅ Done |
| RAR-BERT implementation | Mar 2026 | 🔄 In Progress |
| CLIP integration | Apr 2026 | 🔄 In Progress |
| Dashboard | May 2026 | ⏳ Upcoming |
| Evaluation & testing | Jun 2026 | ⏳ Upcoming |

---

## 🏫 Academic Context

> **Project for:** B.E. Computer Science & Engineering (AI & ML)  
> **Institution:** Mangalore Institute of Technology & Engineering (MITE)  
> **Batch:** 2023–2027  
> **Team size:** 4 members  
> **Started:** January 2026 — Ongoing

---

<p align="center">
  <i>Built with ❤️ at MITE Mangalore</i>
</p>
