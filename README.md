# 🔮 ChurnScope India — Customer Churn Predictor Dashboard

An interactive ML-powered web dashboard that predicts customer churn using a **Random Forest** classifier, built entirely in Python with **Streamlit** + **Plotly**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange)](https://scikit-learn.org)

---

## ✨ Features

| Page | Description |
|------|-------------|
| 📊 **Overview** | KPI cards, insight cards, revenue analysis, support call trends |
| 🔍 **Explore Data** | Filterable table, histograms, scatter, categorical breakdowns |
| 📈 **Feature Analysis** | RF importance, correlation heatmap, feature deep-dive |
| 🤖 **Predict Churn** | Live prediction form with probability gauge + risk radar chart |
| 📦 **Batch Predict** | Upload CSV → score all customers → download results |
| 📉 **Model Performance** | ROC, PR curves, confusion matrix, adjustable threshold slider |
| 🔴 **Live Monitor** | Real-time streaming predictions with auto-refresh |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
This generates `data/churn_data.csv` and saves model files to `model/`:
```bash
python train_model.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

### 6. (Optional) Start the live event stream
Open a **second terminal** and run:
```bash
python simulate_stream.py
```
The 🔴 Live Monitor page will show real-time predictions as new customers arrive.

---

## 🗂 Project Structure

```
ChurnScope/
├── app.py                  # Main Streamlit dashboard (7 pages)
├── train_model.py          # Data generation + model training
├── simulate_stream.py      # Live event stream simulator
├── requirements.txt        # Python dependencies
├── data/
│   └── churn_data.csv      # Auto-generated synthetic dataset
├── model/
│   ├── churn_model.joblib  # Trained Random Forest (git-ignored)
│   └── preprocessor.joblib # Sklearn ColumnTransformer (git-ignored)
└── README.md
```

> **Note:** `model/*.joblib` files are git-ignored (large binaries).  
> Run `python train_model.py` after cloning to regenerate them.

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Pandas / NumPy** | Data manipulation & synthetic dataset generation |
| **Scikit-learn** | Random Forest classifier, preprocessing pipeline |
| **Joblib** | Model serialisation |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts (dark + light themes) |
| **streamlit-autorefresh** | Real-time page auto-refresh |

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — live in ~2 minutes!

> Add a `packages.txt` if you need OS-level dependencies.

---

## 📄 License

MIT © 2026 — feel free to use and adapt.
