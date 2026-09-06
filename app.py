"""
app.py  —  ChurnScope India Edition
=====================================
• Currency: Indian Rupee (₹)
• Palette: Saffron-Indigo theme (dark/light)
• Real-time auto-refresh via streamlit-autorefresh
• 7 pages with rich analytics elements

Run:
    streamlit run app.py
    python simulate_stream.py   (second terminal for live feed)
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnScope · India",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ──────────────────────────────────────────────────────────────
# THEME TOKENS
# ──────────────────────────────────────────────────────────────
DARK = dict(
    bg="#080c14", surface="#0f1623", surface2="#161e2e",
    border="#1e2d45", text="#e8eef6", text_muted="#7a8fa6",
    accent="#f97316",    # saffron orange
    accent2="#8b5cf6",   # indigo purple
    accent3="#06b6d4",   # teal
    churn="#ef4444", retain="#10b981",
    warn="#f59e0b",
    grad1="#1a0f3a", grad2="#2d1b4e",
    grad3="#0f2340", grad4="#1a3a5c",
    plot_tmpl="plotly_dark", plot_paper="#0f1623", plot_bg="#080c14",
)
LIGHT = dict(
    bg="#f0f2fc", surface="#ffffff", surface2="#f5f3ff",
    border="#ddd6fe", text="#1e1b4b", text_muted="#6b7280",
    accent="#ea580c",    # deep saffron
    accent2="#7c3aed",   # deep indigo
    accent3="#0891b2",   # teal
    churn="#dc2626", retain="#059669",
    warn="#d97706",
    grad1="#1e1b4b", grad2="#3b0764",
    grad3="#0c2340", grad4="#1e3a5f",
    plot_tmpl="plotly_white", plot_paper="#ffffff", plot_bg="#f5f3ff",
)
T = DARK if st.session_state.dark_mode else LIGHT
PAL = {"Retained": T["retain"], "Churned": T["churn"]}

# ──────────────────────────────────────────────────────────────
# CSS INJECTION
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family:'Inter',sans-serif !important; }}

.stApp {{ background:{T['bg']} !important; color:{T['text']} !important; }}
.block-container {{ padding-top:1.2rem !important; max-width:1440px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background:{T['surface']} !important;
    border-right:1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] * {{ color:{T['text']} !important; }}

/* ── Headings ── */
h1,h2,h3,h4 {{ color:{T['text']} !important; font-weight:700 !important; }}
p,span,label,div {{ color:{T['text']}; }}

/* ── Widgets ── */
.stSelectbox>div>div, .stMultiSelect>div>div {{
    background:{T['surface2']} !important; border-color:{T['border']} !important;
    border-radius:10px !important; color:{T['text']} !important;
}}
.stTextInput>div>div>input, .stNumberInput>div>div>input {{
    background:{T['surface2']} !important; border-color:{T['border']} !important;
    color:{T['text']} !important; border-radius:10px !important;
}}
.stSlider {{ color:{T['text']} !important; }}
.stSlider [data-testid="stThumbValue"] {{ color:{T['accent']} !important; }}

/* Primary button — saffron gradient */
.stButton>button {{
    background:linear-gradient(135deg,{T['accent']},{T['accent2']}) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    font-weight:600 !important; padding:0.55rem 1.4rem !important;
    box-shadow:0 4px 18px rgba(249,115,22,.3) !important;
    transition:all .2s !important;
}}
.stButton>button:hover {{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(249,115,22,.45) !important;
}}

/* DataFrames */
.stDataFrame {{ border-radius:14px !important; overflow:hidden !important;
    border:1px solid {T['border']} !important; }}
.stDataFrame td,.stDataFrame th {{
    background:{T['surface']} !important; color:{T['text']} !important; }}

/* Expander */
.streamlit-expanderHeader {{
    background:{T['surface2']} !important; border:1px solid {T['border']} !important;
    border-radius:10px !important; color:{T['text']} !important; font-weight:600 !important;
}}
/* Metrics */
[data-testid="metric-container"] {{
    background:{T['surface']} !important; border:1px solid {T['border']} !important;
    border-radius:14px !important; padding:14px 18px !important;
}}
[data-testid="metric-container"] label {{ color:{T['text_muted']} !important; }}
[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color:{T['text']} !important; font-weight:700 !important;
}}

/* Radio nav pills */
.stRadio>div {{ gap:4px !important; }}
.stRadio>div>label {{
    background:{T['surface2']} !important; border:1px solid {T['border']} !important;
    border-radius:10px !important; padding:8px 14px !important;
    color:{T['text']} !important; transition:all .15s !important;
}}
.stRadio>div>label:hover {{ border-color:{T['accent']} !important; }}

/* Progress bar */
.stProgress>div>div {{ background:linear-gradient(90deg,{T['accent']},{T['accent2']}) !important; border-radius:8px !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{T['bg']}; }}
::-webkit-scrollbar-thumb {{ background:{T['border']}; border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:{T['accent']}; }}

/* Alerts */
.stAlert {{ border-radius:12px !important; }}
[data-testid="stInfoMessageContent"] {{ color:{T['text']} !important; }}

/* File uploader */
[data-testid="stFileUploader"] {{
    background:{T['surface2']} !important; border:2px dashed {T['border']} !important;
    border-radius:14px !important;
}}

/* ── CUSTOM COMPONENTS ── */

.page-hero {{
    background:linear-gradient(135deg,{T['grad1']} 0%,{T['grad2']} 50%,{T['grad3']} 100%);
    border-radius:18px; padding:30px 36px; margin-bottom:24px;
    border:1px solid {T['border']};
    box-shadow:0 8px 40px rgba(0,0,0,.35);
    position:relative; overflow:hidden;
}}
.page-hero::after {{
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle,rgba(249,115,22,.15),transparent 70%);
}}
.page-hero h1 {{ margin:0 0 8px !important; font-size:2rem !important;
    color:white !important; font-weight:800 !important; }}
.page-hero p  {{ margin:0 !important; color:rgba(255,255,255,.7) !important;
    font-size:.95rem !important; }}

/* Glassmorphism KPI */
.kpi {{
    background:rgba(255,255,255,.04);
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.10);
    border-radius:18px; padding:22px 18px; text-align:center;
    position:relative; overflow:hidden;
    transition:transform .2s,box-shadow .2s;
    cursor:default;
}}
.kpi:hover {{ transform:translateY(-4px); box-shadow:0 16px 48px rgba(0,0,0,.4); }}
.kpi::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,{T['accent']},{T['accent2']},{T['accent3']});
    border-radius:18px 18px 0 0;
}}
.kpi .icon  {{ font-size:1.8rem; margin-bottom:6px; }}
.kpi .val   {{ font-size:2rem; font-weight:800; color:white; margin:4px 0; line-height:1; }}
.kpi .lbl   {{ font-size:.7rem; text-transform:uppercase; letter-spacing:1.4px;
                color:rgba(255,255,255,.6); }}
.kpi .sub   {{ font-size:.78rem; color:rgba(255,255,255,.4); margin-top:5px; }}

/* Solid KPI for light mode */
.kpi-s {{
    background:linear-gradient(135deg,{T['grad1']},{T['grad2']});
    border-radius:18px; padding:22px 18px; text-align:center;
    border:1px solid {T['border']};
    box-shadow:0 4px 20px rgba(0,0,0,.12);
    transition:transform .2s;
    position:relative; overflow:hidden;
}}
.kpi-s:hover {{ transform:translateY(-4px); }}
.kpi-s::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,{T['accent']},{T['accent2']},{T['accent3']});
    border-radius:18px 18px 0 0;
}}
.kpi-s .icon {{ font-size:1.8rem; margin-bottom:6px; }}
.kpi-s .val  {{ font-size:2rem; font-weight:800; color:white; margin:4px 0; }}
.kpi-s .lbl  {{ font-size:.7rem; text-transform:uppercase; letter-spacing:1.4px; color:rgba(255,255,255,.65); }}
.kpi-s .sub  {{ font-size:.78rem; color:rgba(255,255,255,.4); margin-top:5px; }}

/* Insight card */
.insight {{
    background:{T['surface']};
    border:1px solid {T['border']};
    border-left:4px solid {T['accent2']};
    border-radius:12px; padding:16px 20px; margin-bottom:10px;
    transition:border-color .2s;
}}
.insight:hover {{ border-left-color:{T['accent']}; }}
.insight .title {{ font-weight:700; font-size:.95rem; margin-bottom:4px; color:{T['text']}; }}
.insight .body  {{ font-size:.85rem; color:{T['text_muted']}; line-height:1.5; }}

/* Alert critical */
.alert-c {{
    background:linear-gradient(135deg,rgba(239,68,68,.12),rgba(239,68,68,.04));
    border:1px solid rgba(239,68,68,.35); border-left:4px solid {T['churn']};
    border-radius:12px; padding:14px 18px; margin-bottom:10px;
    font-size:.88rem;
}}
.alert-c b {{ color:{T['churn']}; }}
.alert-w {{
    background:linear-gradient(135deg,rgba(245,158,11,.12),rgba(245,158,11,.04));
    border:1px solid rgba(245,158,11,.35); border-left:4px solid {T['warn']};
    border-radius:12px; padding:14px 18px; margin-bottom:10px;
    font-size:.88rem;
}}
.alert-w b {{ color:{T['warn']}; }}

/* Live banner */
@keyframes blink {{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
@keyframes pulse {{
    0%  {{box-shadow:0 0 0 0 rgba(249,115,22,.6)}}
    70% {{box-shadow:0 0 0 10px rgba(249,115,22,0)}}
    100%{{box-shadow:0 0 0 0 rgba(249,115,22,0)}}
}}
.pdot {{
    display:inline-block; width:9px; height:9px;
    background:{T['accent']}; border-radius:50%;
    animation:pulse 1.6s infinite; margin-right:8px; vertical-align:middle;
}}
.live-bar {{
    display:flex; align-items:center; gap:8px;
    background:{T['surface']}; border:1px solid {T['border']};
    border-left:3px solid {T['accent']}; border-radius:12px;
    padding:10px 18px; margin-bottom:18px; font-size:.88rem;
    color:{T['text_muted']};
}}
.live-bar b {{ color:{T['text']}; }}
.live-bar .badge {{
    background:linear-gradient(90deg,{T['accent']},{T['accent2']});
    color:white; border-radius:20px; padding:2px 12px;
    font-size:.72rem; font-weight:700; letter-spacing:.8px; margin-left:auto;
}}

/* Section header */
.sec {{
    display:flex; align-items:center; gap:10px;
    font-size:.8rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:{T['text_muted']};
    margin:24px 0 14px;
}}
.sec::after {{ content:''; flex:1; height:1px; background:{T['border']}; }}

/* Chart wrapper */
.ccard {{
    background:{T['surface']}; border:1px solid {T['border']};
    border-radius:16px; padding:6px; margin-bottom:6px;
}}

/* Tooltip badge */
.badge-green {{ background:rgba(16,185,129,.15); color:{T['retain']};
    border:1px solid rgba(16,185,129,.3); border-radius:20px;
    padding:2px 10px; font-size:.78rem; font-weight:600; }}
.badge-red {{ background:rgba(239,68,68,.15); color:{T['churn']};
    border:1px solid rgba(239,68,68,.3); border-radius:20px;
    padding:2px 10px; font-size:.78rem; font-weight:600; }}
.badge-orange {{ background:rgba(249,115,22,.15); color:{T['accent']};
    border:1px solid rgba(249,115,22,.3); border-radius:20px;
    padding:2px 10px; font-size:.78rem; font-weight:600; }}

/* Empty state */
.empty-state {{
    background:{T['surface']}; border:2px dashed {T['border']};
    border-radius:18px; padding:60px 40px; text-align:center; margin-top:8px;
}}
.empty-state .es-icon {{ font-size:4rem; margin-bottom:16px; }}
.empty-state .es-title {{ font-size:1.1rem; font-weight:700; color:{T['text']}; margin-bottom:8px; }}
.empty-state .es-body  {{ font-size:.88rem; color:{T['text_muted']}; line-height:1.6; }}

/* Footer */
.footer {{
    text-align:center; padding:24px 0 8px;
    color:{T['text_muted']}; font-size:.78rem;
    border-top:1px solid {T['border']}; margin-top:32px;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
LIVE_PATH = "data/live_feed.csv"
NUM_COLS  = ["tenure","monthly_charges","total_charges","num_services","support_calls"]
CAT_COLS  = ["contract_type","payment_method","internet_service"]
FEAT_COLS = NUM_COLS + CAT_COLS
RUPEE     = "₹"

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def kpi(icon, label, value, sub=""):
    cls = "kpi" if st.session_state.dark_mode else "kpi-s"
    return (f'<div class="{cls}">'
            f'<div class="icon">{icon}</div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>')

def sec(icon, title):
    st.markdown(f'<div class="sec">{icon} {title}</div>', unsafe_allow_html=True)

def ccard(fig):
    st.markdown('<div class="ccard">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def insight(icon, title, body):
    st.markdown(f'<div class="insight"><div class="title">{icon} {title}</div>'
                f'<div class="body">{body}</div></div>', unsafe_allow_html=True)

def fmt_r(v): return f"{RUPEE}{v:,.0f}"
def fmt_rp(v): return f"{RUPEE}{v:.2f}"

def chart(fig, title="", xlab="", ylab="", legend=True):
    fig.update_layout(
        template=T["plot_tmpl"],
        paper_bgcolor=T["plot_paper"], plot_bgcolor=T["plot_bg"],
        title=dict(text=title, font=dict(size=14, color=T["text"], family="Inter"),
                   x=0, xanchor="left", pad=dict(l=6)),
        margin=dict(t=46, b=14, l=14, r=14),
        font=dict(color=T["text_muted"], family="Inter", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=T["text"])),
        showlegend=legend,
    )
    if xlab: fig.update_xaxes(title_text=xlab)
    if ylab: fig.update_yaxes(title_text=ylab)
    fig.update_xaxes(gridcolor=T["border"], linecolor=T["border"],
                     tickfont=dict(color=T["text_muted"]))
    fig.update_yaxes(gridcolor=T["border"], linecolor=T["border"],
                     tickfont=dict(color=T["text_muted"]))
    return fig

# ──────────────────────────────────────────────────────────────
# DATA & MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists("data/churn_data.csv"):
        st.error("Run `python train_model.py` first."); st.stop()
    df = pd.read_csv("data/churn_data.csv")
    df["churn_label"] = df["churn"].map({0:"Retained",1:"Churned"})
    df["clv"] = (df["monthly_charges"] * df["tenure"] * 0.85).round(0)
    df["tenure_band"] = pd.cut(df["tenure"],
        bins=[0,12,24,36,48,72],
        labels=["0-12m","13-24m","25-36m","37-48m","49-72m"])
    df["charge_band"] = pd.cut(df["monthly_charges"],
        bins=[0,40,70,100,120],
        labels=["Low <₹40","Mid ₹40-70","High ₹70-100","Premium >₹100"])
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    for p in ("model/churn_model.joblib","model/preprocessor.joblib"):
        if not os.path.exists(p):
            st.error(f"Missing {p}"); st.stop()
    return (joblib.load("model/churn_model.joblib"),
            joblib.load("model/preprocessor.joblib"))

def read_feed():
    if not os.path.exists(LIVE_PATH): return pd.DataFrame()
    try:
        lf = pd.read_csv(LIVE_PATH, parse_dates=["timestamp"])
        lf["status"] = lf["predicted_churn"].map({1:"Churned",0:"Retained"})
        return lf.sort_values("timestamp").reset_index(drop=True)
    except Exception: return pd.DataFrame()

df          = load_data()
model, prep = load_model()
lf          = read_feed()
feed_ok     = not lf.empty

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 0 20px">
      <span style="font-size:2.2rem">🔮</span>
      <div>
        <div style="font-size:1.15rem;font-weight:800;color:{T['text']}">ChurnScope</div>
        <div style="font-size:.72rem;color:{T['text_muted']};letter-spacing:.5px">
          INDIA EDITION &nbsp;·&nbsp; ML DASHBOARD</div>
      </div>
    </div>""", unsafe_allow_html=True)

    dm_lbl = "☀️  Switch to Light" if st.session_state.dark_mode else "🌙  Switch to Dark"
    if st.button(dm_lbl, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    page = st.radio("", [
        "📊  Overview",
        "🔍  Explore Data",
        "📈  Feature Analysis",
        "🤖  Predict Churn",
        "📦  Batch Predict",
        "📉  Model Performance",
        "🔴  Live Monitor",
    ], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**⚙️  Live Settings**")
    auto_on = st.toggle("Auto-refresh", value=True)
    ref_s   = st.select_slider("Interval",
                options=[2,3,5,10,30], value=3,
                format_func=lambda v: f"{v}s")

    # Dataset quick stats
    st.markdown("---")
    st.markdown(f"**📋  Dataset Stats**")
    st.markdown(f"""
    <div style="font-size:.82rem;color:{T['text_muted']};line-height:2">
      Customers &nbsp;<b style="color:{T['text']}">{len(df):,}</b><br>
      Churn Rate &nbsp;<b style="color:{T['churn']}">{df['churn'].mean():.1%}</b><br>
      Avg Tenure &nbsp;<b style="color:{T['text']}">{df['tenure'].mean():.0f} mo</b><br>
      Avg Charge &nbsp;<b style="color:{T['accent']}">₹{df['monthly_charges'].mean():.0f}</b>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Scikit-learn · Plotly · Streamlit")

# ──────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ──────────────────────────────────────────────────────────────
if auto_on:
    st_autorefresh(interval=ref_s * 1000, key="ar")

# ──────────────────────────────────────────────────────────────
# GLOBAL LIVE BANNER
# ──────────────────────────────────────────────────────────────
if feed_ok and auto_on:
    lr  = lf["predicted_churn"].mean()
    lts = lf["timestamp"].max()
    st.markdown(
        f'<div class="live-bar">'
        f'<span class="pdot"></span>'
        f'<b>LIVE STREAM</b> &nbsp;·&nbsp; {len(lf):,} events &nbsp;·&nbsp; '
        f'Churn rate <b>{lr:.1%}</b> &nbsp;·&nbsp; Last update <b>{lts}</b>'
        f'<span class="badge">● ACTIVE</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Sidebar high-risk alerts
    hr = lf[lf["churn_probability"] >= 0.80].tail(3)
    if not hr.empty:
        with st.sidebar:
            st.markdown(f"**🚨  High-Risk Alerts**")
            for _, r in hr.sort_values("timestamp", ascending=False).iterrows():
                st.markdown(
                    f'<div class="alert-c">'
                    f'<b>Customer {int(r.customer_id)}</b><br>'
                    f'{r.contract_type}<br>'
                    f'₹{r.monthly_charges:.0f}/mo · '
                    f'<b>{r.churn_probability:.0%}</b></div>',
                    unsafe_allow_html=True,
                )

page = page.strip()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page.startswith("📊"):
    st.markdown("""<div class="page-hero">
      <h1>📊 Customer Overview</h1>
      <p>End-to-end snapshot of customer health, revenue, and churn risk across segments.</p>
    </div>""", unsafe_allow_html=True)

    total    = len(df); churned = int(df["churn"].sum())
    rate     = churned/total
    avg_mo   = df["monthly_charges"].mean()
    avg_te   = df["tenure"].mean()
    rev_risk = df.loc[df["churn"]==1,"monthly_charges"].sum()
    avg_clv  = df["clv"].mean()
    retained_clv = df.loc[df["churn"]==0,"clv"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1,"👥","Total Customers",   f"{total:,}",          "Training set"),
        (c2,"🔴","Churn Rate",        f"{rate:.1%}",          "⚠️ At risk" if rate>.30 else "✅ Healthy"),
        (c3,"💰","Monthly Rev at Risk",fmt_r(rev_risk),       f"{churned:,} customers"),
        (c4,"📅","Avg Tenure",        f"{avg_te:.0f} months", "Customer lifetime"),
        (c5,"💎","Avg CLV",           fmt_r(avg_clv),         f"Retained: {fmt_r(retained_clv)}"),
    ]
    for col,icon,lbl,val,sub in cards:
        with col: st.markdown(kpi(icon,lbl,val,sub), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto-generated insights
    sec("💡","Key Insights")
    col_ins, col_pie = st.columns([1,1])
    with col_ins:
        mtm_rate = df[df["contract_type"]=="Month-to-month"]["churn"].mean()
        fiber_rate = df[df["internet_service"]=="Fiber optic"]["churn"].mean()
        high_charge_rate = df[df["monthly_charges"]>80]["churn"].mean()
        new_cust_rate = df[df["tenure"]<=12]["churn"].mean()

        insight("📌","Month-to-Month Contracts Are Highest Risk",
                f"{mtm_rate:.1%} of month-to-month customers churn vs "
                f"{df[df['contract_type']=='Two year']['churn'].mean():.1%} on two-year plans. "
                f"Encouraging longer contracts could significantly reduce churn.")
        insight("📡","Fiber Optic Customers Churn More",
                f"Fiber optic internet users churn at {fiber_rate:.1%} — "
                f"likely due to higher bills. Consider loyalty discounts for this segment.")
        insight("💸","High-Charge Customers Are High-Risk",
                f"Customers paying over ₹80/month churn at {high_charge_rate:.1%}. "
                f"Proactive retention offers for premium customers could reduce revenue loss.")
        insight("🆕","New Customers Need Onboarding",
                f"{new_cust_rate:.1%} of customers in their first year churn. "
                f"Stronger onboarding programs and early engagement can improve retention.")

    with col_pie:
        pie = df["churn_label"].value_counts().reset_index()
        pie.columns = ["Status","Count"]
        fig = px.pie(pie,names="Status",values="Count",color="Status",
                     color_discrete_map=PAL,hole=.55)
        fig = chart(fig,"Overall Churn Distribution",legend=False)
        fig.update_traces(textposition="outside",textinfo="percent+label",
                          marker=dict(line=dict(color=T["bg"],width=3)),
                          textfont_color=T["text"])
        fig.add_annotation(text=f"<b>{rate:.1%}</b><br>Churn",
                           x=.5,y=.5,showarrow=False,
                           font=dict(size=18,color=T["text"]))
        ccard(fig)

    # ── Revenue & tenure charts
    sec("💰","Revenue Analysis")
    r1,r2,r3 = st.columns(3)

    with r1:
        cb = df.groupby("charge_band")["churn"].mean().mul(100).round(1).reset_index()
        cb.columns = ["Charge Band","Churn Rate (%)"]
        fig = px.bar(cb,x="Charge Band",y="Churn Rate (%)",
                     color="Churn Rate (%)",color_continuous_scale=["#10b981","#f97316","#ef4444"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig = chart(fig,"Churn Rate by Monthly Charge Band",legend=False)
        fig.update_layout(coloraxis_showscale=False)
        ccard(fig)

    with r2:
        tb = df.groupby(["tenure_band","churn_label"]).size().reset_index(name="Count")
        fig = px.bar(tb,x="tenure_band",y="Count",color="churn_label",
                     barmode="group",color_discrete_map=PAL,
                     labels={"tenure_band":"Tenure","churn_label":""})
        fig = chart(fig,"Churn by Tenure Band")
        ccard(fig)

    with r3:
        pm = df.groupby("payment_method")["churn"].mean().mul(100).round(1).reset_index()
        pm.columns = ["Payment Method","Churn Rate (%)"]
        pm = pm.sort_values("Churn Rate (%)",ascending=True)
        fig = px.bar(pm,x="Churn Rate (%)",y="Payment Method",
                     orientation="h",color="Churn Rate (%)",
                     color_continuous_scale=["#10b981","#f97316","#ef4444"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig = chart(fig,"Churn Rate by Payment Method",legend=False)
        fig.update_layout(coloraxis_showscale=False)
        ccard(fig)

    # ── Support & services
    sec("📞","Support & Service Usage")
    s1,s2 = st.columns(2)
    with s1:
        sc_data = df.groupby("support_calls")["churn"].mean().mul(100).reset_index()
        sc_data.columns = ["Support Calls","Churn Rate (%)"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sc_data["Support Calls"],y=sc_data["Churn Rate (%)"],
            marker=dict(color=sc_data["Churn Rate (%)"],
                        colorscale=[[0,"#10b981"],[.5,"#f97316"],[1,"#ef4444"]]),
            text=[f"{v:.1f}%" for v in sc_data["Churn Rate (%)"]],
            textposition="outside",name="Churn Rate"))
        fig = chart(fig,"Churn Rate (%) vs Number of Support Calls",
                    xlab="Support Calls",ylab="Churn Rate (%)",legend=False)
        ccard(fig)

    with s2:
        ns_data = df.groupby("num_services")["churn"].mean().mul(100).reset_index()
        ns_data.columns = ["Services","Churn Rate (%)"]
        fig = px.line(ns_data,x="Services",y="Churn Rate (%)",
                      markers=True,color_discrete_sequence=[T["accent2"]])
        fig.update_traces(line=dict(width=3),marker=dict(size=10,color=T["accent"]))
        fig.add_hrect(y0=0,y1=ns_data["Churn Rate (%)"].min()+5,
                      fillcolor=T["retain"],opacity=.08,line_width=0)
        fig = chart(fig,"Churn Rate (%) vs Number of Services Subscribed",
                    xlab="Number of Services",ylab="Churn Rate (%)",legend=False)
        ccard(fig)

    # Live vs training comparison
    if feed_ok and len(lf) >= 5:
        sec("📡","Live Stream vs Training Data")
        l1,l2 = st.columns(2)
        with l1:
            comp = pd.DataFrame({"Source":["Training","Live Stream"],
                                 "Churn Rate (%)": [rate*100, lf["predicted_churn"].mean()*100],
                                 "Avg Charge (₹)": [avg_mo, lf["monthly_charges"].mean()]})
            fig = make_subplots(rows=1,cols=2,subplot_titles=["Churn Rate (%)","Avg Monthly Charge (₹)"])
            fig.add_trace(go.Bar(x=comp["Source"],y=comp["Churn Rate (%)"],
                                  marker_color=[T["accent2"],T["churn"]],
                                  text=[f"{v:.1f}%" for v in comp["Churn Rate (%)"]],
                                  textposition="outside",showlegend=False),row=1,col=1)
            fig.add_trace(go.Bar(x=comp["Source"],y=comp["Avg Charge (₹)"],
                                  marker_color=[T["accent"],T["accent3"]],
                                  text=[f"₹{v:.0f}" for v in comp["Avg Charge (₹)"]],
                                  textposition="outside",showlegend=False),row=1,col=2)
            fig = chart(fig,"Training vs Live Comparison",legend=False)
            ccard(fig)
        with l2:
            fig = px.histogram(lf.tail(100),x="churn_probability",nbins=25,
                               color_discrete_sequence=[T["accent2"]],
                               labels={"churn_probability":"Churn Probability"})
            fig.add_vline(x=0.5,line_dash="dash",line_color=T["churn"],
                          annotation_text="Decision Boundary",annotation_font_color=T["churn"])
            fig = chart(fig,"Live Score Distribution (Last 100 Events)",legend=False)
            ccard(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.markdown("""<div class="page-hero">
      <h1>🔍 Explore Data</h1>
      <p>Filter and visualise the customer dataset. Use the controls below to drill down.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("🎛️  Filters", expanded=True):
        f1,f2,f3,f4 = st.columns(4)
        contracts  = f1.multiselect("Contract",df["contract_type"].unique().tolist(),
                                    default=df["contract_type"].unique().tolist())
        internet   = f2.multiselect("Internet", df["internet_service"].unique().tolist(),
                                    default=df["internet_service"].unique().tolist())
        payment    = f3.multiselect("Payment",  df["payment_method"].unique().tolist(),
                                    default=df["payment_method"].unique().tolist())
        status_f   = f4.multiselect("Churn Status",["Retained","Churned"],
                                    default=["Retained","Churned"])
        t1,t2 = st.columns(2)
        tenure_r   = t1.slider("Tenure (months)",1,72,(1,72))
        charge_r   = t2.slider("Monthly Charge (₹)",20,120,(20,120))

    mask = (df["contract_type"].isin(contracts)
            & df["internet_service"].isin(internet)
            & df["payment_method"].isin(payment)
            & df["churn_label"].isin(status_f)
            & df["tenure"].between(*tenure_r)
            & df["monthly_charges"].between(*charge_r))
    dff = df[mask].copy()

    met1,met2,met3,met4 = st.columns(4)
    met1.metric("Customers",f"{len(dff):,}")
    met2.metric("Churn Rate",f"{dff['churn'].mean():.1%}")
    met3.metric("Avg Charge",f"₹{dff['monthly_charges'].mean():.2f}")
    met4.metric("Avg Tenure",f"{dff['tenure'].mean():.0f} mo")

    with st.expander("📋  Summary Statistics"):
        st.dataframe(dff[NUM_COLS+["churn"]].describe().round(2),
                     use_container_width=True)

    st.dataframe(dff.drop(columns=["churn_label","clv","tenure_band","charge_band"])
                     .reset_index(drop=True),
                 height=240, use_container_width=True)

    sec("📊","Distributions")
    x_col = st.selectbox("Feature to visualise",NUM_COLS,index=0)
    h1,h2 = st.columns(2)

    with h1:
        fig = px.histogram(dff,x=x_col,color="churn_label",barmode="overlay",
                           opacity=.72,color_discrete_map=PAL,nbins=40,
                           labels={"churn_label":""},
                           marginal="box")
        fig = chart(fig,x_col.replace("_"," ").title()+" Distribution by Churn")
        ccard(fig)

    with h2:
        y_col = st.selectbox("Y-axis (scatter)",NUM_COLS,index=1)
        fig = px.scatter(dff.sample(min(1000,len(dff)),random_state=42),
                         x=x_col,y=y_col,color="churn_label",
                         color_discrete_map=PAL,opacity=.6,
                         trendline="ols",trendline_color_override=T["accent"],
                         labels={"churn_label":""})
        fig = chart(fig,f"{x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}")
        ccard(fig)

    sec("📦","Categorical Breakdown")
    b1,b2,b3 = st.columns(3)
    for col,cat in zip([b1,b2,b3],CAT_COLS):
        with col:
            cd = dff.groupby([cat,"churn_label"]).size().reset_index(name="Count")
            fig = px.bar(cd,x=cat,y="Count",color="churn_label",barmode="stack",
                         color_discrete_map=PAL,labels={"churn_label":""},
                         text="Count")
            fig.update_traces(texttemplate="%{text}",textposition="inside")
            fig = chart(fig,cat.replace("_"," ").title())
            ccard(fig)

    sec("📈","Churn Rate by Category")
    for cat in CAT_COLS:
        cr = dff.groupby(cat)["churn"].mean().mul(100).round(1).reset_index()
        cr.columns = [cat,"Churn Rate (%)"]
        fig = px.bar(cr,x=cat,y="Churn Rate (%)",color="Churn Rate (%)",
                     color_continuous_scale=["#10b981","#f97316","#ef4444"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig = chart(fig,f"Churn Rate by {cat.replace('_',' ').title()}",legend=False)
        fig.update_layout(coloraxis_showscale=False)
        ccard(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.markdown("""<div class="page-hero">
      <h1>📈 Feature Analysis</h1>
      <p>Understand variable importance, correlations, and how each feature impacts churn.</p>
    </div>""", unsafe_allow_html=True)

    # Feature importance
    ohe   = list(prep.named_transformers_["cat"].named_steps["encoder"]
                     .get_feature_names_out(CAT_COLS))
    names = NUM_COLS + ohe
    imp   = pd.Series(model.feature_importances_,index=names).sort_values(ascending=False)
    top_imp = imp.head(15).sort_values()

    sec("🏆","Model Feature Importances")
    fi1,fi2 = st.columns([3,2])

    with fi1:
        fig = px.bar(x=top_imp.values,y=top_imp.index,orientation="h",
                     color=top_imp.values,
                     color_continuous_scale=["#8b5cf6","#f97316","#ef4444"],
                     labels={"x":"Importance Score","y":""},
                     text=top_imp.values)
        fig.update_traces(texttemplate="%{text:.4f}",textposition="outside",
                          marker_line_width=0)
        fig = chart(fig,"Top 15 Feature Importances (Random Forest)",legend=False)
        fig.update_layout(coloraxis_showscale=False)
        ccard(fig)

    with fi2:
        # Top-5 importance table with description
        top5 = imp.head(5).reset_index()
        top5.columns = ["Feature","Importance"]
        top5["Rank"] = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        top5["Importance"] = top5["Importance"].map(lambda v: f"{v:.4f}")
        st.markdown(f"**Top 5 Most Important Features**")
        st.table(top5[["Rank","Feature","Importance"]].set_index("Rank"))
        insight("💡","How to Read This",
                "Feature importance tells you how much each variable contributes "
                "to the model's decisions. Higher score = more predictive power. "
                "Features with 0 importance are ignored by the model.")

    # Correlation heatmap
    sec("🔗","Correlation Matrix")
    corr = df[NUM_COLS+["churn"]].corr().round(2)
    fig = px.imshow(corr,text_auto=True,aspect="auto",
                    color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
    fig = chart(fig,"Pearson Correlation Matrix",legend=False)
    ccard(fig)

    sec("🔬","Feature Deep Dive")
    feat_sel = st.selectbox("Select a numeric feature to analyse",NUM_COLS)

    d1,d2,d3 = st.columns(3)
    with d1:
        # Box plot split by churn
        fig = px.box(df,x="churn_label",y=feat_sel,color="churn_label",
                     color_discrete_map=PAL,points="outliers",labels={"churn_label":""})
        fig = chart(fig,f"{feat_sel.replace('_',' ').title()} by Churn Status",legend=False)
        ccard(fig)

    with d2:
        # Violin
        fig = px.violin(df,x="churn_label",y=feat_sel,color="churn_label",
                        color_discrete_map=PAL,box=True,labels={"churn_label":""})
        fig = chart(fig,f"Distribution Shape by Status",legend=False)
        ccard(fig)

    with d3:
        # Avg by contract type
        avg_grp = df.groupby("contract_type")[feat_sel].mean().round(2).reset_index()
        fig = px.bar(avg_grp,x="contract_type",y=feat_sel,color="contract_type",
                     labels={"contract_type":"Contract",feat_sel:feat_sel.replace("_"," ").title()})
        fig = chart(fig,f"Avg {feat_sel.replace('_',' ').title()} by Contract",legend=False)
        ccard(fig)

    # Feature stats table
    stat_r = df.groupby("churn_label")[feat_sel].agg(["mean","median","std","min","max"])
    stat_r.columns = ["Mean","Median","Std Dev","Min","Max"]
    stat_r = stat_r.round(2)
    st.markdown(f"**{feat_sel.replace('_',' ').title()} — Statistics by Churn Status**")
    st.dataframe(stat_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════
elif page.startswith("🤖"):
    st.markdown("""<div class="page-hero">
      <h1>🤖 Predict Customer Churn</h1>
      <p>Enter a customer's details to get a live churn probability with risk breakdown.</p>
    </div>""", unsafe_allow_html=True)

    col_form,col_res = st.columns([1,1])

    with col_form:
        sec("👤","Customer Attributes")
        tenure   = st.slider("Tenure (months)",1,72,12,
                              help="How long has the customer been subscribed?")
        mo_ch    = st.slider("Monthly Charge (₹)",20,120,70,
                              help="Current monthly billing amount")
        tot_ch   = st.number_input("Total Charges (₹)",min_value=0.0,
                                    value=float(mo_ch*tenure),step=100.0)
        num_svc  = st.slider("Number of Services",1,7,3,
                              help="How many services does the customer use?")
        sup_c    = st.slider("Support Calls (last year)",0,10,2,
                              help="High support calls often indicate dissatisfaction")
        st.markdown("---")
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month","One year","Two year"])
        payment  = st.selectbox("Payment Method",
                                 ["Electronic check","Mailed check",
                                  "Bank transfer","Credit card"])
        internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        btn = st.button("🔮  Predict Churn Probability",
                        type="primary", use_container_width=True)

    with col_res:
        sec("📊","Prediction Result")
        if btn:
            inp  = pd.DataFrame([{"tenure":tenure,"monthly_charges":mo_ch,
                "total_charges":tot_ch,"num_services":num_svc,
                "support_calls":sup_c,"contract_type":contract,
                "payment_method":payment,"internet_service":internet}])
            prob  = float(model.predict_proba(prep.transform(inp))[0,1])
            clr   = T["churn"] if prob>=.5 else T["retain"]
            label = "Likely to Churn" if prob>=.5 else "Likely to Retain"
            icon  = "🔴" if prob>=.5 else "🟢"
            risk_lvl = ("🔴 Critical" if prob>=.75 else
                        "🟠 High" if prob>=.5 else
                        "🟡 Moderate" if prob>=.3 else "🟢 Low")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob*100,
                number={"suffix":"%","font":{"size":44,"color":clr}},
                delta={"reference":50,"valueformat":".1f",
                       "increasing":{"color":T["churn"]},
                       "decreasing":{"color":T["retain"]}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":T["text_muted"],
                            "ticksuffix":"%"},
                    "bar":{"color":clr,"thickness":.28},
                    "bgcolor":T["surface2"],
                    "bordercolor":T["border"],
                    "steps":[
                        {"range":[0,30], "color":"rgba(16,185,129,.1)"},
                        {"range":[30,50],"color":"rgba(245,158,11,.08)"},
                        {"range":[50,75],"color":"rgba(249,115,22,.1)"},
                        {"range":[75,100],"color":"rgba(239,68,68,.15)"},
                    ],
                    "threshold":{"line":{"color":T["text"],"width":3},
                                 "thickness":.8,"value":50},
                },
                title={"text":f"Churn Probability<br><span style='font-size:.8rem'>{risk_lvl}</span>",
                       "font":{"size":15,"color":T["text_muted"]}},
            ))
            fig.update_layout(height=290,margin=dict(t=40,b=0,l=20,r=20),
                paper_bgcolor=T["plot_paper"],font=dict(family="Inter",color=T["text"]))
            st.plotly_chart(fig,use_container_width=True)

            st.markdown(f"### {icon}  {label}")
            st.markdown(f"""
            <div style="background:{T['surface2']};border-radius:10px;
                        height:12px;overflow:hidden;margin:4px 0 20px;">
              <div style="width:{prob*100:.1f}%;height:100%;
                          background:linear-gradient(90deg,{T['retain'] if prob<.5 else T['accent']},{clr});
                          border-radius:10px;transition:width .5s;"></div>
            </div>""", unsafe_allow_html=True)

            # Risk radar chart
            factors = ["Contract Risk","Charge Level","New Customer","Support Burden","Low Engagement"]
            values  = [
                1.0 if contract=="Month-to-month" else (.5 if contract=="One year" else 0.1),
                (mo_ch-20)/100,
                max(0,(12-tenure)/12),
                sup_c/10,
                max(0,(4-num_svc)/4),
            ]
            fig_r = go.Figure(go.Scatterpolar(
                r=values+[values[0]], theta=factors+[factors[0]],
                fill="toself", name="Risk Profile",
                fillcolor=f"rgba(239,68,68,.15)",
                line=dict(color=T["churn"],width=2),
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True,range=[0,1],
                                           tickfont=dict(color=T["text_muted"],size=9),
                                           gridcolor=T["border"]),
                           angularaxis=dict(tickfont=dict(color=T["text"],size=10)),
                           bgcolor=T["surface2"]),
                paper_bgcolor=T["plot_paper"],
                margin=dict(t=20,b=20,l=20,r=20),
                height=230,showlegend=False,
                font=dict(family="Inter",color=T["text"]),
            )
            st.plotly_chart(fig_r,use_container_width=True)

            # Risk factor table
            risks = [
                ("📋 Contract Type",  contract,       "⚠️ High" if contract=="Month-to-month" else "✅ Low"),
                ("💰 Monthly Charge", f"₹{mo_ch:.0f}","⚠️ High" if mo_ch>80 else "✅ Low"),
                ("📅 Tenure",         f"{tenure} mo", "✅ Low" if tenure>24 else "⚠️ High"),
                ("📞 Support Calls",  str(sup_c),     "⚠️ High" if sup_c>6 else "✅ Low"),
                ("🛰 Services",        str(num_svc),   "✅ Low" if num_svc>4 else "⚠️ Moderate"),
                ("💳 Payment Method", payment,         "⚠️ Risk" if payment=="Electronic check" else "✅ Low"),
            ]
            rdf = pd.DataFrame(risks,columns=["Factor","Value","Signal"])
            st.table(rdf.set_index("Factor"))

            if feed_ok and len(lf) >= 10:
                pct = (lf["churn_probability"] < prob).mean()
                st.metric("Riskier than",f"{pct:.0%} of live customers")
        else:
            st.markdown("""<div class="empty-state">
              <div class="es-icon">🔮</div>
              <div class="es-title">Ready to Predict</div>
              <div class="es-body">Fill in the customer attributes on the left<br>
              and click <b>Predict</b> to see the churn probability,<br>
              risk radar, and factor breakdown.</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════
elif page.startswith("📦"):
    st.markdown("""<div class="page-hero">
      <h1>📦 Batch Predict</h1>
      <p>Upload a CSV of customers to score them all at once with churn probability.</p>
    </div>""", unsafe_allow_html=True)

    sec("📥","Upload Customer Data")
    col_up, col_tmpl = st.columns([2,1])

    with col_tmpl:
        st.markdown("**Required CSV columns:**")
        template_cols = {c:"" for c in ["tenure","monthly_charges","total_charges",
                                          "num_services","support_calls",
                                          "contract_type","payment_method","internet_service"]}
        tmpl = pd.DataFrame([
            [12,75.0,900.0,3,2,"Month-to-month","Electronic check","Fiber optic"],
            [48,50.0,2400.0,5,0,"Two year","Bank transfer","DSL"],
        ], columns=list(template_cols.keys()))
        st.dataframe(tmpl, use_container_width=True, height=100)
        csv_tmpl = tmpl.to_csv(index=False).encode()
        st.download_button("⬇️  Download Template CSV",csv_tmpl,
                           "churn_template.csv","text/csv",
                           use_container_width=True)

    with col_up:
        uploaded = st.file_uploader("Upload your customer CSV",type=["csv"])

    if uploaded:
        try:
            up_df = pd.read_csv(uploaded)
            st.success(f"✅  Loaded {len(up_df):,} customers")

            X_up = up_df[FEAT_COLS] if all(c in up_df.columns for c in FEAT_COLS) \
                   else None
            if X_up is None:
                missing = [c for c in FEAT_COLS if c not in up_df.columns]
                st.error(f"Missing columns: {missing}")
            else:
                probs = model.predict_proba(prep.transform(X_up))[:,1]
                up_df["churn_probability"] = probs.round(4)
                up_df["predicted_churn"]   = (probs >= 0.5).astype(int)
                up_df["risk_level"] = pd.cut(probs,
                    bins=[0,.3,.5,.75,1.],
                    labels=["🟢 Low","🟡 Moderate","🟠 High","🔴 Critical"])
                up_df["status"] = up_df["predicted_churn"].map({1:"Churned",0:"Retained"})

                sec("📊","Batch Results Summary")
                b1,b2,b3,b4 = st.columns(4)
                b1.metric("Total Customers",f"{len(up_df):,}")
                b2.metric("Predicted Churned",f"{up_df['predicted_churn'].sum():,}")
                b3.metric("Churn Rate",f"{up_df['predicted_churn'].mean():.1%}")
                b4.metric("Avg Churn Prob",f"{probs.mean():.1%}")

                sec("📈","Risk Distribution")
                rc1,rc2 = st.columns(2)
                with rc1:
                    rc = up_df["risk_level"].value_counts().reset_index()
                    rc.columns = ["Risk Level","Count"]
                    fig = px.pie(rc,names="Risk Level",values="Count",hole=.45,
                                 color_discrete_sequence=[T["retain"],T["warn"],T["accent"],T["churn"]])
                    fig = chart(fig,"Customer Risk Distribution",legend=True)
                    ccard(fig)
                with rc2:
                    fig = px.histogram(up_df,x="churn_probability",
                                       color="status",barmode="overlay",
                                       color_discrete_map={"Churned":T["churn"],"Retained":T["retain"]},
                                       nbins=30,opacity=.75,labels={"status":""})
                    fig.add_vline(x=0.5,line_dash="dash",line_color=T["text_muted"])
                    fig = chart(fig,"Churn Probability Histogram")
                    ccard(fig)

                # Full scored table
                sec("📋","Scored Customer Table")
                st.dataframe(up_df.sort_values("churn_probability",ascending=False)
                               .reset_index(drop=True),
                             height=360, use_container_width=True)

                # Download
                out_csv = up_df.to_csv(index=False).encode()
                st.download_button("⬇️  Download Scored CSV", out_csv,
                                   "scored_customers.csv","text/csv",
                                   use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""<div class="empty-state">
          <div class="es-icon">📦</div>
          <div class="es-title">No File Uploaded Yet</div>
          <div class="es-body">Download the template CSV, fill it with your customers,<br>
          then upload it here to get churn predictions for everyone at once.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page.startswith("📉"):
    st.markdown("""<div class="page-hero">
      <h1>📉 Model Performance</h1>
      <p>Evaluate the Random Forest classifier — adjust the threshold and see how metrics change.</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def get_preds(_m,_p,_df):
        X,y = _df[FEAT_COLS],_df["churn"]
        _,Xt,_,yt = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        Xp = _p.transform(Xt)
        return yt.values, _m.predict_proba(Xp)[:,1]

    yt, yp = get_preds(model,prep,df)

    thresh = st.slider("🎚️  Decision Threshold",0.10,0.90,0.50,0.01,
                        help="Adjust to trade off precision vs recall")
    yhat = (yp >= thresh).astype(int)

    acc  = accuracy_score(yt,yhat)
    auc_s= roc_auc_score(yt,yp)
    f1   = f1_score(yt,yhat)
    prec = precision_score(yt,yhat)
    rec  = recall_score(yt,yhat)

    m1,m2,m3,m4,m5 = st.columns(5)
    for col,lbl,val,sub in [
        (m1,"Accuracy",    f"{acc:.3f}",   f"Threshold {thresh:.2f}"),
        (m2,"ROC-AUC",     f"{auc_s:.3f}", "Threshold-independent"),
        (m3,"F1 Score",    f"{f1:.3f}",    "Harmonic mean P+R"),
        (m4,"Precision",   f"{prec:.3f}",  "Of predicted churns"),
        (m5,"Recall",      f"{rec:.3f}",   "Of actual churns"),
    ]:
        with col: st.markdown(kpi("📊",lbl,val,sub),unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec("📈","Performance Curves")
    r1,r2 = st.columns(2)

    with r1:
        fpr,tpr,_ = roc_curve(yt,yp)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",fill="tozeroy",
            fillcolor=f"rgba(139,92,246,.12)",
            name=f"ROC  AUC={auc_s:.3f}",
            line=dict(color=T["accent2"],width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(color=T["border"],dash="dash"),showlegend=False))
        # Mark current threshold operating point
        from sklearn.metrics import roc_curve as rc2
        fpr_t = np.mean(yt[yhat==0]==1) if yhat.sum()<len(yhat) else 0
        tpr_t = rec
        fig.add_trace(go.Scatter(x=[fpr_t],y=[tpr_t],mode="markers",
            marker=dict(color=T["accent"],size=12,symbol="diamond"),
            name=f"Threshold {thresh:.2f}"))
        fig = chart(fig,"ROC Curve",xlab="False Positive Rate",ylab="True Positive Rate")
        ccard(fig)

    with r2:
        pr_prec,pr_rec,_ = precision_recall_curve(yt,yp)
        pr_auc = auc(pr_rec,pr_prec)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr_rec,y=pr_prec,mode="lines",fill="tozeroy",
            fillcolor=f"rgba(249,115,22,.1)",
            name=f"PR  AUC={pr_auc:.3f}",
            line=dict(color=T["accent"],width=2.5)))
        fig.add_trace(go.Scatter(x=[rec],y=[prec],mode="markers",
            marker=dict(color=T["accent2"],size=12,symbol="diamond"),
            name=f"Threshold {thresh:.2f}"))
        fig = chart(fig,"Precision-Recall Curve",xlab="Recall",ylab="Precision")
        ccard(fig)

    sec("🔢","Confusion Matrix & Distributions")
    c1,c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(yt,yhat)
        tn,fp,fn,tp = cm.ravel()
        fig = px.imshow(cm,text_auto=True,
            x=["Pred: Retained","Pred: Churned"],
            y=["True: Retained","True: Churned"],
            color_continuous_scale="Purples",aspect="auto",
            zmin=0)
        fig = chart(fig,f"Confusion Matrix (threshold={thresh:.2f})",legend=False)
        ccard(fig)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                    gap:10px;margin-top:8px;text-align:center;font-size:.85rem">
          <div><b style="color:{T['retain']}">{tn}</b><br><span style="color:{T['text_muted']}">True Neg</span></div>
          <div><b style="color:{T['churn']}">{fp}</b><br><span style="color:{T['text_muted']}">False Pos</span></div>
          <div><b style="color:{T['warn']}">{fn}</b><br><span style="color:{T['text_muted']}">False Neg</span></div>
          <div><b style="color:{T['retain']}">{tp}</b><br><span style="color:{T['text_muted']}">True Pos</span></div>
        </div>""",unsafe_allow_html=True)

    with c2:
        pdf = pd.DataFrame({"Prob":yp,"Actual":["Churned" if y else "Retained" for y in yt]})
        fig = px.histogram(pdf,x="Prob",color="Actual",barmode="overlay",opacity=.75,
                           color_discrete_map=PAL,nbins=50,labels={"Actual":""})
        fig.add_vline(x=thresh,line_dash="dash",line_color=T["accent"],
                      annotation_text=f"Threshold {thresh:.2f}",
                      annotation_font_color=T["accent"])
        fig = chart(fig,"Score Distribution by Actual Class")
        ccard(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════
elif page.startswith("🔴"):
    lh,lb = st.columns([5,1])
    with lh:
        st.markdown("""<div class="page-hero">
          <h1>🔴 Live Customer Monitor</h1>
          <p>Real-time churn predictions streamed from the model — auto-refreshes every few seconds.</p>
        </div>""", unsafe_allow_html=True)
    with lb:
        st.markdown("<br><br>",unsafe_allow_html=True)
        if auto_on: st.success(f"⚡ LIVE · {ref_s}s")
        else: st.warning("⏸ PAUSED")

    if not feed_ok:
        st.markdown("""<div class="empty-state">
          <div class="es-icon">📡</div>
          <div class="es-title">No Live Stream Detected</div>
          <div class="es-body">Open a second terminal and run:<br>
          <code style="font-size:1rem">python simulate_stream.py</code><br><br>
          Events will appear here automatically within a few seconds.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # KPIs
    total_n  = len(lf); churn_n  = int(lf["predicted_churn"].sum())
    ret_n    = total_n - churn_n; lr = lf["predicted_churn"].mean()
    ap       = lf["churn_probability"].mean()
    hi_risk  = int((lf["churn_probability"]>=0.75).sum())
    rev_live = lf.loc[lf["predicted_churn"]==1,"monthly_charges"].sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    for col,ic,lb2,val,sub in [
        (k1,"📡","Events Streamed",  f"{total_n:,}",  ""),
        (k2,"🔴","Predicted Churn",  f"{churn_n:,}",   ""),
        (k3,"🟢","Predicted Retain", f"{ret_n:,}",     ""),
        (k4,"📊","Live Churn Rate",  f"{lr:.1%}",      ""),
        (k5,"⚠️","High-Risk (≥75%)",f"{hi_risk:,}",   ""),
        (k6,"💰","Rev at Risk/mo",   f"₹{rev_live:,.0f}",""),
    ]:
        with col: st.markdown(kpi(ic,lb2,val,sub),unsafe_allow_html=True)

    st.caption(f"Last event received: **{lf['timestamp'].max()}**")
    st.markdown("<br>",unsafe_allow_html=True)

    # High-risk alerts
    hr = lf[lf["churn_probability"]>=0.80].tail(4)
    if not hr.empty:
        sec("🚨","Critical Alerts")
        acols = st.columns(min(4,len(hr)))
        for i,(_,r) in enumerate(hr.sort_values("timestamp",ascending=False).iterrows()):
            with acols[i%4]:
                st.markdown(
                    f'<div class="alert-c">'
                    f'<b>Customer {int(r.customer_id)}</b><br>'
                    f'{r.contract_type}<br>'
                    f'₹{r.monthly_charges:.0f}/mo · {int(r.support_calls)} calls<br>'
                    f'<b>{r.churn_probability:.0%} churn</b></div>',
                    unsafe_allow_html=True)

    sec("📈","Real-Time Charts")
    c1,c2 = st.columns(2)

    with c1:
        lf_s = lf.copy()
        lf_s["rolling"] = lf_s["predicted_churn"].expanding().mean()*100
        lf_s["n"] = range(1,len(lf_s)+1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lf_s["n"],y=lf_s["rolling"],
            mode="lines",fill="tozeroy",fillcolor=f"rgba(239,68,68,.1)",
            line=dict(color=T["churn"],width=2.5),name="Churn %"))
        fig.add_hline(y=50,line_dash="dash",line_color=T["border"],
                      annotation_text="50%",annotation_font_color=T["text_muted"])
        fig = chart(fig,"Rolling Churn Rate (%)",
                    xlab="Customer #",ylab="Churn Rate (%)",legend=False)
        ccard(fig)

    with c2:
        last60 = lf.tail(60).reset_index(drop=True)
        last60["idx"] = range(1,len(last60)+1)
        fig = px.scatter(last60,x="idx",y="churn_probability",
                         color="status",
                         color_discrete_map={"Churned":T["churn"],"Retained":T["retain"]},
                         labels={"idx":"Event #","churn_probability":"Churn Prob","status":""})
        fig.add_hline(y=0.5,line_dash="dash",line_color=T["border"])
        fig.update_traces(marker_size=10,marker_line_width=0)
        fig = chart(fig,"Churn Probability — Last 60 Events")
        ccard(fig)

    c3,c4,c5 = st.columns(3)
    with c3:
        ct = lf.groupby(["contract_type","status"]).size().reset_index(name="Count")
        fig = px.bar(ct,x="contract_type",y="Count",color="status",
                     barmode="group",
                     color_discrete_map={"Churned":T["churn"],"Retained":T["retain"]},
                     labels={"contract_type":"","status":""})
        fig = chart(fig,"Live Churn by Contract")
        ccard(fig)

    with c4:
        pm2 = lf.groupby("payment_method")["predicted_churn"].mean().mul(100).round(1).reset_index()
        pm2.columns = ["Payment","Churn Rate (%)"]
        fig = px.bar(pm2,x="Churn Rate (%)",y="Payment",orientation="h",
                     color="Churn Rate (%)",
                     color_continuous_scale=["#10b981","#f97316","#ef4444"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig = chart(fig,"Live Churn by Payment Method",legend=False)
        fig.update_layout(coloraxis_showscale=False)
        ccard(fig)

    with c5:
        fig = px.histogram(lf,x="churn_probability",
                           color="status",barmode="overlay",opacity=.75,
                           color_discrete_map={"Churned":T["churn"],"Retained":T["retain"]},
                           nbins=25,labels={"churn_probability":"Prob","status":""})
        fig.add_vline(x=0.5,line_dash="dash",line_color=T["text_muted"])
        fig = chart(fig,"Score Distribution")
        ccard(fig)

    # Live feed table
    sec("📋","Live Event Feed")
    n_show = st.selectbox("Show last N events",[25,50,100,250],index=1)
    disp = (lf[["timestamp","customer_id","contract_type","internet_service",
                 "monthly_charges","support_calls","churn_probability","status"]]
              .sort_values("timestamp",ascending=False)
              .head(n_show).reset_index(drop=True))
    disp["monthly_charges"] = disp["monthly_charges"].map(lambda v: f"₹{v:.2f}")
    disp["churn_probability"] = disp["churn_probability"].map(lambda v: f"{v:.2%}")

    def row_color(row):
        bg = "background-color:rgba(239,68,68,.12)" if row["status"]=="Churned" \
             else "background-color:rgba(16,185,129,.08)"
        return [bg]*len(row)

    st.dataframe(disp.style.apply(row_color,axis=1),
                 height=400,use_container_width=True)

    # Throughput
    if len(lf) >= 2:
        el = (lf["timestamp"].max()-lf["timestamp"].min()).total_seconds()
        if el > 0:
            sc1,sc2,_ = st.columns([1,1,3])
            sc1.metric("Stream throughput",f"{len(lf)/el*60:.1f} ev/min")
            sc2.metric("Avg prob (last 10)",
                       f"{lf.tail(10)['churn_probability'].mean():.1%}")

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  🔮 <b>ChurnScope India</b> &nbsp;·&nbsp;
  Random Forest · Scikit-learn · Streamlit · Plotly &nbsp;·&nbsp;
  Currency: Indian Rupee (₹)
</div>""", unsafe_allow_html=True)
