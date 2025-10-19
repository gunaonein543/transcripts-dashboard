# streamlit_app.py
# Napster Advanced Transcript Dashboard — Fully Fixed Version
# Author: AI Engineer (ChatGPT)
# Last Updated: 2025-10-19

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import base64
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download
import textwrap
import html
import uuid

# ------------------ SETUP ------------------
st.set_page_config(page_title="Napster Transcripts — Advanced Weekly Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def prepare_nltk():
    try:
        nltk_download('vader_lexicon')
    except Exception:
        pass

prepare_nltk()
sia = SentimentIntensityAnalyzer()

# ------------------ PASSWORD GATE ------------------
def check_password():
    """Simple password gating using Streamlit secrets"""
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
    if "password" not in st.secrets:
        # no secret set, skip password for local testing
        return True

    def submit():
        st.session_state["password_correct"] = (st.session_state.password == st.secrets["password"])
        if st.session_state["password_correct"]:
            st.rerun()

    st.text_input("Enter dashboard password", type="password", key="password", on_change=submit)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🔒 Incorrect password")
    return False

if not check_password():
    st.stop()

# ------------------ HELPER FUNCTIONS ------------------
def clean_text(t):
    if pd.isna(t):
        return ""
    s = str(t)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_sentiment(text):
    if not text:
        return {"compound": 0.0, "label": "neutral"}
    v = sia.polarity_scores(text)
    comp = v["compound"]
    if comp >= 0.05:
        lab = "positive"
    elif comp <= -0.05:
        lab = "negative"
    else:
        lab = "neutral"
    return {"compound": comp, "label": lab}

def top_keywords(corpus, top_n=10):
    vect = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vect.fit_transform(corpus)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(sums)[::-1][:top_n]
    return list(zip(terms[idx], sums[idx]))

def extractive_summary(text, n_sentences=4):
    if not text or len(text.split()) < 20:
        return text
    sents = re.split(r'(?<=[.!?])\s+', text)
    words = re.findall(r'\w+', text.lower())
    stop = set(["the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an","it","as","by","at","from","or","but","not","will","i"])
    freqs = Counter([w for w in words if w not in stop])
    sent_scores = []
    for i, s in enumerate(sents):
        ws = re.findall(r'\w+', s.lower())
        score = sum(freqs.get(w, 0) for w in ws)
        sent_scores.append((i, score, s))
    sent_scores.sort(key=lambda x: x[1], reverse=True)
    chosen = sorted(sent_scores[:n_sentences], key=lambda x: x[0])
    return " ".join([s for (_, _, s) in chosen])

# ------------------ LOAD DATA ------------------
st.title("Napster Transcripts — Advanced Weekly Dashboard")

with st.sidebar.expander("Data settings", expanded=True):
    uploaded = st.file_uploader("Upload transcripts CSV (optional)", type=["csv"])
    default_path = "all_transcripts (3).csv"
    use_default = st.checkbox(f"Use default file: {default_path}", value=True)
    st.markdown("Upload your latest transcript export or keep default.")

@st.cache_data
def load_df(uploaded_file, use_default_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(default_path)
        except Exception:
            return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    # Rename possible columns
    if 'content' not in df.columns:
        for c in df.columns:
            if any(x in c.lower() for x in ['trans', 'text', 'content']):
                df = df.rename(columns={c: 'content'})
                break

    if 'date' not in df.columns:
        for c in df.columns:
            if 'date' in c.lower():
                df = df.rename(columns={c: 'date'})
                break

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['participant'] = df.get('participant', df.get('speaker', 'Unknown'))
    df['content'] = df['content'].fillna('').apply(clean_text)

    # ✅ Fixed block: safely handle missing duration column
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
    else:
        df['duration'] = 0

    # Priority (if missing)
    df['priority'] = df.get('priority', '')
    return df

df = load_df(uploaded, use_default)

if df.empty:
    st.warning("No transcript data found. Please upload or include the default CSV file.")
    st.stop()

# Compute sentiment
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    df['sentiment_compound'] = df['content'].apply(lambda t: compute_sentiment(t)['compound'])
    df['sentiment'] = df['content'].apply(lambda t: compute_sentiment(t)['label'])
else:
    if 'sentiment_compound' not in df.columns:
        df['sentiment_compound'] = df['content'].apply(lambda t: compute_sentiment(t)['compound'])

# Topic fallback
if 'topics' not in df.columns:
    def top_row_keywords(text, n=4):
        words = re.findall(r'\w+', text.lower())
        stop = set(["the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an","it","as","by","at","from","or","but","not","will","i"])
        words = [w for w in words if w not in stop and len(w) > 2]
        c = Counter(words)
        return [w for w, _ in c.most_common(n)]
    df['topics'] = df['content'].apply(lambda t: top_row_keywords(t, n=4))

# ------------------ FILTERS ------------------
df['date_only'] = df['date'].dt.date
today = pd.to_datetime(datetime.utcnow()).date()
one_week_ago = today - timedelta(days=7)
four_weeks_ago = today - timedelta(days=28)

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date range", (four_weeks_ago, today))
participants = ['All'] + sorted(df['participant'].dropna().unique().tolist())
sel_participant = st.sidebar.selectbox("Participant", participants)
sentiments = ['All'] + sorted(df['sentiment'].unique().tolist())
sel_sent = st.sidebar.selectbox("Sentiment", sentiments)
topic_filter = st.sidebar.text_input("Topic filter (keyword)")
priority_filter = st.sidebar.selectbox("Priority", ['All'] + sorted(df['priority'].dropna().unique().tolist()))

# Apply filters
f = df.copy()
start_date, end_date = date_range
f = f[(f['date_only'] >= start_date) & (f['date_only'] <= end_date)]
if sel_participant != 'All':
    f = f[f['participant'] == sel_participant]
if sel_sent != 'All':
    f = f[f['sentiment'] == sel_sent]
if topic_filter:
    f = f[f['topics'].apply(lambda t: any(topic_filter.lower() in w.lower() for w in (t if isinstance(t, list) else [t])))]
if priority_filter != 'All' and priority_filter:
    f = f[f['priority'] == priority_filter]

# ------------------ KPIs ------------------
k1, k2, k3, k4, k5 = st.columns([1.5, 1, 1, 1, 1])
with k1: st.metric("Transcripts", len(f))
with k2: st.metric("Positive", (f['sentiment'] == 'positive').sum())
with k3: st.metric("Negative", (f['sentiment'] == 'negative').sum())
with k4:
    avg_dur = f['duration'].replace(0, np.nan).dropna().mean()
    st.metric("Avg Duration (min)", f"{avg_dur:.1f}" if not pd.isna(avg_dur) else "N/A")
with k5:
    urgent_hits = f['content'].str.contains(r'\burgent|delay|escalat|critical\b', case=False, na=False).sum()
    st.metric("Urgent Flags", urgent_hits)

st.markdown("---")

# ------------------ MAIN TABS ------------------
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧭 Topics", "💬 Transcripts"])

# -------- Overview --------
with tab1:
    st.header("Overview — Past Week + 4-Week Context")

    trend_df = df[(df['date_only'] >= four_weeks_ago) & (df['date_only'] <= today)]
    trend_daily = trend_df.groupby('date_only').agg(
        count=('id', 'count'),
        avg_comp=('sentiment_compound', 'mean')
    ).reset_index()

    if not trend_daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend_daily['date_only'], y=trend_daily['count'], name='Count', marker_color='lightblue'))
        fig.add_trace(go.Scatter(x=trend_daily['date_only'], y=trend_daily['avg_comp'], name='Avg Sentiment', mode='lines+markers', line=dict(color='darkgreen')))
        fig.update_layout(title="Daily Transcript Count & Sentiment", yaxis_title="Count / Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    week_df = df[(df['date_only'] >= one_week_ago) & (df['date_only'] <= today)]
    bigtext = " ".join(week_df['content'].tolist()[:2000])
    summary_text = extractive_summary(bigtext, n_sentences=5)
    st.subheader("Weekly Summary")
    st.write(summary_text if summary_text else "No content available for this week.")

# -------- Topics --------
with tab2:
    st.header("Top Topics and Keywords")
    corpus = f['content'].tolist()
    kw = top_keywords(corpus or [" "], top_n=30)
    if kw:
        kw_df = pd.DataFrame(kw, columns=['Keyword', 'Score'])
        st.plotly_chart(px.bar(kw_df.head(20), x='Keyword', y='Score', title='Top Keywords'), use_container_width=True)

# -------- Transcripts --------
with tab3:
    st.header("Transcript Details")
    search = st.text_input("Search text")
    view = f if not search else f[f['content'].str.contains(search, case=False, na=False)]
    st.write(f"Showing {len(view)} transcripts")

    for _, row in view.sort_values('date', ascending=False).iterrows():
        with st.expander(f"{row['date']} — {row['participant']} — {row['sentiment']}"):
            st.markdown(f"**Topics:** {row.get('topics')}")
            st.markdown(f"**Priority:** {row.get('priority')}")
            st.markdown(f"**Duration:** {row.get('duration')} min")
            st.write(row['content'])

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly, NLTK, and scikit-learn. 2025 Napster Dashboard")
