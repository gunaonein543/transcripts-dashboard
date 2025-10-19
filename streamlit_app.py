# streamlit_app.py
# Advanced Transcript Dashboard (single-file)
# Features:
# - Data cleaning, parsing, sentiment (VADER), topic extraction (TF-IDF)
# - Past-week focus plus 4-week trend context
# - Daily & weekly extractive summaries (text + charts)
# - Keyword triggers, recommendations, email draft & .ics generator
# - Simple QA/chat using TF-IDF matching
# - Password gate via st.secrets["password"]
#
# Usage:
# - Put your CSV (or upload in UI). Default path expects CSV in repo root.
# - Deploy on Streamlit Cloud with requirements.txt shown next.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import base64
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download
import textwrap
import html
import uuid

# ---- Setup and quick caching ----
st.set_page_config(page_title="Napster Transcripts — Advanced Dashboard", layout="wide")

# Download NLTK VADER lexicon if missing (cached)
@st.cache_data(show_spinner=False)
def prepare_nltk():
    try:
        nltk_download('vader_lexicon')
    except Exception:
        pass
prepare_nltk()

sia = SentimentIntensityAnalyzer()

# ---- Password gate (uses Streamlit Secrets) ----
def check_password():
    """Simple password gating using Streamlit secrets. Add `password = 'yourpass'` in Secrets."""
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
    if "password" not in st.secrets:
        # no secret set — open access (helpful for local dev)
        return True
    def submit():
        st.session_state["password_correct"] = (st.session_state.password == st.secrets["password"])
        if st.session_state["password_correct"]:
            st.experimental_rerun()
    st.text_input("Enter dashboard password", type="password", key="password", on_change=submit)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🔒 Incorrect password")
    return False

if not check_password():
    st.stop()

# ---- Helpers: cleaning, sentiment, topics, summarization ----
def clean_text(t):
    if pd.isna(t):
        return ""
    s = str(t)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_sentiment(text):
    if not text:
        return {"compound": 0.0, "label":"neutral"}
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
    # sum tfidf for each term across corpus
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(sums)[::-1][:top_n]
    return list(zip(terms[idx], sums[idx]))

def extractive_summary(text, n_sentences=4):
    # naive extractive summarizer using sentence scoring by word frequency
    if not text or len(text.split()) < 20:
        return text
    sents = re.split(r'(?<=[.!?])\s+', text)
    words = re.findall(r'\w+', text.lower())
    stop = set(["the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an","it","as","by","at","from","or","but","not","will","i"])
    freqs = Counter([w for w in words if w not in stop])
    sent_scores = []
    for i,s in enumerate(sents):
        ws = re.findall(r'\w+', s.lower())
        score = sum(freqs.get(w,0) for w in ws)
        sent_scores.append((i,score,s))
    sent_scores.sort(key=lambda x: x[1], reverse=True)
    chosen = sorted(sent_scores[:n_sentences], key=lambda x: x[0])
    return " ".join
