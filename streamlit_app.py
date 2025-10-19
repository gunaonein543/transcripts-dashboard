# streamlit_app.py
# Streamlit dashboard to visualize transcripts.
# Reads:
# - /mnt/data/all_transcripts (3).csv  (uploaded CSV)
# - /mnt/data/napster-transcript-analysis-2025-10-19.json (dashboard template)
#
# How to run:
#     pip install streamlit pandas plotly
#     streamlit run /mnt/data/streamlit_app.py

import streamlit as st
# --- Simple password gate using Streamlit secrets ---
def check_password():
    """Returns True if the correct password is entered."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # remove password from memory
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, ask for password
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Wrong password
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("🔒 Incorrect password")
        return False
    else:
        # Correct password
        return True

if not check_password():
    st.stop()  # stop app until correct password entered
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
from collections import Counter
import re

CSV_PATH = "all_transcripts (3).csv"
JSON_PATH = "napster-transcript-analysis-2025-10-19.json"

st.set_page_config(page_title="Transcripts Dashboard", layout="wide")

@st.cache_data
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load JSON template at {path}: {e}")
        return None

@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV at {path}: {e}")
        return pd.DataFrame()

def simple_sentiment(text):
    # Very small rule-based sentiment fallback
    pos_words = {"good","great","well","positive","excited","celebrat","success","improv","growth","on track","received"}
    neg_words = {"bad","concern","delay","delayed","risk","frustrat","issue","critical","urgent","escalat","outage","problem","concerning"}
    t = str(text).lower()
    pos = sum(1 for w in pos_words if w in t)
    neg = sum(1 for w in neg_words if w in t)
    if pos==neg:
        return "neutral" if pos==0 else ("positive" if pos>neg else "negative")
    return "positive" if pos>neg else "negative"

def extract_topics_from_text(text, top_n=5):
    # simple keyword extractor: count words excluding stopwords & numbers
    stopwords = set([
        "the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an",
        "our","have","has","it","as","by","at","from","or","but","not","will"
    ])
    words = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
    words = [w for w in words if w not in stopwords]
    c = Counter(words)
    return [w for w,_ in c.most_common(top_n)]

# Load data
template = load_json(JSON_PATH)
df = load_csv(CSV_PATH)

st.title("Transcripts Dashboard (Streamlit)")
st.markdown("Generated from your uploaded CSV and JSON template. Use the controls to filter and explore.")

# Allow the user to optionally upload a different CSV (from Streamlit UI)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")

# Basic validation and normalization
if df.empty:
    st.warning("No transcript data available to display.")
    st.stop()

# normalize column names
df.columns = [c.strip() for c in df.columns]
# Ensure expected columns exist; if not, try to infer
for col in ["id","date","participant","content","sentiment","priority","topics","duration"]:
    if col not in df.columns:
        # create defaults
        if col=="id" and "index" in df.columns:
            df["id"] = df["index"]
        elif col=="date":
            possible = [c for c in df.columns if "date" in c.lower()]
            if possible:
                df["date"] = pd.to_datetime(df[possible[0]], errors="coerce").dt.date
            else:
                df["date"] = pd.NaT
        elif col=="participant":
            possible = [c for c in df.columns if "participant" in c.lower() or "speaker" in c.lower()]
            df["participant"] = df[possible[0]] if possible else "Unknown"
        elif col=="content":
            possible = [c for c in df.columns if "transcript" in c.lower() or "content" in c.lower() or "text" in c.lower()]
            df["content"] = df[possible[0]] if possible else ""
        elif col=="duration":
            df["duration"] = pd.to_numeric(df.get("duration", pd.Series([None]*len(df))), errors="coerce").fillna(0)
        else:
            df[col] = None

# If sentiment column missing or empty, compute simple sentiment
if df["sentiment"].isnull().all():
    df["sentiment"] = df["content"].apply(simple_sentiment)

# If topics column is missing or empty, extract simple topics
if df["topics"].isnull().all():
    df["topics"] = df["content"].apply(lambda t: extract_topics_from_text(t, top_n=5))

# Convert topics column to lists if stored as strings like "['a','b']"
def normalize_topics(x):
    if isinstance(x, list):
        return x
    s = str(x)
    if s.strip()=="" or s.lower()=="nan":
        return []
    # try to parse comma separated or python list
    s2 = s.strip()
    if s2.startswith("[") and s2.endswith("]"):
        s2 = s2[1:-1]
    parts = re.split(r"[;,|]\s*|\s{2,}", s2)
    parts = [p.strip(" \"'") for p in parts if p.strip()]
    return parts

df["topics"] = df["topics"].apply(normalize_topics)

# Sidebar filters
st.sidebar.header("Filters")
participants = sorted(df["participant"].dropna().unique()[:200])
sel_participants = st.sidebar.multiselect("Participant", participants, default=participants if len(participants)<=10 else participants[:10])
date_min = pd.to_datetime(df["date"].dropna()).min() if pd.to_datetime(df["date"], errors='coerce').notna().any() else None
date_max = pd.to_datetime(df["date"].dropna()).max() if pd.to_datetime(df["date"], errors='coerce').notna().any() else None
if date_min is not None and date_max is not None:
    start_date, end_date = st.sidebar.date_input("Date range", value=(date_min, date_max))
else:
    start_date = None
    end_date = None

sentiments = sorted(df["sentiment"].dropna().unique())
sel_sent = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)
priorities = sorted(df["priority"].dropna().unique())
sel_prio = st.sidebar.multiselect("Priority", priorities, default=priorities)

# Apply filters
f = df.copy()
if sel_participants:
    f = f[f["participant"].isin(sel_participants)]
if start_date is not None and end_date is not None:
    try:
        f_dates = pd.to_datetime(f["date"], errors="coerce").dt.date
        f = f[(f_dates>=start_date) & (f_dates<=end_date)]
    except:
        pass
if sel_sent:
    f = f[f["sentiment"].isin(sel_sent)]
if sel_prio:
    if "priority" in f.columns:
        f = f[f["priority"].isin(sel_prio)]

# Top metrics using template when available
col1, col2, col3, col4 = st.columns(4)
total_transcripts = len(df)
total_filtered = len(f)
avg_duration = df["duration"].replace(0, pd.NA).dropna().mean() if "duration" in df else None
col1.metric("Total transcripts (file)", total_transcripts)
col2.metric("Showing (filtered)", total_filtered)
col3.metric("Average duration (min)", f"{avg_duration:.1f}" if pd.notna(avg_duration) else "N/A")
if template and "generatedAt" in template:
    col4.metric("Template generated", template.get("generatedAt","-"))
else:
    col4.metric("Template", "Not found")

st.markdown("---")

# Left: charts, Right: list
left, right = st.columns((2,1))

with left:
    st.subheader("Sentiment breakdown")
    sent_counts = f["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment","count"]
    fig1 = px.pie(sent_counts, names="sentiment", values="count", hole=0.4, title="Sentiment")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Priority distribution")
    if "priority" in f.columns:
        prio_counts = f["priority"].value_counts().reset_index()
        prio_counts.columns = ["priority","count"]
        fig2 = px.bar(prio_counts, x="priority", y="count", title="Priorities", text="count")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No priority info available in data.")

    st.subheader("Top topics across filtered transcripts")
    # flatten topics
    all_topics = Counter()
    for tlist in f["topics"].dropna():
        if isinstance(tlist, list):
            all_topics.update([t for t in tlist if t])
        else:
            all_topics.update([tlist])
    top_topics = pd.DataFrame(all_topics.most_common(20), columns=["topic","count"])
    if not top_topics.empty:
        fig3 = px.bar(top_topics, x="topic", y="count", title="Top topics", text="count")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No topics found.")

    st.subheader("Transcript durations (minutes)")
    if "duration" in f.columns:
        fig4 = px.histogram(f, x="duration", nbins=20, title="Durations")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No duration data available.")

with right:
    st.subheader("Summary (from template if available)")
    if template and "summary" in template:
        st.json(template["summary"])
    else:
        st.info("No summary in JSON template.")

    st.subheader("Search transcripts")
    q = st.text_input("Search text (content)")
    if q:
        sf = f[f["content"].str.contains(q, case=False, na=False)]
    else:
        sf = f

    st.write(f"Showing {len(sf)} matching transcripts")
    st.dataframe(sf[["id","date","participant","sentiment","priority","topics","duration","content"]].head(200))

st.markdown("---")
st.subheader("Export / Download")
st.markdown("You can download the filtered transcripts as CSV.")

@st.cache_data
def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

csv_bytes = to_csv_bytes(f)
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_transcripts.csv", mime="text/csv")

st.caption("Streamlit app created to mirror your JSON structure and visualize the uploaded CSV.")


