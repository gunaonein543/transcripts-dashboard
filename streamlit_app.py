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
            st.rerun()
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
    return " ".join([s for (_,_,s) in chosen])

# ---- File load & preprocess ----
st.title("Napster Transcripts — Advanced Weekly Dashboard")

with st.sidebar.expander("Data: load/paths", expanded=True):
    st.markdown("Upload CSV or use file from repo.")
    uploaded = st.file_uploader("Upload transcripts CSV (optional)", type=["csv"])
    default_path = "all_transcripts (3).csv"
    use_default = st.checkbox(f"Use repo file: {default_path}", value=True)
    if uploaded is None and not use_default:
        st.info("Please upload CSV or enable repo file option.")
    st.markdown("---")
    st.markdown("Passwords, alerts, and quick actions available in main UI.")

# read dataframe
@st.cache_data
def load_df(uploaded_file, use_default_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(default_path)
        except Exception as e:
            return pd.DataFrame()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure columns
    if 'content' not in df.columns:
        # try to infer
        for c in df.columns:
            if 'trans' in c.lower() or 'text' in c.lower() or 'content' in c.lower():
                df = df.rename(columns={c:'content'})
                break
    # parse date
    if 'date' not in df.columns:
        for c in df.columns:
            if 'date' in c.lower():
                df = df.rename(columns={c:'date'})
                break
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT
    # ensure other cols
    df['id'] = df.get('id', pd.Series([str(uuid.uuid4()) for _ in range(len(df))]))
    df['participant'] = df.get('participant', df.get('speaker', 'Unknown'))
    df['content'] = df['content'].fillna('').apply(clean_text)
    if 'duration' in df.columns:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
else:
    df['duration'] = 0
    # priority fallback
    df['priority'] = df.get('priority', '')
    return df

df = load_df(uploaded, use_default)

if df.empty:
    st.warning("No transcript data available. Upload CSV or add file to repo root named 'all_transcripts (3).csv'.")
    st.stop()

# compute sentiment if not present
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    st.info("Computing sentiment using VADER...")
    df['sentiment_compound'] = df['content'].apply(lambda t: compute_sentiment(t)['compound'])
    df['sentiment'] = df['content'].apply(lambda t: compute_sentiment(t)['label'])
else:
    # ensure compound present
    if 'sentiment_compound' not in df.columns:
        df['sentiment_compound'] = df['content'].apply(lambda t: compute_sentiment(t)['compound'])

# ensure topics column
if 'topics' not in df.columns:
    # simple placeholder: compute top keywords per row (first 4)
    def top_row_keywords(text, n=4):
        words = re.findall(r'\w+', text.lower())
        stop = set(["the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an","it","as","by","at","from","or","but","not","will","i"])
        words = [w for w in words if w not in stop and len(w)>2]
        c = Counter(words)
        return [w for w,_ in c.most_common(n)]
    df['topics'] = df['content'].apply(lambda t: top_row_keywords(t, n=4))

# infer week and day
df['date_only'] = df['date'].dt.date
today = pd.to_datetime(datetime.utcnow()).date()
one_week_ago = today - timedelta(days=7)
four_weeks_ago = today - timedelta(days=28)

# Filter controls
st.sidebar.header("Filters & View")
date_range = st.sidebar.date_input("Date range", value=(four_weeks_ago, today))
participants = ['All'] + sorted(df['participant'].dropna().unique().tolist())
sel_participant = st.sidebar.selectbox("Participant", participants)
sentiments = ['All'] + sorted(df['sentiment'].unique().tolist())
sel_sent = st.sidebar.selectbox("Sentiment", sentiments)
topic_filter = st.sidebar.text_input("Topic contains (keyword)")
priority_filter = st.sidebar.selectbox("Priority", ['All'] + sorted(df['priority'].dropna().unique().tolist()))

# Apply filters
f = df.copy()
start_date, end_date = date_range
f = f[(f['date_only'] >= start_date) & (f['date_only'] <= end_date)]
if sel_participant != 'All':
    f = f[f['participant']==sel_participant]
if sel_sent != 'All':
    f = f[f['sentiment']==sel_sent]
if topic_filter:
    f = f[f['topics'].apply(lambda t: any(topic_filter.lower() in w.lower() for w in (t if isinstance(t,list) else [t]))) ]
if priority_filter != 'All' and priority_filter:
    f = f[f['priority']==priority_filter]

# ---------- Summary & KPI row ----------
k1, k2, k3, k4, k5 = st.columns([1.5,1,1,1,1])
with k1:
    st.metric("Transcripts (shown)", len(f))
with k2:
    pos = (f['sentiment']=='positive').sum()
    st.metric("Positive", pos)
with k3:
    neg = (f['sentiment']=='negative').sum()
    st.metric("Negative", neg)
with k4:
    avg_dur = f['duration'].replace(0, np.nan).dropna().mean()
    st.metric("Avg Duration (min)", f"{avg_dur:.1f}" if not pd.isna(avg_dur) else "N/A")
with k5:
    urgent_hits = f['content'].str.contains(r'\burgent\b|\bescalat\b|\bimmediat\b|\bcritical\b|\bdelay\b|\bdelayed\b', case=False, na=False).sum()
    st.metric("Urgent flags", urgent_hits)

st.markdown("---")

# ---------- Main layout: Overview + Details tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🧭 Topics & Trends", "💬 Transcripts", "⚙️ Actions"])

# ========== Overview Tab ==========
with tab1:
    st.header("Overview — Past week focus with 4-week context")
    # Trend: daily counts + sentiment over last 28 days
    trend_df = df[(df['date_only'] >= four_weeks_ago) & (df['date_only'] <= today)].copy()
    trend_daily = trend_df.groupby('date_only').agg(
        count=('id','count'),
        pos=('sentiment', lambda s: (s=='positive').sum()),
        neg=('sentiment', lambda s: (s=='negative').sum()),
        neutral=('sentiment', lambda s: (s=='neutral').sum()),
        avg_comp=('sentiment_compound','mean')
    ).reset_index()
    if trend_daily.empty:
        st.info("Not enough data for trend charts.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend_daily['date_only'], y=trend_daily['count'], name='Count', marker_color='lightblue', yaxis='y1'))
        fig.add_trace(go.Scatter(x=trend_daily['date_only'], y=trend_daily['avg_comp'], name='Avg Sentiment (compound)', mode='lines+markers', yaxis='y2', line=dict(color='darkgreen')))
        fig.update_layout(title="Daily transcripts (last 4 weeks)", xaxis_title="Date",
                          yaxis=dict(title='Count', side='left'),
                          yaxis2=dict(title='Avg sentiment', overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("This week's textual summary")
    week_df = df[(df['date_only'] >= one_week_ago) & (df['date_only'] <= today)]
    bigtext = " ".join(week_df['content'].tolist()[:2000])  # limit size
    summary_text = extractive_summary(bigtext, n_sentences=6)
    st.markdown("**Auto Summary (extractive)**")
    st.write(summary_text if summary_text else "No significant text in the selected week.")

    st.subheader("Weekly key topics & recommendations")
    kw = top_keywords(week_df['content'].tolist() or [""])
    st.write(", ".join([k for k,_ in kw[:12]]))

    # recommendation engine (rules)
    recs = []
    if (week_df['sentiment']=='negative').mean() > 0.3:
        recs.append("High negative sentiment this week — consider urgent retention analysis & customer outreach.")
    if (week_df['content'].str.contains(r'\bdelay|\bdelayed|\btimeline|\brisk\b', case=False, na=False)).any():
        recs.append("Multiple mentions of delays/timeline — review roadmap and issue updates.")
    if (week_df['content'].str.contains(r'\bescalat|\bcritical|\boutage', case=False, na=False)).any():
        recs.append("Escalations detected — schedule leadership review.")
    if not recs:
        recs.append("No immediate automated recommendations detected — continue monitoring.")
    for r in recs:
        st.info(r)

# ========== Topics & Trends Tab ==========
with tab2:
    st.header("Topic discovery & heatmaps")
    # Top topics across selected range
    corpus = f['content'].tolist()
    top_k = top_keywords(corpus or [" "], top_n=30)
    topk_df = pd.DataFrame(top_k, columns=['keyword','score'])
    if not topk_df.empty:
        fig2 = px.bar(topk_df.head(20), x='keyword', y='score', title='Top keywords (TF-IDF)')
        st.plotly_chart(fig2, use_container_width=True)
    # Heatmap of sentiment by weekday
    sent_pivot = f.copy()
    sent_pivot['weekday'] = pd.to_datetime(sent_pivot['date_only']).apply(lambda d: d.weekday())
    pivot = sent_pivot.groupby(['weekday','sentiment']).size().unstack(fill_value=0)
    if not pivot.empty:
        pivot = pivot.reindex(index=range(0,7), fill_value=0)
        fig3 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
            colorscale='Blues'
        ))
        fig3.update_layout(title="Weekday × Sentiment heatmap (selected range)")
        st.plotly_chart(fig3, use_container_width=True)
    # Recurring topics detection (top topics appearing in > N transcripts)
    topic_counter = Counter()
    for row in f['topics']:
        if isinstance(row, list):
            topic_counter.update([t.lower() for t in row])
        else:
            topic_counter.update([str(row).lower()])
    recurring = [(t,c) for t,c in topic_counter.items() if c>=2]
    recurring_df = pd.DataFrame(sorted(recurring, key=lambda x:-x[1]), columns=['topic','count'])
    st.subheader("Recurring topics (>=2 occurrences)")
    st.table(recurring_df.head(30))

# ========== Transcripts Tab ==========
with tab3:
    st.header("Transcripts — search, highlight, and flag")
    search = st.text_input("Search content (regex supported)", value="")
    view = f.copy()
    if search:
        view = view[view['content'].str.contains(search, regex=True, case=False, na=False)]
    st.write(f"Showing {len(view)} transcripts")
    # show table with expanders and highlight search terms
    for idx, row in view.sort_values('date', ascending=False).iterrows():
        with st.expander(f"{row.get('date')} — {row.get('participant')} — {row.get('sentiment')}"):
            # highlight search terms
            content = html.escape(row['content'])
            if search:
                try:
                    content = re.sub(f"(?i)({re.escape(search)})", r"<mark>\1</mark>", content)
                except:
                    content = content
            st.markdown(f"**Topics:** {row.get('topics')}")
            st.markdown(f"**Priority:** {row.get('priority')}")
            st.markdown(f"**Duration:** {row.get('duration')} min")
            st.markdown(content, unsafe_allow_html=True)
            # Actions per transcript
            cols = st.columns([1,1,1,1])
            if cols[0].button("Flag as Risk", key=f"risk_{row['id']}"):
                st.success("Flagged risk — added to follow-up list.")
                # store flag in session for this demo
                flags = st.session_state.get("flags", [])
                flags.append({"id":row['id'], "type":"risk", "note":""})
                st.session_state["flags"] = flags
            if cols[1].button("Create email draft", key=f"email_{row['id']}"):
                # generate an email draft from the transcript
                subj = f"Follow-up: {row.get('participant')} — {row.get('date')}"
                body = extractive_summary(row['content'], n_sentences=5)
                st.session_state["email_draft"] = {"to":"manager@company.com", "subject":subj, "body":body}
                st.success("Email draft created — open Actions tab to review & download.")
            if cols[2].button("Schedule meeting", key=f"meet_{row['id']}"):
                # create a basic .ics calendar invite for download
                dt = row.get('date')
                if pd.isna(dt):
                    dt = datetime.utcnow()
                else:
                    dt = pd.to_datetime(dt)
                start = dt.replace(hour=9, minute=0, second=0, microsecond=0)
                end = start + pd.Timedelta(minutes=30)
                ics_text = create_ics(subject=f"Follow-up with {row.get('participant')}", start=start, end=end, description=extractive_summary(row['content'],3))
                b = ics_text.encode('utf-8')
                fname = f"invite_{row['id']}.ics"
                st.download_button("Download invite (.ics)", b, file_name=fname, mime="text/calendar")
            if cols[3].button("Mark done", key=f"done_{row['id']}"):
                st.success("Marked done.")
    st.download_button("Download shown transcripts (CSV)", view.to_csv(index=False).encode('utf-8'), "filtered_transcripts.csv", "text/csv")

# helper for .ics creation used above (must be after referenced)
def create_ics(subject="Meeting", start=None, end=None, description=""):
    # simple ICS generator
    if start is None:
        start = datetime.utcnow()
    if end is None:
        end = start + timedelta(minutes=30)
    def fmt(dt):
        return dt.strftime("%Y%m%dT%H%M%SZ")
    uid = str(uuid.uuid4())
    ics = textwrap.dedent(f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Napster Dashboard//EN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{fmt(datetime.utcnow())}
DTSTART:{fmt(start)}
DTEND:{fmt(end)}
SUMMARY:{subject}
DESCRIPTION:{description}
END:VEVENT
END:VCALENDAR
""")
    return ics

# ========== Actions Tab ==========
with tab4:
    st.header("Actions & Automation")
    st.subheader("Email draft (auto-generated)")
    ed = st.session_state.get("email_draft", None)
    if ed:
        st.text_input("To", value=ed.get("to",""), key="email_to")
        st.text_input("Subject", value=ed.get("subject",""), key="email_subject")
        st.text_area("Body", value=ed.get("body",""), height=240, key="email_body")
    else:
        st.info("Create an email draft from a transcript to see it here.")
    cols = st.columns([1,1,1])
    if cols[0].button("Download email (.eml)"):
        ed2 = {
            "to": st.session_state.get("email_to",""),
            "subject": st.session_state.get("email_subject",""),
            "body": st.session_state.get("email_body","")
        }
        eml = f"To: {ed2['to']}\nSubject: {ed2['subject']}\n\n{ed2['body']}"
        st.download_button("Download .eml file", eml, "draft.eml", "text/plain")
    if cols[1].button("Export summary (TXT)"):
        summary = extractive_summary(" ".join(f['content'].tolist()),5)
        st.download_button("Download summary", summary, "summary.txt", "text/plain")
    if cols[2].button("Export report (CSV)"):
        st.download_button("Download report CSV", f.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")

    st.markdown("---")
    st.subheader("Simple QA / Chat (search-based)")
    q = st.text_input("Ask a question about the transcripts")
    if st.button("Answer"):
        # TF-IDF match using corpus of transcripts
        corpus = df['content'].fillna("").tolist()
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(corpus)
        qv = vect.transform([q])
        scores = (X @ qv.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:3]
        answers = []
        for i in top_idx:
            if scores[i] > 0:
                answers.append(df.iloc[i]['content'][:800])
        if answers:
            for a in answers:
                st.write(a)
        else:
            st.write("No good match found. Try rephrasing.")

# ---------- Alerts ----------
# Simple alert system for high-priority sentiment spikes in the last 7 days
recent = df[(df['date_only'] >= one_week_ago) & (df['date_only'] <= today)]
if not recent.empty:
    neg_share = (recent['sentiment']=='negative').mean()
    if neg_share >= 0.35:
        st.sidebar.error(f"Alert: Negative sentiment is high this week ({neg_share:.0%})")
    elif neg_share >= 0.20:
        st.sidebar.warning(f"Warning: Elevated negative sentiment ({neg_share:.0%})")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ — Extractive summarization, TF-IDF topic extraction, VADER sentiment. For advanced AI features (LLM summarization, STT), we can integrate external APIs/models.")



