# streamlit_app.py
# Napster transcripts dashboard — Option B (LLM-enabled when OPENAI_API_KEY provided)
# Replace existing file with this content. Add OPENAI_API_KEY to Streamlit Secrets to enable LLM features.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re, io, html, textwrap, uuid, json
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download
import openai
from tqdm import tqdm

# ----------------- Page config -----------------
st.set_page_config(page_title="Napster Transcripts — Advanced (LLM)", layout="wide")
st.title("Napster Transcripts — Advanced Dashboard (LLM-enabled)")

# ----------------- Prepare NLTK VADER -----------------
@st.cache_data(show_spinner=False)
def prepare_nltk():
    try:
        nltk_download("vader_lexicon")
    except Exception:
        pass

prepare_nltk()
sia = SentimentIntensityAnalyzer()

# ----------------- OpenAI setup (optional) -----------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini") if hasattr(st, "secrets") else "gpt-4o-mini"
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ----------------- Password gate (preserve old behavior) -----------------
def check_password():
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
    if not hasattr(st, "secrets") or "password" not in st.secrets:
        # no secret → open access (useful for local dev)
        return True
    def _on_change():
        st.session_state["password_correct"] = (st.session_state.get("pw","") == st.secrets["password"])
        if st.session_state["password_correct"]:
            st.rerun()
    st.text_input("Enter dashboard password", type="password", key="pw", on_change=_on_change)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🔒 Incorrect password")
    return False

if not check_password():
    st.stop()

# ----------------- Helpers: text cleaning, sentiment, topics -----------------
def clean_text(t):
    if pd.isna(t):
        return ""
    s = str(t)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_sentiment_label_and_score(text):
    if not text:
        return 0.0, "neutral"
    v = sia.polarity_scores(text)
    c = v["compound"]
    if c >= 0.05:
        lab = "positive"
    elif c <= -0.05:
        lab = "negative"
    else:
        lab = "neutral"
    return c, lab

def row_top_keywords(text, n=4):
    words = re.findall(r'\w+', str(text).lower())
    stop = set(["the","and","to","of","we","is","in","for","on","with","are","be","that","this","a","an","it","as","by","at","from","or","but","not","will","i"])
    words = [w for w in words if w not in stop and len(w) > 2]
    c = Counter(words)
    return [w for w,_ in c.most_common(n)]

def top_keywords_corpus(corpus, top_n=15):
    vect = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vect.fit_transform(corpus)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(sums)[::-1][:top_n]
    return list(zip(terms[idx], sums[idx]))

# ----------------- OpenAI helpers (abstractive summary, email draft, embeddings+QA) -----------------
def openai_summarize(text, max_tokens=300):
    if not OPENAI_KEY:
        return None
    prompt = f"Please provide a concise summary (bullet points, 4-6 items) of the following meeting transcripts/text. Focus on major issues, actions, and risks:\n\n{text}"
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are an assistant that summarizes meeting transcripts into action-oriented bullet points."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

def openai_create_email(subject, body_summary, to="manager@company.com"):
    if not OPENAI_KEY:
        return None
    prompt = f"Draft a professional follow-up email to {to} with subject '{subject}'. Use the summary below as the body content and add 3 bullet next steps.\n\n{body_summary}"
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a helpful assistant that writes professional emails."},
                      {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

# Simple embedding-based retrieval using OpenAI embeddings if available
def build_embeddings_index(texts):
    """Return list of embeddings (and raw texts). Fallback: return None if no API key."""
    if not OPENAI_KEY:
        return None
    embeddings = []
    batch = []
    ids = []
    for i,t in enumerate(texts):
        batch.append(t)
        ids.append(i)
    try:
        # Use text-embedding-3-small or default
        model = "text-embedding-3-small"
        resp = openai.Embedding.create(model=model, input=batch)
        embeddings = [r['embedding'] for r in resp['data']]
        return embeddings
    except Exception:
        return None

def openai_answer_question(question, docs, embeddings=None):
    """If embeddings provided and OPENAI_KEY set, run a retrieval-augmented answer. Else return None."""
    if not OPENAI_KEY:
        return None
    # Build prompt with top-k docs (simple similarity using cosine if embeddings given; otherwise use top tfidf)
    top_docs = []
    if embeddings:
        # compute cosine similarities
        import numpy as np
        q_emb = openai.Embedding.create(model="text-embedding-3-small", input=[question])['data'][0]['embedding']
        def cos(a,b):
            a = np.array(a); b = np.array(b)
            return float(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        scores = [(i, cos(q_emb, emb)) for i,emb in enumerate(embeddings)]
        scores.sort(key=lambda x:x[1], reverse=True)
        for i,score in scores[:4]:
            top_docs.append(docs[i])
    else:
        # fallback to TF-IDF matching
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(docs)
        try:
            qv = vect.transform([question])
            scores = (X @ qv.T).toarray().ravel()
            top_idx = np.argsort(scores)[::-1][:4]
            for i in top_idx:
                if scores[i] > 0:
                    top_docs.append(docs[i])
        except Exception:
            pass
    context = "\n\n---\n\n".join(top_docs[:4])
    prompt = f"Answer the question using ONLY the context below. If the answer is not present, say 'No answer found in transcripts.'\n\nContext:\n{context}\n\nQuestion: {question}"
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a helpful assistant that answers questions using provided transcript context."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

# ----------------- Load CSV + JSON template -----------------
CSV_PATH = "all_transcripts (3).csv"
JSON_PATH = "napster-transcript-analysis-2025-10-19.json"

@st.cache_data
def load_json(path):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

template = load_json(JSON_PATH)

# allow upload
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_repo = st.sidebar.checkbox("Use repo CSV (default)", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if use_repo:
        df = load_csv(CSV_PATH)
    else:
        df = pd.DataFrame()

# validate
if df.empty:
    st.warning("No transcripts available. Upload CSV or add default file.")
    st.stop()

# normalize columns
df.columns = [c.strip() for c in df.columns]

# infer content/date/participant columns
if 'content' not in df.columns:
    for c in df.columns:
        if any(k in c.lower() for k in ['trans','text','content']):
            df = df.rename(columns={c:'content'}); break
if 'date' not in df.columns:
    for c in df.columns:
        if 'date' in c.lower():
            df = df.rename(columns={c:'date'}); break

# parse date safely
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    df['date'] = pd.NaT

df['date_only'] = df['date'].dt.date
df['id'] = df.get('id', [str(uuid.uuid4()) for _ in range(len(df))])
df['participant'] = df.get('participant', df.get('speaker', 'Unknown'))
df['content'] = df['content'].fillna('').apply(clean_text)

# duration safe handling
if 'duration' in df.columns:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
else:
    df['duration'] = 0

# priority
df['priority'] = df.get('priority', '')

# sentiment: compute if missing
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    st.sidebar.info("Computing sentiment (VADER)...")
    df['sentiment_score'], df['sentiment'] = zip(*df['content'].apply(lambda t: compute_sentiment_label_and_score(t)))
else:
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['content'].apply(lambda t: compute_sentiment_label_and_score(t)[0])

# topics fallback
if 'topics' not in df.columns:
    df['topics'] = df['content'].apply(lambda t: row_top_keywords(t, n=4))

# ----------------- Date & filter controls -----------------
today = pd.to_datetime(datetime.utcnow()).date()
one_week_ago = today - timedelta(days=7)
four_weeks_ago = today - timedelta(days=28)

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date range", value=(four_weeks_ago, today))
participants = ['All'] + sorted(df['participant'].dropna().unique().tolist())
sel_participant = st.sidebar.selectbox("Participant", participants)
sentiments = ['All'] + sorted(df['sentiment'].dropna().unique().tolist())
sel_sent = st.sidebar.selectbox("Sentiment", sentiments)
topic_contains = st.sidebar.text_input("Topic contains (keyword)")
priority_sel = st.sidebar.selectbox("Priority", ['All'] + sorted(df['priority'].dropna().unique().tolist()))

f = df.copy()
start_date, end_date = date_range
f = f[(f['date_only'] >= start_date) & (f['date_only'] <= end_date)]
if sel_participant != 'All':
    f = f[f['participant'] == sel_participant]
if sel_sent != 'All':
    f = f[f['sentiment'] == sel_sent]
if topic_contains:
    f = f[f['topics'].apply(lambda t: any(topic_contains.lower() in w.lower() for w in (t if isinstance(t,list) else [t])) )]
if priority_sel != 'All' and priority_sel:
    f = f[f['priority'] == priority_sel]

# ----------------- KPIs -----------------
col1,col2,col3,col4,col5 = st.columns([1.5,1,1,1,1])
col1.metric("Transcripts (shown)", len(f))
col2.metric("Positive", (f['sentiment']=='positive').sum())
col3.metric("Negative", (f['sentiment']=='negative').sum())
avgdur = f['duration'].replace(0, np.nan).dropna().mean()
col4.metric("Avg duration (min)", f"{avgdur:.1f}" if not pd.isna(avgdur) else "N/A")
urgent_flags = f['content'].str.contains(r'\burgent\b|\bescalat\b|\bdelay|\bdelayed|\bcritical\b', case=False, na=False).sum()
col5.metric("Urgent flags", urgent_flags)

st.markdown("---")

# ----------------- Tabs -----------------
tab_overview, tab_trends, tab_transcripts, tab_actions = st.tabs(["📈 Overview","📊 Trends & Topics","💬 Transcripts","⚙️ Actions"])

# ----- Overview -----
with tab_overview:
    st.header("Weekly summary (LLM if enabled)")
    week_mask = (df['date_only'] >= one_week_ago) & (df['date_only'] <= today)
    week_text = "\n".join(df.loc[week_mask, 'content'].astype(str).tolist())[:3000]
    if OPENAI_KEY and week_text.strip():
        with st.spinner("Generating abstractive summary using OpenAI..."):
            llm_summary = openai_summarize(week_text, max_tokens=350)
        st.subheader("Abstractive summary (LLM)")
        st.write(llm_summary)
    else:
        st.subheader("Extractive summary (local)")
        st.write(extractive_summary(week_text, n_sentences=6) if week_text.strip() else "No content this week.")

    st.subheader("Top topic keywords this week")
    topk = top_keywords_corpus(df.loc[week_mask, 'content'].tolist() or [""])
    if topk:
        st.write(", ".join([k for k,_ in topk[:15]]))

    # Recommendations
    st.subheader("Automated recommendations")
    recs = []
    if (df.loc[week_mask,'sentiment']=='negative').mean() > 0.3:
        recs.append("High negative sentiment this week — consider urgent customer outreach and retention measures.")
    if df.loc[week_mask,'content'].str.contains(r'\bdelay|\btimeline|\brisk', case=False, na=False).any():
        recs.append("Multiple mentions of delay/timeline — investigate release schedule and communicate status.")
    if df.loc[week_mask,'content'].str.contains(r'\bescalat|\boutage|\bcritical', case=False, na=False).any():
        recs.append("Escalations detected — schedule leadership meeting.")
    if not recs:
        recs.append("No automated recommendations detected; continue monitoring.")
    for r in recs:
        st.info(r)

# ----- Trends & Topics -----
with tab_trends:
    st.header("Trends: 4-week context")
    trend_df = df[(df['date_only'] >= four_weeks_ago) & (df['date_only'] <= today)].copy()
    trend_daily = trend_df.groupby('date_only').agg(count=('id','count'), avg_sent=('sentiment_score','mean')).reset_index()
    if not trend_daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend_daily['date_only'], y=trend_daily['count'], name='Count', marker_color='skyblue'))
        fig.add_trace(go.Scatter(x=trend_daily['date_only'], y=trend_daily['avg_sent'], name='Avg Sentiment', yaxis='y2', mode='lines+markers', line=dict(color='green')))
        fig.update_layout(yaxis=dict(title='Count'), yaxis2=dict(overlaying='y', side='right', title='Avg sentiment'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for trend chart.")

    st.subheader("Keyword TF-IDF (selected range)")
    tfidf_kw = top_keywords_corpus(f['content'].tolist() or [" "], top_n=30)
    if tfidf_kw:
        tfidf_df = pd.DataFrame(tfidf_kw, columns=['keyword','score'])
        st.plotly_chart(px.bar(tfidf_df.head(20), x='keyword', y='score', title='Top keywords'), use_container_width=True)

    st.subheader("Weekday × Sentiment heatmap")
    if not f.empty:
        ftemp = f.copy()
        ftemp['weekday'] = pd.to_datetime(ftemp['date_only']).apply(lambda d: d.weekday())
        pivot = ftemp.groupby(['weekday','sentiment']).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=range(7), fill_value=0)
        hm = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], colorscale='Blues'))
        st.plotly_chart(hm, use_container_width=True)

# ----- Transcripts -----
with tab_transcripts:
    st.header("Transcripts — browse, search, and take actions")
    q = st.text_input("Search transcripts (regex supported)")
    view = f if not q else f[f['content'].str.contains(q, regex=True, case=False, na=False)]
    st.write(f"Showing {len(view)} transcripts")
    for _, row in view.sort_values('date', ascending=False).iterrows():
        with st.expander(f"{row.get('date')} — {row.get('participant')} — {row.get('sentiment')}"):
            content_html = html.escape(row['content'])
            if q:
                try:
                    content_html = re.sub(f"(?i)({re.escape(q)})", r"<mark>\1</mark>", content_html)
                except re.error:
                    pass
            st.markdown(f"**Topics:** {row.get('topics')}")
            st.markdown(f"**Priority:** {row.get('priority')}")
            st.markdown(f"**Duration:** {row.get('duration')} min")
            st.markdown(content_html, unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            if c1.button("Flag risk", key=f"risk_{row['id']}"):
                flags = st.session_state.get("flags", [])
                flags.append({"id":row['id'], "type":"risk", "note":""})
                st.session_state["flags"] = flags
                st.success("Flagged as risk.")
            if c2.button("Draft email (LLM)", key=f"email_{row['id']}"):
                subj = f"Follow-up: {row.get('participant')} — {row.get('date')}"
                body_summary = extractive_summary(row['content'], n_sentences=4)
                if OPENAI_KEY:
                    with st.spinner("Generating email with OpenAI..."):
                        email_text = openai_create_email(subj, body_summary)
                else:
                    email_text = f"Subject: {subj}\n\n{body_summary}\n\nNext steps:\n- [ ] Follow up\n- [ ] Assign owner"
                st.session_state["draft_email"] = {"to":"manager@company.com","subject":subj,"body":email_text}
                st.success("Email draft created. Open Actions tab to view/download.")
            if c3.button("Create .ics invite", key=f"ics_{row['id']}"):
                dt = row.get('date')
                if pd.isna(dt):
                    dt = datetime.utcnow()
                else:
                    dt = pd.to_datetime(dt)
                start = dt.replace(hour=9, minute=0, second=0, microsecond=0)
                end = start + timedelta(minutes=30)
                ics_text = textwrap.dedent(f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Napster//EN
BEGIN:VEVENT
UID:{str(uuid.uuid4())}
DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:Follow-up with {row.get('participant')}
DESCRIPTION:{extractive_summary(row['content'],3)}
END:VEVENT
END:VCALENDAR
""")
                st.download_button("Download invite (.ics)", ics_text.encode('utf-8'), file_name=f"invite_{row['id']}.ics", mime="text/calendar")

    st.download_button("Download shown transcripts (CSV)", view.to_csv(index=False).encode('utf-8'), file_name="filtered_transcripts.csv", mime="text/csv")

# ----- Actions -----
with tab_actions:
    st.header("Actions & Automation")
    st.subheader("Email draft (view & download)")
    ed = st.session_state.get("draft_email")
    if ed:
        to = st.text_input("To", value=ed.get("to",""), key="email_to")
        subj = st.text_input("Subject", value=ed.get("subject",""), key="email_subject")
        body = st.text_area("Body", value=ed.get("body",""), height=240, key="email_body")
        if st.button("Download .eml"):
            eml = f"To: {to}\nSubject: {subj}\n\n{body}"
            st.download_button("Click to download email file", eml.encode('utf-8'), file_name="draft.eml", mime="text/plain")
    else:
        st.info("Create an email draft from the Transcripts tab.")

    st.markdown("---")
    st.subheader("Question-answer (LLM retrieval)")
    user_q = st.text_input("Ask a question about the transcripts")
    if st.button("Get answer"):
        if OPENAI_KEY:
            docs = df['content'].fillna("").tolist()
            with st.spinner("Building embeddings and answering (OpenAI)..."):
                embeddings = build_embeddings_index(docs)
                ans = openai_answer_question(user_q, docs, embeddings=embeddings)
            st.write(ans)
        else:
            # fallback: TF-IDF match results
            vect = TfidfVectorizer(stop_words='english')
            X = vect.fit_transform(df['content'].fillna("").tolist())
            qv = vect.transform([user_q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(scores)[::-1][:3]
            for i in top:
                if scores[i] > 0:
                    st.write(df.iloc[i]['content'][:800])
            if scores.max() == 0:
                st.write("No match found. Try rephrasing or enable OPENAI_API_KEY for better answers.")

# ----------------- Alerts (sidebar) -----------------
recent = df[(df['date_only'] >= one_week_ago) & (df['date_only'] <= today)]
if not recent.empty:
    neg_share = (recent['sentiment']=='negative').mean()
    if neg_share >= 0.35:
        st.sidebar.error(f"Alert: Negative sentiment high ({neg_share:.0%})")
    elif neg_share >= 0.20:
        st.sidebar.warning(f"Warning: Elevated negative sentiment ({neg_share:.0%})")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("LLM features enabled when OPENAI_API_KEY present in Streamlit Secrets. Built with Streamlit + OpenAI + NLTK + scikit-learn.")
