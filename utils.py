# utils.py
import re
import html
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache a SentenceTransformer model."""
    return SentenceTransformer(model_name)

def split_sentences(text: str):
    """Naive sentence splitter that keeps sentence punctuation.
       Splits on punctuation followed by whitespace or newlines."""
    if not text or not text.strip():
        return []
    text = text.strip()
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def clean_text(text: str):
    return text.strip()

def compute_embeddings(model: SentenceTransformer, sents_a, sents_b):
    """Return numpy cosine-similarity matrix (A x B)."""
    if len(sents_a) == 0 or len(sents_b) == 0:
        return np.zeros((len(sents_a), len(sents_b)))
    emb_a = model.encode(sents_a, convert_to_tensor=True)
    emb_b = model.encode(sents_b, convert_to_tensor=True)
    with np.errstate(all='ignore'):
        cos = util.cos_sim(emb_a, emb_b).cpu().numpy()
    return cos

def map_opacity(score, threshold):
    """Map similarity score to opacity for highlighting. Returns float 0..1."""
    if score < threshold:
        return 0.0
    t = (score - threshold) / (1 - threshold)
    return max(0.05, min(0.95, 0.2 + 0.75 * t))

def highlight_sentence(sent, score, threshold):
    """Return HTML for a sentence highlighted according to score."""
    if score is None or score <= 0 or score < threshold:
        return html.escape(sent)
    opacity = map_opacity(score, threshold)
    return f'<mark title="sim={score:.3f}" style="background: rgba(255,165,0,{opacity}); padding:2px;">{html.escape(sent)}</mark>'

def top_common_words(a, b, n=6):
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    common = sorted(list(wa & wb), key=lambda x: -len(x))
    return ", ".join(common[:n]) if common else "-"

def top_tfidf_terms_for_sentence(sentence, corpus_sentences, n=5):
    """Return top tf-idf terms for one sentence relative to a corpus of sentences."""
    try:
        vect = TfidfVectorizer(max_features=2000, stop_words='english')
        X = vect.fit_transform(corpus_sentences)
        feature_names = np.array(vect.get_feature_names_out())
        idx = None
        for i, s in enumerate(corpus_sentences):
            if s == sentence:
                idx = i
                break
        if idx is None:
            tfidf_single = vect.transform([sentence]).toarray()[0]
            top_idx = np.argsort(tfidf_single)[::-1][:n]
            return ", ".join(feature_names[top_idx])
        row = X[idx].toarray()[0]
        top_idx = np.argsort(row)[::-1][:n]
        return ", ".join(feature_names[top_idx])
    except Exception:
        tokens = re.findall(r"\w+", sentence.lower())
        tokens = sorted(set(tokens), key=lambda x: -len(x))
        return ", ".join(tokens[:n]) if tokens else "-"

def make_report(doc_score, rows):
    lines = []
    lines.append("PlagioScan Report")
    lines.append("")
    lines.append(f"Document semantic similarity: {doc_score:.2f}%")
    lines.append("")
    lines.append("Top matched sentence pairs:")
    for r in rows:
        lines.append("-" * 60)
        lines.append(f"A[{r['A_index']}] (score={r['score']:.3f}) --> B[{r['B_index']}]")
        lines.append(f"A: {r['A_sentence']}")
        lines.append(f"B: {r['B_sentence']}")
        lines.append(f"Common words: {r['common_words']}")
        lines.append(f"Top tokens (A): {r['top_tokens_a']}")
        lines.append(f"Top tokens (B): {r['top_tokens_b']}")
        lines.append("")
    return "\n".join(lines)
