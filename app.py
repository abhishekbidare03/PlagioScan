# app.py (corrected + preset/session_state support)
import streamlit as st
import textwrap
import pandas as pd
import numpy as np
from utils import (
    load_model,
    split_sentences,
    clean_text,
    compute_embeddings,
    highlight_sentence,
    top_common_words,
    top_tfidf_terms_for_sentence,
    make_report,
)

# Professional theme configuration
st.set_page_config(
    page_title="PlagioScan — AI-Powered Plagiarism Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, elegant design with Muted Green theme
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #6B8E7F 0%, #5A7968 100%);
        padding: 0;
    }
            /* REMOVE the black top background */
[data-testid="stAppViewContainer"] {
    background: #f5f7fb !important;   /* light background */
}

/* Remove black header bar from theme */
[data-testid="stHeader"] {
    background: transparent !important;
}

    
    .block-container {
        padding: 2rem 3rem;
        background: white;
        border-radius: 20px;
        margin: 2rem auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        max-width: 1400px;
    }
    
    /* Header styling */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #6B8E7F 0%, #4A6B5C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stCaption {
        color: #6b7280 !important;
        font-size: 1.1rem !important;
        font-weight: 400;
    }
    
   /* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8F5E9 0%, #C8E6C9 100%); /* Light muted green gradient */
    padding: 2rem 1rem;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2 {
    color: black; /* White */
    font-weight: 600;
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
}

/* Markdown text */
[data-testid="stSidebar"] .stMarkdown {
    color: black; /* White */
}

/* Labels */
[data-testid="stSidebar"] label {
    color: black !important; /* White */
    font-weight: 500;
}

/* Selectbox and Slider labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #ffffff !important; /* White */
}

/* Target selectbox dropdown */
div[data-baseweb="select"] {
    background-color: #ffffff;   /* White background */
    color: #000000;              /* Black text */
    border-radius: 6px;
    border: 1px solid #ccc;
}

/* Dropdown options */
div[data-baseweb="select"] span {
    color: #000000 !important;   /* Ensure options are visible */
}
    
    /* Subheaders */
    h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
        margin-top: 2rem !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        border: 2px solid #D4E4DC;
        border-radius: 12px;
        padding: 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: #F8FBF9;
        color: #0f172a;
    }
    
    .stTextArea textarea:focus {
        border-color: #6B8E7F;
        box-shadow: 0 0 0 3px rgba(107, 142, 127, 0.15);
        background: white;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #F8FBF9;
        border: 2px dashed #B8CEC4;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6B8E7F;
        background: #EDF5F1;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6B8E7F 0%, #5A7968 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(107, 142, 127, 0.4);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(107, 142, 127, 0.6);
        background: linear-gradient(135deg, #5A7968 0%, #4A6B5C 100%);
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #6B8E7F 0%, #5A7968 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(107, 142, 127, 0.35);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(107, 142, 127, 0.55);
        background: linear-gradient(135deg, #5A7968 0%, #4A6B5C 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6B8E7F 0%, #4A6B5C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #4b5563;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F8FBF9 0%, #EDF5F1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(107, 142, 127, 0.12);
        border: 2px solid #D4E4DC;
    }
    
    /* Tables */
    .stTable {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    table {
        border: none !important;
    }
    
    thead tr th {
        background: linear-gradient(135deg, #6B8E7F 0%, #5A7968 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    tbody tr:nth-child(even) {
        background: #F8FBF9 !important;
    }
    
    tbody tr:hover {
        background: #EDF5F1 !important;
    }
    
    tbody td {
        padding: 0.9rem !important;
        border-bottom: 1px solid #D4E4DC !important;
    }
    
    /* Highlighted documents */
    .highlighted-doc {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #D4E4DC;
        box-shadow: 0 2px 10px rgba(107, 142, 127, 0.08);
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #D4E4DC, transparent);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #6B8E7F transparent transparent transparent !important;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #1B5E20;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        color: #B71C1C;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #F44336;
    }
    
    /* Info message */
    .stInfo {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        color: #0D47A1;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2196F3;
    }
    
    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Column headers */
    [data-testid="column"] h3 {
        color: #2C3E3C;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6B8E7F;
        display: inline-block;
        margin-bottom: 1.5rem !important;
    }
    
    /* Card-like sections */
    .analysis-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(107, 142, 127, 0.1);
        border: 1px solid #D4E4DC;
    }
    
    /* Heatmap improvements */
    table[style*="border-collapse"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Checkbox */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    /* Logo/Brand area */
    .brand-subtitle {
        background: linear-gradient(135deg, #6B8E7F 0%, #5A7968 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
        box-shadow: 0 2px 10px rgba(107, 142, 127, 0.3);
    }
    
    /* Status badges for similarity scores */
    .status-low {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFF59D 100%);
        color: #F57F17;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #FBC02D;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #FFE0B2 0%, #FFCC80 100%);
        color: #E65100;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #FB8C00;
    }
    
    .status-high {
        background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%);
        color: #B71C1C;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #E53935;
    }
</style>
""", unsafe_allow_html=True)

# Header with professional branding
st.title("🔍 PlagioScan")
st.caption("AI-Powered Semantic Plagiarism Detection System")
st.markdown('<div class="brand-subtitle">Powered by Neural Embeddings • Fast • Accurate • Offline</div>', unsafe_allow_html=True)

# Setup session_state defaults so we can update textareas reliably
if "doc_a" not in st.session_state:
    st.session_state["doc_a"] = ""
if "doc_b" not in st.session_state:
    st.session_state["doc_b"] = ""

# Sidebar with professional controls
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    st.markdown("### Detection Sensitivity")
    sim_threshold = st.slider(
        "Highlight threshold", 
        0.20, 0.9, 0.45, 0.05,
        help="Lower values detect more matches, higher values only flag strong similarities"
    )
    
    st.markdown("### Results Display")
    top_k = st.slider("Top matched pairs", 1, 12, 6, help="Number of top matches to display in detail")
    show_heatmap = st.checkbox("Show similarity matrix", value=False)
    
    st.markdown("---")
    st.markdown("### 📋 Quick Start")
    preset = st.selectbox(
        "Load sample data",
        ["None", "Identical sample", "Paraphrase sample", "Unrelated sample"],
        help="Load example texts to see the system in action"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.caption("**Engine:** all-MiniLM-L6-v2")
    st.caption("**Type:** Sentence Transformer")
    st.caption("**Speed:** Optimized for CPU")
    
    st.markdown("---")
    st.markdown("### 🎨 Status Legend")
    st.markdown('<div class="status-low">Low (< 50%)</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-medium">Medium (50-75%)</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-high">High (> 75%)</div>', unsafe_allow_html=True)

# Sample texts
SAMPLE_IDENTICAL_A = textwrap.dedent("""\
The quick brown fox jumps over the lazy dog. This sentence is commonly used as a pangram to test fonts and input devices.
""")
SAMPLE_IDENTICAL_B = SAMPLE_IDENTICAL_A

SAMPLE_PARAPHRASE_A = textwrap.dedent("""\
Neural networks are widely used in modern AI systems. They automatically learn representations from large datasets and often generalize well to unseen examples.
""")
SAMPLE_PARAPHRASE_B = textwrap.dedent("""\
Modern AI frequently relies on neural networks which learn features from extensive datasets and can generalize to novel inputs effectively.
""")

SAMPLE_UNRELATED_A = textwrap.dedent("""\
Photosynthesis enables plants to convert sunlight into chemical energy while producing oxygen as a byproduct.
""")
SAMPLE_UNRELATED_B = textwrap.dedent("""\
Cloud computing delivers on-demand computing resources over the internet, allowing applications to scale without owning physical servers.
""")

# If a preset is chosen, update session_state so textareas reflect it
if preset != "None":
    if preset == "Identical sample":
        st.session_state["doc_a"] = SAMPLE_IDENTICAL_A
        st.session_state["doc_b"] = SAMPLE_IDENTICAL_B
    elif preset == "Paraphrase sample":
        st.session_state["doc_a"] = SAMPLE_PARAPHRASE_A
        st.session_state["doc_b"] = SAMPLE_PARAPHRASE_B
    elif preset == "Unrelated sample":
        st.session_state["doc_a"] = SAMPLE_UNRELATED_A
        st.session_state["doc_b"] = SAMPLE_UNRELATED_B

# Input columns with professional styling
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📄 Document A")
    txt_a = st.text_area("Enter your source document", height=280, key="doc_a", placeholder="Paste your original text here...")
    file_a = st.file_uploader("📎 Upload text file", type=["txt"], key="file_a")
    if file_a:
        try:
            content = file_a.read().decode("utf-8")
        except Exception:
            content = file_a.getvalue().decode("utf-8")
        st.session_state["doc_a"] = content
        txt_a = content

with col_b:
    st.subheader("📄 Document B")
    txt_b = st.text_area("Enter your comparison document", height=280, key="doc_b", placeholder="Paste the text to check against...")
    file_b = st.file_uploader("📎 Upload text file", type=["txt"], key="file_b")
    if file_b:
        try:
            content = file_b.read().decode("utf-8")
        except Exception:
            content = file_b.getvalue().decode("utf-8")
        st.session_state["doc_b"] = content
        txt_b = content

st.markdown("---")

# Analyze button
if st.button("🚀 Analyze for Plagiarism"):
    txt_a = clean_text(st.session_state.get("doc_a", "") or "")
    txt_b = clean_text(st.session_state.get("doc_b", "") or "")
    if not txt_a or not txt_b:
        st.error("⚠️ Please provide text for both documents (paste or upload).")
    else:
        sents_a = split_sentences(txt_a)
        sents_b = split_sentences(txt_b)

        if len(sents_a) == 0 or len(sents_b) == 0:
            st.error("⚠️ Could not split sentences. Please provide longer/valid text in both documents.")
        else:
            # optional speed guard: limit sentences (uncomment if needed)
            # MAX_SENTS = 300
            # sents_a = sents_a[:MAX_SENTS]
            # sents_b = sents_b[:MAX_SENTS]

            with st.spinner("🔬 Analyzing documents with neural embeddings..."):
                model = load_model()
                sim_mat = compute_embeddings(model, sents_a, sents_b)

            if sim_mat.size == 0:
                doc_score = 0.0
            else:
                max_per_a = sim_mat.max(axis=1)
                doc_score = float(np.mean(max_per_a)) * 100.0

            # Display metric with status badge
            st.markdown("### 📊 Overall Similarity Score")
            
            # Determine status level
            if doc_score < 50:
                status_class = "status-low"
                status_text = "Low Risk"
            elif doc_score < 75:
                status_class = "status-medium"
                status_text = "Medium Risk"
            else:
                status_class = "status-high"
                status_text = "High Risk"
            
            col_metric, col_status = st.columns([2, 1])
            with col_metric:
                st.metric("Semantic Similarity", f"{doc_score:.2f}%")
            with col_status:
                st.markdown(f'<div style="padding-top: 1rem;"><div class="{status_class}">{status_text}</div></div>', unsafe_allow_html=True)

            best_matches = []
            for i in range(len(sents_a)):
                if sim_mat.shape[1] == 0:
                    j = -1
                    sc = 0.0
                else:
                    j = int(sim_mat[i].argmax())
                    sc = float(sim_mat[i, j])
                best_matches.append((i, j, sc))

            best_matches_b = []
            if sim_mat.shape[0] > 0:
                for j in range(len(sents_b)):
                    i = int(sim_mat[:, j].argmax())
                    sc = float(sim_mat[i, j])
                    best_matches_b.append((i, j, sc))

            # Build highlighted HTML
            left_html = ""
            for i, s in enumerate(sents_a):
                _, best_j, best_score = best_matches[i]
                left_html += highlight_sentence(s, best_score, sim_threshold) + " "

            right_html = ""
            for j, s in enumerate(sents_b):
                best_score = 0.0
                if best_matches_b:
                    best_score = best_matches_b[j][2]
                right_html += highlight_sentence(s, best_score, sim_threshold) + " "

            st.markdown("---")
            st.markdown("### 🎯 Highlighted Analysis")
            st.caption("Sentences are color-coded based on similarity strength (Yellow: Minor, Orange: Medium, Red: High)")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📄 Document A**")
                st.markdown(f'<div class="highlighted-doc">{left_html}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**📄 Document B**")
                st.markdown(f'<div class="highlighted-doc">{right_html}</div>', unsafe_allow_html=True)

            pairs_sorted = sorted(best_matches, key=lambda x: x[2], reverse=True)[:top_k]
            rows = []
            corpus_for_tfidf = sents_a + sents_b
            for i, j, sc in pairs_sorted:
                if j < 0 or j >= len(sents_b):
                    continue
                a_sent = sents_a[i]
                b_sent = sents_b[j]
                common = top_common_words(a_sent, b_sent)
                top_tokens_a = top_tfidf_terms_for_sentence(a_sent, corpus_for_tfidf, n=5)
                top_tokens_b = top_tfidf_terms_for_sentence(b_sent, corpus_for_tfidf, n=5)
                rows.append({
                    "A_index": i,
                    "A_sentence": a_sent,
                    "B_index": j,
                    "B_sentence": b_sent,
                    "score": sc,
                    "common_words": common,
                    "top_tokens_a": top_tokens_a,
                    "top_tokens_b": top_tokens_b
                })

            st.markdown("---")
            st.markdown("### 🔍 Top Matched Pairs")
            st.caption("Most similar sentence pairs ranked by semantic similarity")
            
            if rows:
                df = pd.DataFrame([{k: (v if k not in ["A_sentence", "B_sentence"] else (v if len(v) < 120 else v[:117] + '...')) for k, v in r.items()} for r in rows])
                st.table(df)
            else:
                st.info("ℹ️ No high-scoring pairs found. Try adjusting the threshold slider.")

            if show_heatmap and sim_mat.size > 0:
                st.markdown("---")
                st.markdown("### 🌡️ Similarity Heatmap")
                st.caption("Visual representation of sentence-to-sentence similarity scores")
                
                maxv = float(sim_mat.max()) if sim_mat.size>0 else 1.0
                heat_html = '<div style="overflow:auto; max-height:280px;"><table style="border-collapse:collapse;">'
                heat_html += "<tr><th style='padding:6px;'></th>"
                for j in range(len(sents_b)):
                    heat_html += f'<th style="padding:6px;border:1px solid #eee; min-width:80px;">B{j}</th>'
                heat_html += "</tr>"
                for i in range(len(sents_a)):
                    heat_html += f"<tr><th style='padding:6px;border:1px solid #eee;'>A{i}</th>"
                    for j in range(len(sents_b)):
                        v = sim_mat[i,j]
                        bg = f"rgb({255},{255-int((v/maxv)*180) if maxv>0 else 255},{255-int((v/maxv)*220) if maxv>0 else 255})"
                        heat_html += f'<td style="padding:6px; border:1px solid #eee; background:{bg}; min-width:80px; text-align:center;">{v:.2f}</td>'
                    heat_html += "</tr>"
                heat_html += "</table></div>"
                st.markdown(heat_html, unsafe_allow_html=True)

            st.markdown("---")
            report_text = make_report(doc_score, rows)
            st.download_button("📥 Download Detailed Report", data=report_text, file_name="plagioscan_report.txt", mime="text/plain")

            st.success("✅ Analysis complete! Adjust the threshold in the sidebar to fine-tune results.")
