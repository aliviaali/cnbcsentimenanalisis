# app.py
# =============================================================================
# Streamlit App — Sentiment Analysis Berita CNBC Indonesia
# =============================================================================

import pickle
import time
import streamlit as st
from preprocessing import preprocess

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — CNBC Indonesia",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card { padding: 1.4rem; border-radius: 14px; text-align:center; font-weight:600; }
.positif  { background: #E8F5E9; color: #2E7D32; }
.negatif  { background: #FFEBEE; color: #B71C1C; }
.netral   { background: #ECEFF1; color: #37474F; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "Naive Bayes": "naive_bayes.pkl",
        "SVM": "svm.pkl",
        "Naive Bayes Optimized": "naive_bayes_optimized.pkl",
        "SVM Optimized": "svm_optimized.pkl",
    }
    for name, path in files.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except:
            models[name] = None

    try:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
    except:
        tfidf = None

    return models, tfidf

models, tfidf = load_models()

LABEL_MAP = {0: "negatif", 1: "netral", 2: "positif"}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📰 Analisis Sentimen Berita")

# =========================
# FIX SESSION STATE
# =========================
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# =========================
# TEXT INPUT (FIXED)
# =========================
user_input = st.text_area(
    "✏️ Judul Berita",
    height=120,
    key="user_input"
)

# =========================
# QUICK BUTTONS (FIXED)
# =========================
st.markdown("### 🔖 Contoh Cepat")
col1, col2, col3 = st.columns(3)

example_texts = {
    "positif": "Bank BRI Catat Laba Bersih Tumbuh 18%",
    "negatif": "Garuda Indonesia Rugi Rp 2,4 Triliun",
    "netral": "OJK Rilis Aturan Baru Investasi 2024"
}

with col1:
    if st.button("😊 Positif"):
        st.session_state.user_input = example_texts["positif"]
        st.rerun()

with col2:
    if st.button("😞 Negatif"):
        st.session_state.user_input = example_texts["negatif"]
        st.rerun()

with col3:
    if st.button("😐 Netral"):
        st.session_state.user_input = example_texts["netral"]
        st.rerun()

# =========================
# PILIH MODEL
# =========================
model_name = st.selectbox("Pilih Model", list(models.keys()))

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🔍 Analisis"):

    if not user_input.strip():
        st.warning("Masukkan teks dulu")
    else:
        model = models[model_name]

        if model is None or tfidf is None:
            st.error("Model atau TF-IDF belum tersedia")
        else:
            clean = preprocess(user_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

            label = LABEL_MAP[pred]

            st.markdown("---")

            if label == "positif":
                st.markdown(f"<div class='result-card positif'>😊 POSITIF</div>", unsafe_allow_html=True)
            elif label == "negatif":
                st.markdown(f"<div class='result-card negatif'>😞 NEGATIF</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card netral'>😐 NETRAL</div>", unsafe_allow_html=True)

            st.write("### 🧹 Preprocessing")
            st.write(clean)
```
