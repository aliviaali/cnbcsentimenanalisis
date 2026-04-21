```python
# app.py
# =============================================================================
# Streamlit App — Sentiment Analysis Berita CNBC Indonesia
# Model  : Naive Bayes | SVM | Naive Bayes Optimized | SVM Optimized
# =============================================================================

import pickle
import time
import streamlit as st
from preprocessing import preprocess

st.set_page_config(
    page_title="Sentiment Analysis — CNBC Indonesia",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.result-card {
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin-top: 1rem;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
}
.positif  { background: #E8F5E9; color: #2E7D32; }
.negatif  { background: #FFEBEE; color: #B71C1C; }
.netral   { background: #ECEFF1; color: #37474F; }
</style>
""", unsafe_allow_html=True)

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

SENTIMENT_META = {
    "positif": {"emoji": "😊", "label": "POSITIF", "css": "positif"},
    "netral" : {"emoji": "😐", "label": "NETRAL",  "css": "netral"},
    "negatif": {"emoji": "😞", "label": "NEGATIF", "css": "negatif"},
}
LABEL_MAP = {0: "negatif", 1: "netral", 2: "positif"}

st.title("📰 Analisis Sentimen Berita")

# =========================
# SESSION STATE
# =========================
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# =========================
# TEXT INPUT (FIX DI SINI)
# =========================
user_input = st.text_area(
    label="✏️ Judul Berita",
    placeholder="Contoh: Bank BRI Catat Laba Bersih Tumbuh 18%",
    height=110,
    help="Masukkan teks berita",
    key="user_input"  # ✅ FIX
)

# =========================
# CONTOH CEPAT
# =========================
st.markdown("**🔖 Coba contoh cepat:**")
col1, col2, col3 = st.columns(3)

example_texts = {
    "😊 Positif": "Bank BRI Catat Laba Bersih Tumbuh 18%",
    "😞 Negatif": "Garuda Indonesia Rugi Rp 2,4 Triliun",
    "😐 Netral": "OJK Rilis Aturan Baru Investasi 2024",
}

with col1:
    if st.button("😊 Contoh Positif"):
        st.session_state.user_input = example_texts["😊 Positif"]
        st.rerun()

with col2:
    if st.button("😞 Contoh Negatif"):
        st.session_state.user_input = example_texts["😞 Negatif"]
        st.rerun()

with col3:
    if st.button("😐 Contoh Netral"):
        st.session_state.user_input = example_texts["😐 Netral"]
        st.rerun()

# =========================
# PILIH MODEL
# =========================
model_name = st.selectbox("Pilih Model", list(models.keys()))

# =========================
# ANALISIS
# =========================
if st.button("🔍 Analisis Sentimen"):

    if not user_input.strip():
        st.warning("Masukkan teks terlebih dahulu")
    else:
        model = models[model_name]

        if model is None or tfidf is None:
            st.error("Model belum tersedia")
        else:
            clean = preprocess(user_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

            label = LABEL_MAP[pred]
            meta = SENTIMENT_META[label]

            st.markdown("---")
            st.markdown(
                f"<div class='result-card {meta['css']}'>"
                f"{meta['emoji']} {meta['label']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### 🧹 Hasil Preprocessing")
            st.write(clean)
```
