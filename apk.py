# app.py
# =============================================================================
# Streamlit App — Sentiment Analysis Berita CNBC Indonesia
# Model  : Naive Bayes | SVM | Naive Bayes Optimized | SVM Optimized
# Deploy : streamlit run app.py
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
.result-card {
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin-top: 1rem;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    letter-spacing: 0.5px;
}
.positif  { background: #E8F5E9; color: #2E7D32; border: 2px solid #66BB6A; }
.negatif  { background: #FFEBEE; color: #B71C1C; border: 2px solid #EF5350; }
.netral   { background: #ECEFF1; color: #37474F; border: 2px solid #90A4AE; }

.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    background: #1565C0;
    color: white;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.preprocess-box {
    background: #F5F5F5;
    border-left: 4px solid #1565C0;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #333;
    word-break: break-word;
}

.conf-label { font-size: 0.85rem; color: #555; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    models = {}
    files = {
        "Naive Bayes"          : "naive_bayes.pkl",
        "SVM"                  : "svm.pkl",
        "Naive Bayes Optimized": "naive_bayes_optimized.pkl",
        "SVM Optimized"        : "svm_optimized.pkl",
    }
    for name, path in files.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            models[name] = None

    try:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
    except FileNotFoundError:
        tfidf = None

    return models, tfidf


models, tfidf = load_models()

# Emoji & warna per kelas
SENTIMENT_META = {
    "positif": {"emoji": "😊", "label": "POSITIF", "css": "positif"},
    "netral" : {"emoji": "😐", "label": "NETRAL",  "css": "netral"},
    "negatif": {"emoji": "😞", "label": "NEGATIF", "css": "negatif"},
}
LABEL_MAP     = {0: "negatif", 1: "netral", 2: "positif"}
MODEL_ICONS   = {
    "Naive Bayes"          : "🔵",
    "SVM"                  : "🟠",
    "Naive Bayes Optimized": "🟢",
    "SVM Optimized"        : "🔴",
}
MODEL_DESC = {
    "Naive Bayes"          : "Model probabilistik berbasis Teorema Bayes dengan alpha=1.0.",
    "SVM"                  : "Linear SVC dengan C=1.0 dan fitur TF-IDF N-gram (1,2).",
    "Naive Bayes Optimized": "Naive Bayes dengan alpha optimal hasil GridSearchCV 10-Fold.",
    "SVM Optimized"        : "LinearSVC dengan C optimal hasil GridSearchCV 10-Fold.",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/CNBC_logo.svg/"
        "320px-CNBC_logo.svg.png",
        width=130,
    )
    st.title("⚙️ Pengaturan")
    st.markdown("---")

    selected_model = st.selectbox(
        "🤖 Pilih Model",
        options=list(models.keys()),
        format_func=lambda x: f"{MODEL_ICONS[x]}  {x}",
    )

    st.markdown(f"<small>📄 {MODEL_DESC[selected_model]}</small>", unsafe_allow_html=True)
    st.markdown("---")

    show_preprocess = st.toggle("Tampilkan hasil preprocessing", value=True)
    show_steps      = st.toggle("Tampilkan langkah-langkah preprocessing", value=False)

    st.markdown("---")
    st.markdown("### 📊 Status Model")
    for name, mdl in models.items():
        icon = "✅" if mdl is not None else "❌"
        st.markdown(f"{icon} `{name}`")
    tfidf_status = "✅" if tfidf is not None else "❌"
    st.markdown(f"{tfidf_status} `TF-IDF Vectorizer`")

    st.markdown("---")
    st.caption("Dataset: CNBC Indonesia News\n9.819 judul berita\n(positif / netral / negatif)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📰 Analisis Sentimen Berita")
st.markdown(
    "Masukkan judul berita berbahasa Indonesia untuk dianalisis sentimennya "
    "menggunakan model **Machine Learning** berbasis **TF-IDF N-gram (1,2)**."
)

# ── Input teks (FIX DI SINI) ────────────────────────────────────────────────
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    label="✏️ Judul Berita",
    placeholder="Contoh: Bank BRI Catat Laba Bersih Tumbuh 18% pada Kuartal III 2024",
    height=110,
    help="Masukkan satu atau beberapa kalimat judul berita berbahasa Indonesia.",
    key="user_input"
)

# ── Contoh cepat ─────────────────────────────────────────────────────────────
st.markdown("**🔖 Coba contoh cepat:**")
col1, col2, col3 = st.columns(3)
example_texts = {
    "😊 Positif": "Bank BRI Catat Laba Bersih Tumbuh 18% pada Kuartal III 2024",
    "😞 Negatif": "Garuda Indonesia Rugi Rp 2,4 Triliun, Saham Anjlok 15%",
    "😐 Netral" : "OJK Rilis Aturan Baru Terkait Investasi Reksa Dana 2024",
}

with col1:
    if st.button("😊 Contoh Positif", use_container_width=True):
        st.session_state.user_input = example_texts["😊 Positif"]
        st.rerun()

with col2:
    if st.button("😞 Contoh Negatif", use_container_width=True):
        st.session_state.user_input = example_texts["😞 Negatif"]
        st.rerun()

with col3:
    if st.button("😐 Contoh Netral", use_container_width=True):
        st.session_state.user_input = example_texts["😐 Netral"]
        st.rerun()

# ── Tombol Analisis ───────────────────────────────────────────────────────────
analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("Menganalisis..."):
        time.sleep(0.4)

        model = models[selected_model]
        if model is None or tfidf is None:
            st.error("Model atau TF-IDF belum tersedia")
        else:
            clean = preprocess(user_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

            label = LABEL_MAP[pred]
            meta = SENTIMENT_META[label]

            st.markdown("---")
            st.markdown(
                f"<div class='result-card {meta['css']}'>"
                f"{meta['emoji']}  {meta['label']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if show_preprocess:
                st.markdown("#### 🧹 Teks Setelah Preprocessing")
                st.markdown(
                    f"<div class='preprocess-box'>{clean}</div>",
                    unsafe_allow_html=True,
                )
