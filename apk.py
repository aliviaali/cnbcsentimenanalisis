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
/* Card hasil prediksi */
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

/* Badge model */
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

/* Pre-processed text box */
.preprocess-box {
    background: #F5F5F5;
    border-left: 4px solid #1565C0;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #333;
    word-break: break-word;
}

/* Confidence bar labels */
.conf-label { font-size: 0.85rem; color: #555; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    """Load semua model dan TF-IDF vectorizer dari file .pkl"""
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

    # Pilihan model
    selected_model = st.selectbox(
        "🤖 Pilih Model",
        options=list(models.keys()),
        format_func=lambda x: f"{MODEL_ICONS[x]}  {x}",
    )

    st.markdown(f"<small>📄 {MODEL_DESC[selected_model]}</small>", unsafe_allow_html=True)
    st.markdown("---")

    # Opsi tampilkan preprocessing
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

# ── Input teks ────────────────────────────────────────────────────────────────
# Inisialisasi session state untuk user input
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

# Perbarui user_input dari session state
user_input = st.session_state.user_input
# Gunakan session state untuk quick input
if "quick_input" in st.session_state and not user_input:
    user_input = st.session_state["quick_input"]

# ── Tombol Analisis ───────────────────────────────────────────────────────────
analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment(text: str, model_name: str):
    """
    Melakukan prediksi sentimen:
    1. Preprocessing teks
    2. Transformasi TF-IDF
    3. Prediksi dengan model yang dipilih
    Mengembalikan: (label_str, confidence_dict, clean_text, steps_dict)
    """
    if tfidf is None:
        return None, None, None, None

    model = models[model_name]
    if model is None:
        return None, None, None, None

    # Rekam tiap langkah preprocessing
    from preprocessing import (
        case_folding, cleaning, tokenization,
        stopword_removal, stemming,
    )
    steps = {}
    t1 = case_folding(text)
    steps["1. Case Folding"] = t1
    t2 = cleaning(t1)
    steps["2. Cleaning"] = t2
    t3 = tokenization(t2)
    steps["3. Tokenization"] = t3
    t4 = stopword_removal(t3)
    steps["4. Stopword Removal"] = t4
    t5 = stemming(t4)
    steps["5. Stemming"] = t5
    clean_text = " ".join(t5)

    # TF-IDF transform
    X_vec = tfidf.transform([clean_text])

    # Prediksi
    pred_int = model.predict(X_vec)[0]
    label    = LABEL_MAP[pred_int]

    # Confidence (decision_function atau predict_proba)
    conf = {}
    try:
        # LinearSVC → decision_function (bukan probabilitas)
        df_scores = model.decision_function(X_vec)[0]
        # Softmax normalization untuk visualisasi
        import numpy as np
        exp_s = [float(x) for x in df_scores]
        min_s = min(exp_s)
        shifted = [x - min_s for x in exp_s]
        total = sum(shifted) + 1e-9
        conf = {
            LABEL_MAP[i]: round(shifted[i] / total * 100, 1)
            for i in range(len(shifted))
        }
    except AttributeError:
        pass
    try:
        proba = model.predict_proba(X_vec)[0]
        conf  = {LABEL_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)}
    except AttributeError:
        pass

    return label, conf, clean_text, steps


# ─────────────────────────────────────────────────────────────────────────────
# HASIL PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("Menganalisis..."):
        time.sleep(0.4)   # UX: beri kesan proses
        label, conf, clean_text, steps = predict_sentiment(user_input.strip(), selected_model)

    if label is None:
        st.error(
            "❌ Model atau TF-IDF belum dimuat. "
            "Pastikan semua file .pkl sudah ada di direktori yang sama dengan app.py."
        )
    else:
        meta = SENTIMENT_META[label]
        st.markdown("---")
        st.markdown(f"<div class='model-badge'>{MODEL_ICONS[selected_model]} {selected_model}</div>",
                    unsafe_allow_html=True)

        # Kartu hasil
        st.markdown(
            f"<div class='result-card {meta['css']}'>"
            f"{meta['emoji']}  Sentimen: {meta['label']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Confidence score
        if conf:
            st.markdown("#### 📊 Skor Kepercayaan Model")
            order = ["positif", "netral", "negatif"]
            bar_colors = {"positif": "#66BB6A", "netral": "#90A4AE", "negatif": "#EF5350"}
            for cls in order:
                score = conf.get(cls, 0)
                st.markdown(f"<div class='conf-label'>{cls.capitalize()} — {score:.1f}%</div>",
                            unsafe_allow_html=True)
                st.progress(min(score / 100, 1.0))

        # Teks setelah preprocessing
        if show_preprocess:
            st.markdown("#### 🧹 Teks Setelah Preprocessing")
            clean_display = clean_text if clean_text.strip() else "_(teks kosong setelah preprocessing)_"
            st.markdown(
                f"<div class='preprocess-box'>{clean_display}</div>",
                unsafe_allow_html=True,
            )

        # Detail langkah preprocessing
        if show_steps and steps:
            st.markdown("#### 🔬 Detail Langkah Preprocessing")
            with st.expander("Lihat tiap langkah", expanded=True):
                for step_name, result in steps.items():
                    st.markdown(f"**{step_name}**")
                    if isinstance(result, list):
                        st.code(str(result), language="python")
                    else:
                        st.markdown(
                            f"<div class='preprocess-box'>{result}</div>",
                            unsafe_allow_html=True,
                        )

elif analyze_btn and not user_input.strip():
    st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")


# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS BATCH
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Analisis Batch (banyak teks sekaligus)", expanded=False):
    st.markdown(
        "Masukkan beberapa judul berita, **satu judul per baris**. "
        "Hasil akan ditampilkan dalam tabel."
    )
    batch_input = st.text_area(
        "Judul berita (satu per baris):",
        height=160,
        placeholder=(
            "Garuda Indonesia Rugi Rp 2,4 Triliun\n"
            "Bank BRI Laba Tumbuh 18%\n"
            "OJK Rilis Aturan Baru Reksa Dana"
        ),
    )
    batch_btn = st.button("🔍 Analisis Batch", use_container_width=True)

    if batch_btn and batch_input.strip():
        import pandas as pd
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        rows  = []
        with st.spinner(f"Menganalisis {len(lines)} teks..."):
            for line in lines:
                lbl, cf, ct, _ = predict_sentiment(line, selected_model)
                if lbl:
                    rows.append({
                        "Judul Berita"     : line,
                        "Teks Bersih"      : ct,
                        "Sentimen"         : lbl.capitalize(),
                    })

        if rows:
            df_result = pd.DataFrame(rows)
            st.dataframe(df_result, use_container_width=True)

            # Distribusi
            dist = df_result["Sentimen"].value_counts()
            st.bar_chart(dist)

            # Download CSV
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )
# Tambahkan bagian ini setelah section "ANALISIS BATCH"

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD & ANALISIS CSV
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Upload File CSV", expanded=False):
    st.markdown(
        "Upload file CSV dengan kolom berisi teks yang ingin dianalisis. "
        "Format: CSV dengan minimal satu kolom teks."
    )
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV:",
        type=["csv"],
        help="File CSV dengan kolom teks untuk dianalisis"
    )
    
    if uploaded_file is not None:
        import pandas as pd
        
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**📊 File berhasil dimuat** ({len(df_upload)} baris)")
            
            # Pilih kolom
            column_name = st.selectbox(
                "Pilih kolom yang berisi teks:",
                options=df_upload.columns
            )
            
            if st.button("🔍 Analisis CSV", use_container_width=True):
                rows = []
                progress_bar = st.progress(0)
                
                with st.spinner(f"Menganalisis {len(df_upload)} teks..."):
                    for idx, row in df_upload.iterrows():
                        text = str(row[column_name]).strip()
                        if text:
                            lbl, cf, ct, _ = predict_sentiment(text, selected_model)
                            if lbl:
                                rows.append({
                                    "Teks Asli": text,
                                    "Teks Bersih": ct,
                                    "Sentimen": lbl.capitalize(),
                                })
                        progress_bar.progress((idx + 1) / len(df_upload))
                
                if rows:
                    df_result = pd.DataFrame(rows)
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Statistik
                    st.markdown("#### 📈 Statistik Hasil")
                    dist = df_result["Sentimen"].value_counts()
                    st.bar_chart(dist)
                    
                    # Download hasil
                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil (CSV)",
                        data=csv,
                        file_name="hasil_analisis_csv.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Sentiment Analysis · Dataset CNBCI · "
    "Preprocessing: Sastrawi · TF-IDF N-gram (1,2) · "
    "Model: Naive Bayes &amp; SVM"
    "</div>",
    unsafe_allow_html=True,
)
