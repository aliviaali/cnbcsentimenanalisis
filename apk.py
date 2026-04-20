# app.py
# =============================================================================
# Streamlit App — Sentiment Analysis Berita CNBC Indonesia
# Model  : Naive Bayes | SVM | Naive Bayes Optimized | SVM Optimized
# Deploy : streamlit run app.py
# =============================================================================

import pickle
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

/* Sentiment gauge container */
.sentiment-gauge-container {
    display: flex;
    justify-content: space-around;
    gap: 1rem;
    margin: 1.5rem 0;
}

.sentiment-gauge-item {
    flex: 1;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
}
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
    "positif": {"emoji": "😊", "label": "POSITIF", "css": "positif", "color": "#66BB6A"},
    "netral" : {"emoji": "😐", "label": "NETRAL",  "css": "netral", "color": "#90A4AE"},
    "negatif": {"emoji": "😞", "label": "NEGATIF", "css": "negatif", "color": "#EF5350"},
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
user_input = st.text_area(
    label="✏️ Judul Berita",
    placeholder="Contoh: Bank BRI Catat Laba Bersih Tumbuh 18% pada Kuartal III 2024",
    height=110,
    help="Masukkan satu atau beberapa kalimat judul berita berbahasa Indonesia.",
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
        user_input = example_texts["😊 Positif"]
        st.session_state["quick_input"] = user_input
with col2:
    if st.button("😞 Contoh Negatif", use_container_width=True):
        user_input = example_texts["😞 Negatif"]
        st.session_state["quick_input"] = user_input
with col3:
    if st.button("😐 Contoh Netral", use_container_width=True):
        user_input = example_texts["😐 Netral"]
        st.session_state["quick_input"] = user_input

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
    pred_int = model.predict(X_vec)
    label    = LABEL_MAP[pred_int]

    # Confidence (decision_function atau predict_proba)
    conf = {}
    try:
        # LinearSVC → decision_function (bukan probabilitas)
        df_scores = model.decision_function(X_vec)
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
        proba = model.predict_proba(X_vec)
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

        # Confidence score dengan Grafik Pie Chart
        if conf:
            st.markdown("#### 📊 Skor Kepercayaan Model")
            
            # Buat 2 kolom: Pie Chart + Progress Bars
            col_chart, col_bars = st.columns([1.2, 1])
            
            with col_chart:
                # Pie Chart dengan Plotly
                sentiments = list(conf.keys())
                scores = list(conf.values())
                colors_list = [SENTIMENT_META[s] ["color"] for s in sentiments]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[s.capitalize() for s in sentiments],
                    values=scores,
                    marker=dict(colors=colors_list),
                    textposition='inside',
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Skor: %{value:.1f}%<extra></extra>',
                )])
                
                fig_pie.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    font=dict(size=11),
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_bars:
                # Progress bars vertikal
                st.markdown("**Persentase:**")
                order = ["positif", "netral", "negatif"]
                for cls in order:
                    score = conf.get(cls, 0)
                    color = SENTIMENT_META[cls] ["color"]
                    st.markdown(
                        f"<div class='conf-label'>"
                        f"<span style='color:{color}; font-weight:bold;'>"
                        f"{cls.capitalize()}</span> — {score:.1f}%"
                        f"</div>",
                        unsafe_allow_html=True
                    )
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
        "Hasil akan ditampilkan dalam tabel dan grafik."
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

            # Distribusi dengan Grafik yang lebih menarik
            st.markdown("#### 📈 Distribusi Sentimen")
            dist = df_result["Sentimen"].value_counts()
            
            # Bar Chart dengan Plotly
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=dist.index,
                    y=dist.values,
                    marker=dict(
                        color=[SENTIMENT_META[s.lower()] ["color"] for s in dist.index]
                    ),
                    text=dist.values,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Jumlah: %{y}<extra></extra>',
                )
            ])
            
            fig_bar.update_layout(
                xaxis_title="Sentimen",
                yaxis_title="Jumlah Berita",
                height=350,
                showlegend=False,
                hovermode='x unified',
                margin=dict(l=40, r=40, t=40, b=40),
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Pie Chart untuk batch
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Persentase Sentimen:**")
                for sentiment in ["Positif", "Netral", "Negatif"]:
                    count = len(df_result[df_result["Sentimen"] == sentiment])
                    pct = (count / len(df_result)) * 100
                    color = SENTIMENT_META[sentiment.lower()] ["color"]
                    st.markdown(
                        f"<div style='padding:0.5rem; margin:0.3rem 0; "
                        f"background:{color}20; border-left:4px solid {color}; border-radius:4px;'>"
                        f"<b>{sentiment}</b>: {count} ({pct:.1f}%)"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Donut Chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=[s.capitalize() for s in dist.index],
                    values=dist.values,
                    hole=0.4,
                    marker=dict(
                        colors=[SENTIMENT_META[s.lower()] ["color"] for s in dist.index]
                    ),
                    textposition='inside',
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<extra></extra>',
                )])
                
                fig_donut.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                )
                
                st.plotly_chart(fig_donut, use_container_width=True)

            # Download CSV
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )

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
