import streamlit as st
import pickle
import re
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sentimen Analisis CNBC",
    page_icon="🚀",
    layout="wide"
)

# --- Fungsi Load Model & Tools ---
@st.cache_resource
def load_assets():
    file_names = {
        'tfidf': 'tfidf (2).pkl',
        'tools': 'preprocessing_tools (1).pkl',
        'nb_base': 'nb_baseline (2).pkl',
        'nb_opt': 'nb_optimized (2).pkl',
        'svm_base': 'svm_baseline (2).pkl',
        'svm_opt': 'svm_optimized (2).pkl'
    }
    
    assets = {}
    
    # Validasi instalasi library
    try:
        import sklearn
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError as e:
        st.error(f"Library penting tidak ditemukan: {e}")
        st.info("Pastikan file requirements.txt berisi: streamlit, scikit-learn, PySastrawi, dan pandas.")
        return None

    for key, name in file_names.items():
        try:
            with open(name, 'rb') as f:
                assets[key] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"File {name} tidak ditemukan. Pastikan file pkl sudah diunggah ke root folder.")
            assets[key] = None
        except Exception as e:
            st.error(f"Gagal memuat {name}: {e}")
            assets[key] = None
    return assets

# Memuat aset
assets = load_assets()

if assets:
    tfidf = assets.get('tfidf')
    tools = assets.get('tools')
    models = {
        'Naive Bayes (Baseline)': assets.get('nb_base'),
        'Naive Bayes (Optimized)': assets.get('nb_opt'),
        'SVM (Baseline)': assets.get('svm_base'),
        'SVM (Optimized)': assets.get('svm_opt')
    }

    # Ambil tools dari pkl (stopword & stemmer)
    stemmer = tools.get('stemmer') if tools else None
    stopword_remover = tools.get('stopword_remover') if tools else None

    def preprocess_text(text):
        steps = []
        steps.append({"tahap": "Teks Asli", "hasil": text})
        
        # Cleaning & Case Folding
        clean = text.lower()
        clean = re.sub(r'[^a-zA-Z\s]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        steps.append({"tahap": "Cleaning & Case Folding", "hasil": clean})
        
        # Tokenizing (Preview)
        tokens = clean.split()
        steps.append({"tahap": "Tokenizing", "hasil": str(tokens)})
        
        # Stopword Removal
        if stopword_remover:
            clean = stopword_remover.remove(clean)
            steps.append({"tahap": "Stopword Removal", "hasil": clean})
        
        # Stemming
        if stemmer:
            clean = stemmer.stem(clean)
            steps.append({"tahap": "Stemming (Sastrawi)", "hasil": clean})
            
        return clean, steps

    def beri_penjelasan(label):
        label_lower = str(label).lower()
        if 'positif' in label_lower:
            return "Teks ini dikategorikan **Positif** karena mengandung istilah yang diasosiasikan dengan pertumbuhan ekonomi, kenaikan nilai investasi, atau berita pasar yang baik."
        elif 'negatif' in label_lower:
            return "Teks ini dikategorikan **Negatif** karena model mendeteksi kata kunci terkait penurunan harga, kerugian emiten, atau sentimen pasar yang lesu."
        else:
            return "Teks ini dikategorikan **Netral** karena bersifat informatif/faktual tanpa adanya opini atau bobot emosional yang signifikan."

    # --- Tampilan Dashboard ---
    st.title("📊 Dashboard Analisis Sentimen Multi-Model")
    st.markdown("Bandingkan hasil prediksi sentimen teks ekonomi antara model **Naive Bayes** dan **SVM**.")

    input_user = st.text_area("Input Berita / Opini Ekonomi:", height=150)

    if st.button("Mulai Analisis"):
        if not input_user.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        elif not tfidf:
            st.error("Vectorizer TF-IDF tidak tersedia. Pastikan file pkl sudah benar.")
        else:
            with st.spinner('Sedang memproses teks...'):
                clean_text, alur_proses = preprocess_text(input_user)
                
                # Tampilkan Alur Preprocessing
                st.subheader("🔍 Alur Kerja Preprocessing")
                for i, step in enumerate(alur_proses):
                    with st.expander(f"Langkah {i+1}: {step['tahap']}"):
                        st.info(step['hasil'])
                
                st.divider()
                st.subheader("🎯 Hasil Prediksi Perbandingan")
                
                # Vectorization
                vec = tfidf.transform([clean_text])
                
                # Grid hasil
                cols = st.columns(2)
                for i, (nama, model) in enumerate(models.items()):
                    with cols[i % 2]:
                        if model:
                            hasil = model.predict(vec)[0]
                            # Warna Kartu
                            bg = "#d1fae5" if 'positif' in str(hasil).lower() else "#fee2e2" if 'negatif' in str(hasil).lower() else "#f3f4f6"
                            border = "#065f46" if 'positif' in str(hasil).lower() else "#991b1b" if 'negatif' in str(hasil).lower() else "#374151"
                            
                            st.markdown(f"""
                                <div style="background-color:{bg}; padding:25px; border-radius:12px; border: 2px solid {border}; margin-bottom:20px">
                                    <h4 style="color:black; margin-top:0">{nama}</h4>
                                    <h2 style="color:{border}; margin:10px 0">{str(hasil).upper()}</h2>
                                    <p style="color:#333; font-size:0.9rem">{beri_penjelasan(hasil)}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Model {nama} tidak dapat digunakan.")
else:
    st.error("Aplikasi terhenti: Library atau File Model tidak lengkap.")
