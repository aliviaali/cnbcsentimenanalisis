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
    # Nama file sesuai dengan yang Anda unggah
    file_names = {
        'tfidf': 'tfidf (2).pkl',
        'tools': 'preprocessing_tools (1).pkl',
        'nb_base': 'nb_baseline (2).pkl',
        'nb_opt': 'nb_optimized (2).pkl',
        'svm_base': 'svm_baseline (2).pkl',
        'svm_opt': 'svm_optimized (2).pkl'
    }
    
    assets = {}
    for key, name in file_names.items():
        try:
            with open(name, 'rb') as f:
                assets[key] = pickle.load(f)
        except Exception as e:
            st.error(f"Gagal memuat {name}: {e}")
            assets[key] = None
    return assets

# Memuat semua aset
assets = load_assets()
tfidf = assets['tfidf']
tools = assets['tools']
models = {
    'Naive Bayes (Baseline)': assets['nb_base'],
    'Naive Bayes (Optimized)': assets['nb_opt'],
    'SVM (Baseline)': assets['svm_base'],
    'SVM (Optimized)': assets['svm_opt']
}

# Ambil tool preprocessing jika ada
stemmer = tools.get('stemmer') if tools else None
stopword_remover = tools.get('stopword_remover') if tools else None

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    steps = []
    
    # 1. Teks Asli
    steps.append({"tahap": "Teks Asli", "hasil": text})
    
    # 2. Case Folding & Cleaning
    clean = text.lower()
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    steps.append({"tahap": "Cleaning & Case Folding", "hasil": clean})
    
    # 3. Tokenizing (Visualisasi saja)
    tokens = clean.split()
    steps.append({"tahap": "Tokenizing", "hasil": str(tokens)})
    
    # 4. Stopword Removal
    if stopword_remover:
        clean = stopword_remover.remove(clean)
        steps.append({"tahap": "Stopword Removal", "hasil": clean})
    
    # 5. Stemming
    if stemmer:
        clean = stemmer.stem(clean)
        steps.append({"tahap": "Stemming (Sastrawi)", "hasil": clean})
        
    return clean, steps

# --- Fungsi Penjelasan Hasil ---
def beri_penjelasan(label):
    if label.lower() == 'positif':
        return "Teks ini dikategorikan **Positif** karena model mendeteksi kata-kata yang bermakna optimis, pertumbuhan, atau berita baik bagi pasar ekonomi."
    elif label.lower() == 'negatif':
        return "Teks ini dikategorikan **Negatif** karena model menemukan istilah yang berkaitan dengan penurunan, kerugian, atau sentimen pesimis."
    else:
        return "Teks ini dikategorikan **Netral** karena kalimat bersifat informatif/faktual tanpa adanya muatan emosional atau opini yang kuat."

# --- Antarmuka Pengguna (UI) ---
st.title("📊 Dashboard Analisis Sentimen Ekonomi")
st.markdown("""
Aplikasi ini membandingkan hasil prediksi dari 4 model klasifikasi (Naive Bayes & SVM) 
untuk berita ekonomi CNBC Indonesia.
""")

# Input Teks
input_user = st.text_area("Masukkan Berita atau Opini Ekonomi:", height=150, placeholder="Contoh: IHSG hari ini menguat tajam didorong oleh sektor perbankan...")

if st.button("Mulai Analisis"):
    if not input_user.strip():
        st.warning("Mohon masukkan teks terlebih dahulu.")
    elif not tfidf:
        st.error("Error: TF-IDF Vectorizer tidak termuat.")
    else:
        # Proses Preprocessing
        clean_text, alur_proses = preprocess_text(input_user)
        
        # Kolom Kiri: Preprocessing
        st.subheader("🔍 Proses Preprocessing Teks")
        for i, step in enumerate(alur_proses):
            with st.expander(f"Langkah {i+1}: {step['tahap']}"):
                st.write(step['hasil'])
        
        st.divider()
        
        # Kolom Kanan: Hasil Prediksi
        st.subheader("🎯 Hasil Prediksi Multi-Model")
        
        # Transformasi TF-IDF
        vec = tfidf.transform([clean_text])
        
        # Tampilkan hasil dalam grid
        cols = st.columns(2)
        for i, (nama_model, model) in enumerate(models.items()):
            with cols[i % 2]:
                if model:
                    hasil = model.predict(vec)[0]
                    
                    # Desain Kartu Hasil
                    bg_color = "#d1fae5" if hasil == 'positif' else "#fee2e2" if hasil == 'negatif' else "#f3f4f6"
                    text_color = "#065f46" if hasil == 'positif' else "#991b1b" if hasil == 'negatif' else "#374151"
                    
                    st.markdown(f"""
                        <div style="background-color:{bg_color}; padding:20px; border-radius:10px; border: 1px solid {text_color}; margin-bottom:10px">
                            <h4 style="color:black; margin-top:0">{nama_model}</h4>
                            <h2 style="color:{text_color}; margin:10px 0">{hasil.upper()}</h2>
                            <p style="color:#444; font-size:0.9rem">{beri_penjelasan(hasil)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Model {nama_model} tidak tersedia.")

st.sidebar.info("""
**Tentang Aplikasi:**
- **Model:** Naive Bayes & SVM
- **Tuning:** Grid Search CV
- **Preprocessing:** Sastrawi & Custom Cleaning
""")
