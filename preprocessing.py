import streamlit as st
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. SETTINGS & CSS (KEBAHARUAN: WARNA TOMBOL) ---
st.set_page_config(page_title="Analisis Sentimen", layout="centered")

st.markdown("""
    <style>
    /* Warna tombol saat ditekan (active) dan hover */
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: black;
        border-radius: 10px;
    }
    div.stButton > button:active {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    div.stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_stdio=True)

# --- 2. INISIALISASI SASTRAWI ---
@st.cache_resource
def load_nlp_tools():
    stemmer = StemmerFactory().create_stemmer()
    base_stopwords = set(StopWordRemoverFactory().get_stop_words())
    
    ADDITIONAL = {"rp", "pt", "tbk", "idr", "usd"}
    EXCLUDE = {"naik", "turun", "anjlok", "rosot", "laba", "rugi", "saham", "garuda", "indonesia"}
    custom_stopwords = (base_stopwords | ADDITIONAL) - EXCLUDE
    
    return stemmer, custom_stopwords

stemmer, CUSTOM_STOPWORDS = load_nlp_tools()

# --- 3. FUNGSI PREPROCESSING ---
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    # Stemming dulu baru stopword agar bersih
    stemmed = [stemmer.stem(w) for w in tokens]
    final = [w for w in stemmed if w not in CUSTOM_STOPWORDS and len(w) > 1]
    return " ".join(final)

# --- 4. LOGIKA TOMBOL (PERBAIKAN: SESSION STATE) ---
if 'input_teks' not in st.session_state:
    st.session_state.input_teks = ""

def set_text(example):
    st.session_state.input_teks = example

st.title("📰 Analisis Sentimen Berita")

# Baris tombol contoh
col1, col2 = st.columns(2)
with col1:
    if st.button("🔖 Contoh Positif"):
        set_text("Laba Bank BRI naik pesat tahun ini")
with col2:
    # Tombol Negatif dengan key unik
    if st.button("😞 Contoh Negatif", key="btn_neg"):
        set_text("Garuda Indonesia rugi triliunan saham anjlok parah")

# Input teks terikat ke session_state
judul_berita = st.text_input("Masukkan judul berita:", value=st.session_state.input_teks)

if st.button("Analisis Sekarang", type="primary"):
    if judul_berita:
        hasil_bersih = preprocess(judul_berita)
        st.write("### Hasil Preprocessing:")
        st.success(f"Result: {hasil_bersih}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")
