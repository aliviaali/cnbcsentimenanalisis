import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Inisialisasi
stemmer = StemmerFactory().create_stemmer()
base_stopwords = set(StopWordRemoverFactory().get_stop_words())

# 2. Kamus Custom
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "m", "t", "redaksi", 
    "berita", "halaman", "tersebut", "yakni", "yaitu", "via", "siap"
}

EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "melonjak", "merosot", "tumbuh",
    "tingkat", "laba", "rugi", "untung", "defisit", "surplus",
    "saham", "rupiah", "dolar", "kuat", "lemah", "sangat", "parah",
    "triliun", "miliar", "juta", "persen"
}

CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── Fungsi-Fungsi yang Diperbaiki ─────────────────────────────────────────────

def cleaning(text: str) -> str:
    # Hapus URL & Mention (Penting jika data dari Twitter/Social Media)
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca (diganti spasi agar kata tidak menempel)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_v2(text: str):
    # 1. Case Folding
    text = text.lower()
    
    # 2. Cleaning
    text = cleaning(text)
    
    # 3. Tokenization & 4. Stemming (Digabung agar efisien)
    # Kita stem dulu SEMUA kata
    raw_tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in raw_tokens]
    
    # 5. Stopword Removal (Dilakukan di akhir agar kata dasar yang tersaring)
    final_tokens = [word for word in stemmed_tokens if word not in CUSTOM_STOPWORDS and len(word) > 1]
    
    return {
        "original": text,
        "tokens": final_tokens,
        "result": " ".join(final_tokens)
    }

# Uji Coba
kalimat = "Laba Bersih PT Garuda Indonesia Merosot Rp5,4 Triliun, Saham GIAA Anjlok Sangat Parah!"
hasil = preprocess_v2(kalimat)
print(hasil['tokens'])
