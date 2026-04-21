import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Inisialisasi
# Gunakan factory untuk mendapatkan stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
base_stopwords = set(stopword_factory.get_stop_words())

# 2. Kamus Custom (Sesuaikan dengan kata DASAR hasil stemming)
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "m", "t", "via", "giaa"
}

# Pastikan di sini adalah KATA DASAR karena kita akan filter SETELAH stemming
EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "lonjak", "rosot", "tumbuh", # 'rosot' adalah dasar dari 'merosot'
    "tingkat", "laba", "rugi", "untung", "saham", "sangat", "parah",
    "triliun", "miliar", "juta", "indonesia", "garuda"
}

CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── Fungsi-Fungsi Preprocessing ─────────────────────────────────────────────

def case_folding(text: str) -> str:
    return text.lower()

def cleaning(text: str) -> str:
    # Hapus angka dan simbol, ganti tanda baca dengan spasi agar kata tidak menempel
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text: str) -> list:
    return text.split()

def stemming(tokens: list) -> list:
    # Stemming harus dilakukan sebelum stopword removal agar kata berimbuhan jadi kata dasar
    return [stemmer.stem(word) for word in tokens]

def stopword_removal(tokens: list) -> list:
    return [word for word in tokens if word not in CUSTOM_STOPWORDS and len(word) > 1]

def preprocess_debug(text: str):
    """Fungsi untuk melihat perubahan tiap tahap secara jelas"""
    t1 = case_folding(text)
    print(f"1. Case Folding: {t1}")
    
    t2 = cleaning(t1)
    print(f"2. Cleaning: {t2}")
    
    t3 = tokenization(t2)
    print(f"3. Tokenization: {t3}")
    
    t4 = stemming(t3)
    print(f"4. Stemming: {t4}") # 'merosot' harusnya jadi 'rosot' di sini
    
    t5 = stopword_removal(t4)
    print(f"5. Stopword Removal: {t5}")
    
    return " ".join(t5)

# Uji Coba
kalimat = "Laba Bersih PT Garuda Indonesia Merosot Rp5,4 Triliun, Saham GIAA Anjlok Sangat Parah!"
hasil_akhir = preprocess_debug(kalimat)
print(f"\nHasil Akhir String: {hasil_akhir}")
