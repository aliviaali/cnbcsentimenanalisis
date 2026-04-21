import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── 1. INISIALISASI (Hanya dijalankan sekali) ────────────────────────────────
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
base_stopwords = set(stopword_factory.get_stop_words())

# ── 2. KAMUS CUSTOM (Untuk membersihkan Rp, PT, dan menjaga kata sentimen) ──
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "m", "t", "via", "redaksi", "berita"
}

# Kata penting (Gunakan kata dasar karena difilter setelah stemming)
EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "rosot", "lonjak", "tumbuh", "tingkat",
    "laba", "rugi", "untung", "saham", "kuat", "lemah", "sangat", "parah",
    "triliun", "miliar", "juta", "indonesia", "garuda", "positif", "negatif"
}

CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── 3. FUNGSI PREPROCESSING (Satu fungsi utama untuk UI/Tombol) ──────────────

def case_folding(text: str) -> str:
    return text.lower()

def cleaning(text: str) -> str:
    # Ganti tanda baca dengan spasi agar kata tidak menempel
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Hapus angka
    text = re.sub(r'\d+', ' ', text)
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text: str) -> list:
    return text.split()

def stemming(tokens: list) -> list:
    return [stemmer.stem(word) for word in tokens]

def stopword_removal(tokens: list) -> list:
    return [word for word in tokens if word not in CUSTOM_STOPWORDS and len(word) > 1]

def preprocess(text: str) -> str:
    """
    Pipeline lengkap yang dipanggil oleh tombol UI.
    Mengembalikan string agar kompatibel dengan model ML.
    """
    # Langkah 1-5
    t1 = case_folding(text)
    t2 = cleaning(t1)
    t3 = tokenization(t2)
    t4 = stemming(t3)           # Stemming dulu
    t5 = stopword_removal(t4)   # Baru buang stopword
    
    return " ".join(t5)

# ── 4. CONTOH PENGGUNAAN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_kalimat = "Laba Bersih PT Garuda Indonesia Merosot Rp5,4 Triliun, Saham GIAA Anjlok Sangat Parah!"
    hasil = preprocess(test_kalimat)
    print(f"Input : {test_kalimat}")
    print(f"Hasil : {hasil}")
