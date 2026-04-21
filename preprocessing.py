import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Inisialisasi Sastrawi ─────────────────────────────────────────────────────
stemmer_factory  = StemmerFactory()
stemmer          = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
base_stopwords   = set(stopword_factory.get_stop_words())

# ── Kebaharuan 1: Daftar Sampah & Pengecualian ────────────────────────────────
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "m", "t", "redaksi", 
    "berita", "halaman", "tersebut", "yakni", "yaitu", "via",
    "iii", "ii", "iv", "v"  # Tambahan: Angka romawi kuartal
}

EXCLUDE_FROM_STOPWORDS = {
    # Gunakan KATA DASAR di sini karena kita filter SETELAH stemming
    "naik", "turun", "anjlok", "rosot", "lonjak", "tumbuh", "tingkat",
    "laba", "rugi", "untung", "defisit", "surplus", "saham", "kuat", 
    "lemah", "sangat", "parah", "triliun", "miliar", "juta", "indonesia", "garuda"
}

CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── Kebaharuan 2: Manual Stemming Rules ──────────────────────────────────────
MANUAL_STEM_RULES = {
    "merosot": "rosot",
    "melonjak": "lonjak",
    "meningkat": "tingkat",
    "menurun": "turun"
}

# ── Fungsi Preprocessing ──────────────────────────────────────────────────────

def case_folding(text: str) -> str:
    return text.lower()

def cleaning(text: str) -> str:
    # Ganti tanda baca dengan spasi (mencegah kata menempel)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Hapus angka
    text = re.sub(r'\d+', ' ', text)
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text: str) -> list:
    return text.split()

def stemming(tokens: list) -> list:
    """Melakukan stemming dengan manual rules + Sastrawi."""
    result = []
    for word in tokens:
        if word in MANUAL_STEM_RULES:
            result.append(MANUAL_STEM_RULES[word])
        else:
            result.append(stemmer.stem(word))
    return result

def stopword_removal(tokens: list) -> list:
    """Filter kata setelah mereka menjadi bentuk dasar (Stemmed)."""
    # Tambahan len(word) > 2 untuk hapus sisa karakter sampah seperti 'rp', 'pt'
    return [word for word in tokens if word not in CUSTOM_STOPWORDS and len(word) > 2]

def preprocess(text: str) -> str:
    """Pipeline dengan urutan yang sudah diperbaiki (Stemming sebelum Stopword)."""
    t1 = case_folding(text)
    t2 = cleaning(t1)
    t3 = tokenization(t2)
    # PERBAIKAN: Stemming dulu agar kata dasar bisa difilter dengan akurat
    t4 = stemming(t3)
    t5 = stopword_removal(t4)
    return " ".join(t5)
