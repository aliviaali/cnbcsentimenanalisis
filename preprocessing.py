import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Inisialisasi Sastrawi ─────────────────────────────────────────────────────
stemmer_factory  = StemmerFactory()
stemmer          = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
base_stopwords   = set(stopword_factory.get_stop_words())

# ── Konfigurasi Kamus ────────────────────────────────────────────────────────
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "redaksi", 
    "berita", "halaman", "tersebut", "yakni", "yaitu", "via",
    "iii", "ii", "iv", "v", "www", "https", "com", "ig", "twitter", "indonesia"
}

# Penting: Kata sentimen ekonomi jangan dihapus!
EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "rosot", "lonjak", "tumbuh", "tingkat",
    "laba", "rugi", "untung", "defisit", "surplus", "saham", "kuat", 
    "lemah", "sangat", "parah", "triliun", "miliar", "juta", "berita,
}

CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

MANUAL_STEM_RULES = {
    "merosot": "rosot",
    "melonjak": "lonjak",
    "meningkat": "tingkat",
    "menurun": "turun",
    "ekonomiindonesia": "ekonomi"
}

# ── Fungsi Preprocessing ──────────────────────────────────────────────────────

def case_folding(text: str) -> str:
    return text.lower()

def cleaning(text: str) -> str:
    # 1. Hapus URL secara utuh sebelum tanda baca dihapus
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Hapus Mention (@) dan Hashtag (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # 3. Hapus Angka
    text = re.sub(r'\d+', ' ', text)
    # 4. Ganti tanda baca dengan spasi
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # 5. Normalisasi spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text: str) -> list:
    return text.split()

def stemming(tokens: list) -> list:
    """Stemming dengan prioritas Manual Rules baru Sastrawi."""
    return [MANUAL_STEM_RULES[word] if word in MANUAL_STEM_RULES else stemmer.stem(word) for word in tokens]

def stopword_removal(tokens: list) -> list:
    """Menghapus kata tidak bermakna dan token terlalu pendek (<3 karakter)."""
    return [word for word in tokens if word not in CUSTOM_STOPWORDS and len(word) > 2]

def preprocess(text: str) -> str:
    """
    Pipeline Preprocessing:
    Case Folding -> Cleaning -> Tokenization -> Stemming -> Stopword Removal -> Rejoining
    """
    if not text: return ""
    
    t1 = case_folding(text)
    t2 = cleaning(t1)
    t3 = tokenization(t2)
    t4 = stemming(t3)
    t5 = stopword_removal(t4)
    
    return " ".join(t5)
