# preprocessing.py
# =============================================================================
# Modul preprocessing teks Bahasa Indonesia
# Digunakan oleh: sentiment_analysis_colab.py  &  app.py (Streamlit)
# =============================================================================

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ─── Inisialisasi (lazy singleton agar tidak reload tiap call) ───────────────
_stemmer      = None
_stopword_set = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def _get_stopwords():
    global _stopword_set
    if _stopword_set is None:
        base = set(StopWordRemoverFactory().get_stop_words())
        custom = {
            # Partikel informal
            "yuk", "nih", "sih", "deh", "dong", "kok", "loh", "lah", "kan",
            # Singkatan dokumen
            "tsb", "svp", "ybs", "dkk", "dll", "dsb", "dst",
            # Noise domain berita
            "cnbc", "foto", "video", "baca", "simak", "lihat", "klik",
            "artikel", "berita", "news", "breaking", "update", "terkini",
            "diketahui", "tersebut",
            # Nama hari & bulan (tidak membawa makna sentimen)
            "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
            "januari", "februari", "maret", "april", "mei", "juni",
            "juli", "agustus", "september", "oktober", "november", "desember",
        }
        _stopword_set = base | custom
    return _stopword_set


# ─── Langkah-langkah preprocessing ─────────────────────────────────────────

def case_folding(text: str) -> str:
    """Ubah teks ke huruf kecil (lowercase)."""
    return text.lower()


def cleaning(text: str) -> str:
    """
    Bersihkan teks:
    - Hapus URL, mention, hashtag
    - Hapus angka dan satuan persentase
    - Ganti tanda baca dengan spasi (agar kata tidak tersambung)
    - Normalkan spasi berlebih
    """
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+[.,]?\d*\s*%?", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def tokenization(text: str) -> list:
    """Pecah kalimat menjadi token; buang token < 2 karakter."""
    return [tok for tok in text.split() if len(tok) >= 2]


def stopword_removal(tokens: list) -> list:
    """Hapus stopword dari daftar token."""
    sw = _get_stopwords()
    return [tok for tok in tokens if tok not in sw]


def stemming(tokens: list) -> list:
    """Ubah setiap token ke bentuk kata dasar (Sastrawi)."""
    s = _get_stemmer()
    return [s.stem(tok) for tok in tokens]


def preprocess(text: str) -> str:
    """
    Pipeline lengkap: case_folding → cleaning → tokenization
                      → stopword_removal → stemming.
    Mengembalikan string untuk kompatibilitas TfidfVectorizer.
    """
    text   = case_folding(text)
    text   = cleaning(text)
    tokens = tokenization(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)
