import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Inisialisasi Sastrawi ─────────────────────────────────────────────────────
stemmer_factory  = StemmerFactory()
stemmer          = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
base_stopwords   = set(stopword_factory.get_stop_words())

# ── Kata yang HARUS dikecualikan dari stopword (penting untuk berita keuangan) ─
# 1. Tambahkan daftar kata "sampah" yang sering muncul di berita tapi tidak ada di stopword dasar
ADDITIONAL_STOPWORDS = {
    "rp", "pt", "tbk", "idr", "usd", "jt", "jt", "m", "t", "redaksi", 
    "berita", "halaman", "tersebut", "yakni", "yaitu", "via"
}

# 2. Daftar kata penting yang TIDAK BOLEH dihapus (tetap gunakan daftar Anda)
EXCLUDE_FROM_STOPWORDS = {
    # Arah / perubahan
    "naik", "turun", "anjlok", "melonjak", "merosot", "tumbuh",
    "meningkat", "menurun", "stagnan", "rebound", "koreksi",
    # Kondisi keuangan
    "laba", "rugi", "untung", "defisit", "surplus", "bangkrut",
    "pailit", "likuid", "solven",
    # Kata penting konteks
    "baru", "perdana", "pertama", "terakhir", "terbesar", "terkecil",
    "tertinggi", "terendah", "rekor", "normal", "positif", "negatif",
    "baik", "buruk", "kuat", "lemah",
    # Kata keuangan umum
    "saham", "ihsg", "rupiah", "dolar", "inflasi", "deflasi",
    "suku", "bunga", "investasi", "ekspor", "impor", "reksa", "dana",
    # Kata temporal & konteks
    "mulai", "bulan", "depan", "hari", "minggu", "tahun", "kuartal",
    "setelah", "sebelum", "saat", "ketika", "sementara", "terkait", "ojk",
    "rilis", "aturan", "laporan", "efektif",
    # Organisasi & institusi
    "bank", "bri", "garuda", "indonesia", "cnbc",
    # Kata aksi & status
    "catat", "cetak", "catatkan", "raih", "capai", "peroleh",
    # Kata tambahan untuk sentimen
    "kuat", "lemah", "solid", "rapuh", "sehat", "sakit",
    "sangat", "amat", "sekali",
    # Angka & kuantitas
    "persen", "triliun", "miliar", "juta",
}

# 3. Logika Final: Gabungkan stopword dasar dengan sampah baru, 
# lalu buang kata-kata yang dianggap penting (EXCLUDE).
CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── MANUAL STEMMING RULES (untuk kata yang tidak ditangani Sastrawi) ────────
MANUAL_STEM_RULES = {
    "merosot": "rosot",
    "melonjak": "lonjak",
    "meningkat": "tingkat",
    "menurun": "turun",
    "anjlok": "anjlok",      # sudah bentuk dasar
    "tumbuh": "tumbuh",      # sudah bentuk dasar
    "stagnan": "stagnan",    # sudah bentuk dasar
    "rebound": "rebound",    # sudah bentuk dasar
    "koreksi": "koreksi",    # sudah bentuk dasar
    "laba": "laba",          # sudah bentuk dasar
    "rugi": "rugi",          # sudah bentuk dasar
    "untung": "untung",      # sudah bentuk dasar
    "defisit": "defisit",    # sudah bentuk dasar
    "surplus": "surplus",    # sudah bentuk dasar
    "bangkrut": "bangkrut",  # sudah bentuk dasar
    "pailit": "pailit",      # sudah bentuk dasar
    "likuid": "likuid",      # sudah bentuk dasar
    "solven": "solven",      # sudah bentuk dasar
}

# ── Fungsi Preprocessing ──────────────────────────────────────────────────────

def case_folding(text: str) -> str:
    """Mengubah semua huruf menjadi lowercase."""
    return text.lower()


def cleaning(text: str) -> str:
    """Menghapus angka, tanda baca, dan karakter non-alfabet."""
    text = re.sub(r'\d+', '', text)                    # hapus angka
    text = re.sub(r'[^\w\s]', '', text)                # hapus tanda baca
    text = re.sub(r'_', '', text)                      # hapus underscore
    text = re.sub(r'\s+', ' ', text).strip()           # normalisasi spasi
    return text


def tokenization(text: str) -> list:
    """Memecah teks menjadi list token."""
    return text.split()


def stopword_removal(tokens: list) -> list:
    """
    Menghapus stopword menggunakan kamus custom
    (Sastrawi base - kata penting keuangan).
    """
    return [word for word in tokens if word not in CUSTOM_STOPWORDS]


def stemming(tokens: list) -> list:
    """Melakukan stemming dengan manual rules + Sastrawi."""
    result = []
    for word in tokens:
        # Cek manual rules dulu
        if word in MANUAL_STEM_RULES:
            result.append(MANUAL_STEM_RULES[word])
        else:
            # Jika tidak ada di manual rules, gunakan Sastrawi
            result.append(stemmer.stem(word))
    return result


def preprocess(text: str) -> str:
    """Pipeline lengkap preprocessing, mengembalikan string bersih."""
    t1 = case_folding(text)
    t2 = cleaning(t1)
    t3 = tokenization(t2)
    t4 = stopword_removal(t3)
    t5 = stemming(t4)
    return " ".join(t5)
