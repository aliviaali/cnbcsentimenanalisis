import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── 1. Inisialisasi ───────────────────────────────────────────────────────────
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
# Ambil base stopwords dan ubah menjadi set agar bisa dikurangi/ditambah
base_stopwords = set(stopword_factory.get_stop_words())

# ── 2. Kebaharuan: Kamus Sampah & Pengecualian ────────────────────────────────
# Kata yang pasti dibuang (Sampah Finansial)
ADDITIONAL_STOPWORDS = {"rp", "pt", "tbk", "idr", "usd", "giaa", "jt", "m", "t"}

# Kata yang HARUS ADA (Jangan dihapus walau masuk daftar stopword)
# Tips: Gunakan kata dasar karena akan dieksekusi SETELAH stemming
EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "rosot", "lonjak", "tumbuh", "tingkat",
    "laba", "rugi", "untung", "saham", "kuat", "lemah", "sangat", "parah",
    "triliun", "miliar", "juta", "indonesia", "garuda"
}

# Gabungkan: (Base + Sampah) - Kata Penting
CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

# ── 3. Fungsi-Fungsi Preprocessing ─────────────────────────────────────────────

def case_folding(text: str) -> str:
    return text.lower()

def cleaning(text: str) -> str:
    # Ganti tanda baca dengan spasi agar kata tidak nempel (ex: "rugi,saham" -> "rugi saham")
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Hapus angka
    text = re.sub(r'\d+', ' ', text)
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text: str) -> list:
    return text.split()

def stemming(tokens: list) -> list:
    # Mengubah kata berimbuhan menjadi kata dasar
    return [stemmer.stem(word) for word in tokens]

def stopword_removal(tokens: list) -> list:
    # Menghapus kata berdasarkan kamus CUSTOM_STOPWORDS
    return [word for word in tokens if word not in CUSTOM_STOPWORDS and len(word) > 1]

def preprocess(text: str) -> str:
    """Pipeline Utama dengan urutan yang sudah diperbaiki"""
    t1 = case_folding(text)
    t2 = cleaning(t1)
    t3 = tokenization(t2)
    # KEBAHARUAN: Stemming dilakukan sebelum Stopword Removal
    t4 = stemming(t3) 
    t5 = stopword_removal(t4)
    return " ".join(t5)

# ── 4. Uji Coba ───────────────────────────────────────────────────────────────
kalimat = "Laba Bersih PT Garuda Indonesia Merosot Rp5,4 Triliun, Saham GIAA Anjlok Sangat Parah!"
hasil = preprocess(kalimat)

print(f"Kalimat Asli : {kalimat}")
print(f"Hasil Akhir  : {hasil}")
