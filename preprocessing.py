import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi
stemmer = StemmerFactory().create_stemmer()
base_stopwords = set(StopWordRemoverFactory().get_stop_words())

ADDITIONAL_STOPWORDS = {"rp", "pt", "tbk", "idr", "usd", "jt", "m", "t"}
EXCLUDE_FROM_STOPWORDS = {
    "naik", "turun", "anjlok", "rosot", "lonjak", "tumbuh", "tingkat",
    "laba", "rugi", "untung", "saham", "kuat", "lemah", "sangat", "parah",
    "triliun", "miliar", "juta", "indonesia", "garuda"
}
CUSTOM_STOPWORDS = (base_stopwords | ADDITIONAL_STOPWORDS) - EXCLUDE_FROM_STOPWORDS

def preprocess(text: str) -> str:
    # 1. Case Folding
    text = text.lower()
    # 2. Cleaning
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Tokenization
    tokens = text.split()
    # 4. Stemming
    stemmed = [stemmer.stem(word) for word in tokens]
    # 5. Stopword Removal
    final = [word for word in stemmed if word not in CUSTOM_STOPWORDS and len(word) > 1]
    return " ".join(final)
