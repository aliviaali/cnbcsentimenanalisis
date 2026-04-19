import os
import pickle
import re
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- 1. Load Models & Tools ---
def load_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load tools
tfidf = load_pickle('tfidf (2).pkl')
tools = load_pickle('preprocessing_tools (1).pkl')
stemmer = tools.get('stemmer') if tools else None
stopword_remover = tools.get('stopword_remover') if tools else None

# Load Models
models = {
    'NB_Baseline': load_pickle('nb_baseline (2).pkl'),
    'NB_Optimized': load_pickle('nb_optimized (2).pkl'),
    'SVM_Baseline': load_pickle('svm_baseline (2).pkl'),
    'SVM_Optimized': load_pickle('svm_optimized (2).pkl')
}

# --- 2. Preprocessing Logic ---
def preprocess_step_by_step(text):
    steps = []
    
    # Raw
    current_text = text
    steps.append({"step": "Teks Asli", "result": current_text})
    
    # Case Folding & Cleaning
    current_text = current_text.lower()
    current_text = re.sub(r'[^a-zA-Z\s]', '', current_text)
    steps.append({"step": "Case Folding & Cleaning", "result": current_text})
    
    # Tokenizing
    tokens = current_text.split()
    steps.append({"step": "Tokenizing", "result": ", ".join(tokens)})
    
    # Stopword Removal
    if stopword_remover:
        current_text = stopword_remover.remove(current_text)
        tokens = current_text.split()
        steps.append({"step": "Stopword Removal", "result": current_text})
    
    # Stemming
    if stemmer:
        current_text = stemmer.stem(current_text)
        steps.append({"step": "Stemming (Sastrawi)", "result": current_text})
        
    return current_text, steps

# --- 3. Explanation Logic ---
def get_explanation(sentiment, model_name):
    explanations = {
        "positif": "Teks mengandung kata-kata kunci yang diasosiasikan model dengan sentimen mendukung, apresiasi, atau pertumbuhan ekonomi.",
        "negatif": "Model mendeteksi pola kata yang merujuk pada kerugian, kritik, atau sentimen pesimis dalam data latih.",
        "netral": "Teks bersifat informatif atau objektif, tidak mengandung bobot emosional yang cukup untuk dikategorikan positif/negatif."
    }
    return explanations.get(sentiment.lower(), "Analisis pola kata berdasarkan bobot model.")

# --- 4. Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = None
    preprocessing_steps = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('text_input', '')
        
        if input_text and tfidf:
            # Preprocess
            clean_text, preprocessing_steps = preprocess_step_by_step(input_text)
            
            # Vectorize
            vec_text = tfidf.transform([clean_text])
            
            prediction_results = []
            for name, model in models.items():
                if model:
                    pred = model.predict(vec_text)[0]
                    
                    # Try to get probability if available
                    prob = ""
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec_text)[0]
                        max_idx = proba.argmax()
                        prob = f"{proba[max_idx]*100:.2f}% Confidence"
                    
                    prediction_results.append({
                        "model": name.replace('_', ' '),
                        "sentiment": pred,
                        "confidence": prob,
                        "explanation": get_explanation(pred, name)
                    })

    return render_template('index.html', 
                           results=prediction_results, 
                           steps=preprocessing_steps,
                           input_text=input_text)

# --- 5. HTML Template (Inline for Single File Mandate) ---
# Note: In a real app, this goes into templates/index.html
@app.before_first_request
def create_template():
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen">
    <nav class="bg-blue-600 p-4 text-white shadow-lg">
        <div class="container mx-auto flex items-center">
            <i class="fas fa-brain mr-3 text-2xl"></i>
            <h1 class="text-xl font-bold">Analisis Sentimen Multi-Model</h1>
        </div>
    </nav>

    <div class="container mx-auto py-8 px-4">
        <div class="max-w-4xl mx-auto">
            <!-- Input Section -->
            <div class="bg-white rounded-xl shadow-md p-6 mb-8">
                <h2 class="text-lg font-semibold mb-4 text-gray-700">Input Teks Berita / Opini Ekonomi</h2>
                <form method="POST">
                    <textarea 
                        name="text_input" 
                        rows="4" 
                        class="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                        placeholder="Masukkan teks di sini..."
                    >{{ input_text }}</textarea>
                    <button type="submit" class="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg transition flex items-center">
                        <i class="fas fa-search mr-2"></i> Analisis Sekarang
                    </button>
                </form>
            </div>

            {% if steps %}
            <!-- Preprocessing Steps -->
            <div class="bg-white rounded-xl shadow-md p-6 mb-8">
                <h2 class="text-lg font-semibold mb-4 text-gray-700"><i class="fas fa-cogs mr-2 text-blue-500"></i> Alur Preprocessing</h2>
                <div class="space-y-3">
                    {% for step in steps %}
                    <div class="flex items-start">
                        <div class="flex-shrink-0 h-6 w-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold mt-1">
                            {{ loop.index }}
                        </div>
                        <div class="ml-4">
                            <p class="text-sm font-bold text-gray-600">{{ step.step }}</p>
                            <p class="text-sm text-gray-500 italic">"{{ step.result }}"</p>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if results %}
            <!-- Results Cards -->
            <h2 class="text-xl font-bold mb-6 text-gray-800">Hasil Perbandingan Model</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {% for res in results %}
                <div class="bg-white rounded-xl shadow-md border-t-4 {% if res.sentiment == 'positif' %}border-green-500{% elif res.sentiment == 'negatif' %}border-red-500{% else %}border-gray-400{% endif %} p-6">
                    <div class="flex justify-between items-start mb-4">
                        <h3 class="font-bold text-gray-700">{{ res.model }}</h3>
                        <span class="px-3 py-1 rounded-full text-xs font-bold uppercase 
                            {% if res.sentiment == 'positif' %}bg-green-100 text-green-700{% elif res.sentiment == 'negatif' %}bg-red-100 text-red-700{% else %}bg-gray-100 text-gray-700{% endif %}">
                            {{ res.sentiment }}
                        </span>
                    </div>
                    <p class="text-sm text-gray-600 mb-4">{{ res.explanation }}</p>
                    <div class="text-xs text-gray-400 flex items-center">
                        <i class="fas fa-check-circle mr-1"></i> {{ res.confidence }}
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </div>
    
    <footer class="mt-12 py-6 bg-gray-100 text-center text-gray-500 text-sm">
        <p>&copy; 2024 Analisis Sentimen Ekonomi - Naive Bayes & SVM Comparison</p>
    </footer>
</body>
</html>
        ''')

if __name__ == '__main__':
    app.run(debug=True)
