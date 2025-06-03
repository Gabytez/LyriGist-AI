from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import re
import os
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import joblib

nltk.download('punkt_tab')

app = Flask(__name__)

# Load the model
model = joblib.load('model/svm_lyrics_model_proba.joblib') 

# Label encoder model
label_encoder = joblib.load('model/label_encoder.joblib')

# Home page
@app.route('/') 
def home():
    return render_template('home.html') 

# Ensure that there is a directory to store uploaded images
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Predict page
@app.route('/predict', methods=['GET', 'POST'])
def index():
    def preprocess_lyrics(lyrics):
        def clean_lyrics(lyric):
            lyric = re.sub(r"\[.*?\]", "", lyric)
            lyric = re.sub(r"[^a-zA-Z0-9\s]", "", lyric).lower()
            lyric = lyric.replace("lyric", "", 1)
            lyric = lyric.replace("intro", "", 1)
            lyric = lyric.replace("verse", "")
            lyric = lyric.replace("chorus", "")
            lyric = lyric.replace("bridge", "", 1)
            lyric = lyric.replace("sample", "", 1)
            lyric = " ".join(lyric.split())
            return lyric

        clean_text = clean_lyrics(lyrics)
        tokens = word_tokenize(clean_text.lower())
        word2vec_model = Word2Vec.load("model/word2vec_model_3k.model")

        def sentence_vector(tokens, model):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

        vector = sentence_vector(tokens, word2vec_model)
        return vector
    
    if request.method == 'POST':
        # Get the lyrics from the form
        lyrics = request.form.get('lyrics')
        prediction_result = None
        confidence_level = None

        if lyrics:
            try:
                lyrics_vector = preprocess_lyrics(lyrics)
                lyrics_vector = np.expand_dims(lyrics_vector, axis=0)  # Make it a batch of 1
                
                predicted_class_index = model.predict(lyrics_vector)[0]
                predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(lyrics_vector)
                    confidence_level = round(100 * np.max(proba), 2)
                else:
                    confidence_level = "-"
                
                prediction_result = predicted_class_name
                print("Prediction Result: ", prediction_result)
                # prediction_result = "test"
            except Exception as e:
                print("Prediction error:", e)
                prediction_result = "Error"
                confidence_level = 0

            return render_template(
                'index.html',
                prediction_result=prediction_result,
                confidence_level=confidence_level,
                lyrics=lyrics
            )

    return render_template('index.html')

# About page
@app.route('/about') 
def about():
    return render_template('about.html') 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))