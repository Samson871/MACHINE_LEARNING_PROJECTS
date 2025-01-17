import os
from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model/trained_model.sav'  # Replace with your actual path

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model
model = load_model()

# Load tokenizer
TOKENIZER_PATH = 'model/vectorizer.pickle'
def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as file:
       tokenizer = pickle.load(file)
    return tokenizer
tokenizer = load_tokenizer()


# Preprocessing function
def preprocess_text(text):
    if not text:
        return ""
    
    # Remove special characters, links, mentions, and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    
    # Tokenize text
    tokens = text.lower().split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return " ".join(stemmed_tokens)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)

    # Transform text using TfidfVectorizer
    transformed_text = tokenizer.transform([preprocessed_text])  # Vectorizer, not tokenizer
    
    # Make the prediction
    prediction = model.predict(transformed_text)
    
    sentiment_label = "positive"
    if prediction[0] < 0.4:
        sentiment_label = "negative"
    elif prediction[0] < 0.6:
        sentiment_label = "neutral"
    
    return sentiment_label



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Initially set to None
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_sentiment(text)  # This should return a sentiment label
        
    return render_template('index.html', prediction=prediction)  # Pass prediction to template


if __name__ == '__main__':
    nltk.download('stopwords')
    app.run(debug=True)
