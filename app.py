from flask import Flask, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load the model and vectorizer
clf = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

@app.route('/classify', methods=['POST'])
def classify_email():
    email = request.json['email']
    email_processed = preprocess_text(email)
    email_vectorized = vectorizer.transform([email_processed])
    prediction = clf.predict(email_vectorized)
    return jsonify({'label': 'spam' if prediction[0] == 1 else 'not spam'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
