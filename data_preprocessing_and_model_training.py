import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
data = pd.read_csv('C:/Users/balav/OneDrive/Desktop/BALU/COLLEGE/SEM 9/Intelligent Cyber Security Lab/ex 1_email spam filtering using ML/spam.csv', encoding='latin1')

# Drop unnecessary columns
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Remove duplicates
data = data.drop_duplicates(keep='first')

# Encode labels
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
data['v2'] = data['v2'].apply(preprocess_text)

# Split dataset
X = data['v2']
y = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(clf, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")
