import streamlit as st
import string
import nltk
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# Helper: Clean and preprocess input text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# -----------------------------
# Load trained model and vectorizer
# -----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    import joblib
    model = joblib.load("phishing_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📧 Phishing Email Detector")
st.write("Paste an email below to check if it's **phishing** or **legitimate**.")

user_input = st.text_area("Email content:", height=250)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_input = clean_text(user_input)
        vec_input = vectorizer.transform([clean_input])
        prediction = model.predict(vec_input)[0]

        if prediction == 1:
            st.error("⚠️ This email is **likely phishing**.")
        else:
            st.success("✅ This email is **likely legitimate**.")
