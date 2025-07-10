import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download stopwords (only once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
df = pd.read_csv('email.csv')  # Rename if needed

print("Sample data:")
print(df.head())

# Assuming columns: 'EmailText', 'Label'
X = df['Message']
y = df['Category']


# -----------------------------
# Step 2: Preprocess the text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

X = X.apply(clean_text)

# -----------------------------
# Step 3: Vectorize the text
# -----------------------------
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# -----------------------------
# Step 4: Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 5: Train the model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Evaluate the model
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




import joblib

# Save model and vectorizer
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
