# fake-news-detection-

1. Import Libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import xgboost as xgb
import joblib
2. Download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
3. Load Dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.shape)
df.head()
4. Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)
5. TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(df["clean_text"])

y = df["label"]
6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
7. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("LR Accuracy:",
      accuracy_score(y_test, lr_pred))
8. Naive Bayes
nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("NB Accuracy:",
      accuracy_score(y_test, nb_pred))
9. Support Vector Machine (Best Model)
svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("SVM Accuracy:",
      accuracy_score(y_test, svm_pred))
10. XGBoost
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))
11. Save Best Model
joblib.dump(
    svm_model,
    "best_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)
12. Streamlit App (app.py)
import streamlit as st
import joblib

model = joblib.load(
    "best_model.pkl"
)

vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

st.title("Fake News Detection")

headline = st.text_area(
    "Enter News Headline"
)

if st.button("Predict"):

    vectorized = vectorizer.transform(
        [headline]
    )

    prediction = model.predict(
        vectorized
    )[0]

    if prediction == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")
13. Run Streamlit
streamlit run app.py
requirements.txt
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
xgboost
streamlit
joblib