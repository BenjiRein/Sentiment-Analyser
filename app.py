import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report

# Suppress Streamlit warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_data():
    # Load dataset
    train_data = pd.read_csv('twitter_training.csv', header=None)
    val_data = pd.read_csv('twitter_validation.csv', header=None)
    data = pd.concat([train_data, val_data], ignore_index=True)
    data.columns = ['id', 'game', 'sentiment', 'text']
    
    # Preprocess data
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    
    # Prepare features and target
    X = data['cleaned_text']
    y = data['sentiment']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

@st.cache_resource
def load_model(X_train, y_train):
    # Create and train model
    model = make_pipeline(
        TfidfVectorizer(max_features=5000),
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    model.fit(X_train, y_train)
    return model

def predict_sentiment(text, model):
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])
    
    # model prediction
    st.write(f"prediction: {prediction}")

    # Map the numeric prediction to the corresponding sentiment label
    sentiment_map = {
        0: "Irrelevant", 
        1: "Negative", 
        2: "Neutral", 
        3: "Positive"
    }

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Show classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Show sentiment distribution for the test set
    sentiment_counts = pd.Series(y_pred).value_counts()
    sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue', 'yellow'])
    plt.title('Sentiment Distribution in Predictions')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot()

# Streamlit UI
st.set_page_config(page_title="Sentiment Seer", layout="centered")

# Custom CSS for styling
st.markdown(f"""
    <style>
        .stTextInput textarea {{
            background-color: #E0F7FA;
        }}
        .stButton>button {{
            background-color: #008080;
            color: white;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #006666;
        }}
        .result {{
            font-size: 1.2rem;
            padding: 1rem;
            background-color: #E0F7FA;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }}
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Sentiment Analyser 🔮")

# Load data and model
X_train, X_test, y_train, y_test = load_data()
model = load_model(X_train, y_train)

# User input
user_input = st.text_area("Enter your text here:", height=150, 
                         placeholder="Type or paste your text here...")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, model)
        
        sentiment_mapping = {
            'Positive': 'positive',
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Irrelevant': 'neutral'
        }
    else:
        st.warning("Please enter some text to analyze.")

# Evaluate model button
if st.button("Evaluate Model"):
    evaluate_model(model, X_test, y_test)