
...(Available on Kaggle: [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/markmedhat/twitter))*

---

## Usage

### Run the Streamlit app
```bash
streamlit run app.py
```
```bash
python -m streamlit run sentiment_analyser.py
```


### Interface Preview
- Input text into the sidebar
- View the predicted sentiment instantly
- Explore visual insights like:
  - Sentiment distribution
  - Model evaluation metrics
  - Word clouds for each sentiment

---

## Dataset

This project uses the publicly available Twitter Sentiment Analysis Dataset from Kaggle, which contains labeled tweets classified as:

- `Positive`
- `Negative`
- `Neutral`
- `Irrelevant`

Each entry in the dataset includes:
- Tweet ID
- Entity (topic of tweet)
- Sentiment label
- Tweet text

---

## How It Works

1. ### Preprocessing
   - Lowercasing text
   - Removing URLs, mentions, hashtags
   - Handling emojis and emoticons
   - Stopword removal
   - Tokenization & Lemmatization

2. ### Feature Extraction
   - TF-IDF Vectorization for numerical representation of text

3. ### Model
   - Logistic Regression Classifier (Scikit-learn)
   - Trained on preprocessed data with optimized hyperparameters

4. ### Evaluation
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Visualization using Matplotlib and Seaborn



5. ### Limitations
- May struggle with sarcasm or slang
- Context awareness is limited
- Dataset bias may affect predictions
- Limited performance on very short texts
