import time
import re
import numpy as np
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from summarizer import Summarizer

stemmer = SnowballStemmer("english")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Use the Summarizer class without specifying use_gpu
bert_summarizer = Summarizer()


# Placeholder for the missing decode_sentiment function
def decode_sentiment(score, include_neutral=True, include_percentage=False):
    if include_neutral:
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        else:
            label = NEUTRAL

        if include_percentage:
            return label, max(score * 100, 100 - score * 100)
        else:
            return label

    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def preprocess(text):
    # Use the text summarization function from the library
    summary = bert_summarizer(text)

    # Use sentiment analysis to identify the sentiment for the entire paragraph
    sentiment_info = decode_sentiment(0.5)  # You can adjust the threshold as needed
    sentiment_label = sentiment_info[0].lower() if sentiment_info[0] else "NEUTRAL"

    # Generate the summary with sentiment analysis
    summary_with_sentiment = f"the user expressed: {summary} : sentiment {sentiment_label.upper()}"

    return summary_with_sentiment


def predict(model, tokenizer, tweets, include_neutral=True, sequence_length=300):
    """
    model: Keras model
    tokenizer: Tokenizer object
    tweets: List of strings (tweets)
    """

    start_at = time.time()
    # Tokenize text
    tweets = [preprocess(t) for t in tweets]
    X = pad_sequences(tokenizer.texts_to_sequences(tweets), maxlen=sequence_length)
    # Predict
    score = model.predict(X, batch_size=50)
    # Decode sentiment
    labels = []
    for s in score:
        label = decode_sentiment(s, include_neutral=include_neutral)
        labels.append(label)

    return {"label": labels, "score": score,
            "elapsed_time": time.time() - start_at}
