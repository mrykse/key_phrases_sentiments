from flask import Flask, render_template, request
from keras.src.saving.saving_api import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from utils import preprocess, predict, decode_sentiment

app = Flask(__name__)

# Load the model, tokenizer, and encoder
model = load_model("twitter.h5")
tokenizer = pickle.load(open("tokenizer.p", "rb"))
encoder = pickle.load(open("encoder.p", "rb"))

SEQUENCE_LENGTH = 300


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess(text)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
        prediction = model.predict(np.array(padded_sequence))
        sentiment, percentage = decode_sentiment(prediction[0], include_percentage=True)

        return render_template('index.html', text=text, sentiment=sentiment, percentage=percentage)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
