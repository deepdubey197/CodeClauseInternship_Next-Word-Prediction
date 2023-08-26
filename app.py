import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

# Update the model's input shape to (None, 3)
model.layers[0].batch_input_shape = (None, 3)
model = tf.keras.models.model_from_json(model.to_json())  # Rebuild the model with the new input shape

def Predict_Next_Words(model, tokenizer, text, num_words=1):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)

    predicted_words = []
    for _ in range(num_words):
        preds = np.argmax(model.predict(sequence))
        predicted_word = ""

        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break

        predicted_words.append(predicted_word)
        text += " " + predicted_word  # Append the predicted word to the existing text
        text_words = text.split(" ")
        if len(text_words) > 3:
            text = " ".join(text_words[-3:])  # Join the last 3 words back to a single string
        sequence = tokenizer.texts_to_sequences([text])
        sequence = np.array(sequence)

    return predicted_words

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict_next_words():
    predicted_words = []
    if request.method == 'POST':
        text = request.form.get('text_input')
        if text:
            text = text.split(" ")
            text = " ".join(text[-3:])  # Join the last 3 words back to a single string
            num_words_to_predict = 4  # Set the number of words to predict
            predicted_words = Predict_Next_Words(model, tokenizer, text, num_words_to_predict)

    return render_template('predict.html', predicted_words=predicted_words)

if __name__ == '__main__':
    app.run(debug=True)
