# Step 1: Import necessary libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Load the pre-trained model
model = load_model('simple_rnn_imdb_model.h5')

## Step 2: Helper 
# function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review]) # Adjusting index by 3 as per Keras documentation

# function to preprocess input text
def preprocess_text(text):
    # Tokenize the text
    words = text.lower().split()
    # Convert words to their respective indices based on the IMDB word index
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # +3 to account for reserved encoded_review
    # Pad the sequence to ensure uniform input length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

# User input
user_input = st.text_area("Movie Review", "Type your review here...")
if st.button("Predict Sentiment"):
    # Preprocess the input text
    preprocessed_review = preprocess_text(user_input)
    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_review)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    st.write(f"Predicted Sentiment: {sentiment} (Prediction Score: {prediction[0][0]:.4f})")

else:
    st.write("Please enter a review and click the 'Predict Sentiment' button.")
