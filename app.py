import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense



from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

model=load_model('/Users/aashutoshkumarbhardwaj/dp/imdb_CNN_sentiment/simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.split()
    encoded = [word_index.get(word, 0) + 3 for word in words]
    return sequence.pad_sequences([encoded], maxlen=500)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    return 'Positive' if prediction[0][0] > 0.5 else 'Negative'


import streamlit as st

st.title("Sentiment Analysis of Movie Reviews")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", height=200)
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Prediction Score: {model.predict(preprocess_text(user_input))[0][0]:.4f}")
    else:
        st.write("Please enter a movie review to analyze.")


