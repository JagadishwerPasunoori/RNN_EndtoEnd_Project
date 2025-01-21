# import all the required libraries and load the model:
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Load the model
model = load_model('rnn_model.h5')

# Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess the user input:
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# function to predict the sentiment of the review
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# streamlit app:
import streamlit as st

st.title('IMDB Moview Review Sentiment Analysis')
st.write('Enter a movie review to predict the sentiment (positive or negative)')
user_input = st.text_area('Enter your review:')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    prediction= model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Confidence: {prediction[0][0]}')
   
else:
    st.write('Please enter a review to classify.')
   


