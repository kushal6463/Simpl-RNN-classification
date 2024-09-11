from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from helper import predict_sentiment
import streamlit as st

model = load_model("simple_rnn_imdb.keras")
word_index = imdb.get_word_index()


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User review
custom_review = st.text_area("Movie Review")

if st.button("Classify"):

    sentiment, confidence = predict_sentiment(custom_review, model, word_index)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {confidence}")
else:
    st.write("Please enter a movie review.")
