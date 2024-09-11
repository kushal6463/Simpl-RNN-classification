from tensorflow.keras.utils import pad_sequences


def preprocess_text(text, word_index, maxlen=100):
    words = text.lower().split()  
    # Encode the input text based on the IMDb word index
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review


def predict_sentiment(review, model, word_index, maxlen=100):
    preprocessed_input = preprocess_text(review, word_index, maxlen) 
    prediction = model.predict(preprocessed_input)  
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    confidence = prediction[0][0]
    return sentiment, confidence
