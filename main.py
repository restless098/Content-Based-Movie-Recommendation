import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={
    value:key for key,value in word_index.items()
}

#loading the model
model=load_model('simpleRNN_imdb.h5')

#Helper functions 

#function to decode the review

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess input review

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#Prediction sentiment
def predict_sentiment(review):
    preprocessed_text=preprocess_text(review)
    prediction=model.predict(preprocessed_text)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]

#importing streamlit 
import streamlit as st

st.title('IMDB sentiment Analysis')
st.write('enter movie review to classify it as positive or negative')

#user input

input_1=st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,score=predict_sentiment(input_1)
    
    #display the result
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Score:{score}')
else:   
    st.write("Please enter movie review")