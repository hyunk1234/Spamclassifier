import streamlit as st
import joblib 
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#Streamlit app title
st.title("Spam Email Classifier")

#User input
user_input = st.text_area("Enter the email text: ")

#function to clean and preprocess the input text
def preprocess_input(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

#Preprocess and classify the input text
if st.button("Check for Spam"):
    if user_input.strip() == "":
        st.write("Please enter some text to classify.")
    else:
        #Preprocess the input text
        cleaned_input = preprocess_input(user_input)
        #transform the input using countvectorizer
        input_features = vectorizer.transform([cleaned_input])
        #Predict the label
        prediction = model.predict(input_features)[0]
        #Display the result 
        if prediction == 1:
            st.write("This email is classified as **Spam**.")
        else:
            st.write("This email is classified as **Not Spam**.")