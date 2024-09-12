# Spamclassifier

Overview
- This project implements a spam email classifier using a Naive Bayes model trained on the SMS Spam Collection dataset. The classifier can determine whether a given email is spam or not, providing a simple and effective way to filter unwanted messages.

Files included
- train_model.py: This script handles the data loading, preprocessing, model training, and saving of the trained model and vectorizer.
- vectorizer.pkl, train_model.pkl: This is the vectorizer file and trained and preprocessed model
- app.py: A Streamlit application that provides a user interface for inputting email text and receiving spam classification results.

Requirements
To run this project, you will need the following Python packages:
- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- requests
You can download these using pip:
pip install pandas numpy scikit-learn joblib streamlit requests

How to run it
- You can run the Streamlit app by the command: streamlit run app.py
- Open the provided local URL in your web browser. You will see an input area where you can paste the email text you want to classify.
Click the "Check for Spam" button to receive feedback on whether the email is classified as spam or not.

Acknowledgments
- Dataset: SMS Spam Collection Dataset(https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- Streamlit for the user interface.
