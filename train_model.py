import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#Load dataset from URL
from io import BytesIO
from zipfile import ZipFile
import requests

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as zfile:
    with zfile.open('SMSSpamCollection') as file:
        data = pd.read_csv(file, sep='\t', header=None, names=['Label', 'EmailText'])

#Data Preprocessing
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})
data['EmailText'] = data['EmailText'].str.replace(r'\W', ' ').str.lower()

#Split dataset
X= data['EmailText']
y= data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorization
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

#Model Training 
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

#Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and Vectorizer saved successfully")