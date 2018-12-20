# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from pprint import pprint
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from time import time
import csv as CSV

print("Initializing...")
train_data_frame = pd.read_csv('./data/crowdai_cleaned_train.csv')
test_data_frame = pd.read_csv('./data/test_data_cleaned.csv')

x_train = train_data_frame.text.astype('U')
y_train = train_data_frame.sentiment
x_test = test_data_frame.text.astype('U')

#Regressiontype
#linear_classifier = SGDClassifier()
linear_classifier = LogisticRegression()
#linear_classifier = Ridge()

#Create count vectorizer and fit it to data
count_vectorizer = CountVectorizer()
count_vectorizer.fit(pd.Series.append(x_train, x_test))

#n_gram_range = (1, 1)
#n_gram_range = (1, 2)
n_gram_range = (1, 3)
maximum_number_of_words = 100000
count_vectorizer.set_params(max_features=maximum_number_of_words, ngram_range=n_gram_range)

print("Training model...")
pipeline = Pipeline([('vectorizer', count_vectorizer),('classifier', linear_classifier)])
sentiment_fit = pipeline.fit(x_train, y_train)

print("Predicting sentiments...")
y_pred = sentiment_fit.predict(x_test)
y_pred[y_pred==0] = -1

ids = np.linspace(1, len(y_pred), num=len(y_pred),dtype=int)
with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = CSV.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(ids, y_pred):
        writer.writerow({'Id':int(r1),'Prediction':int(r2)})

print("Done. Predictions saved in predictions.csv")



