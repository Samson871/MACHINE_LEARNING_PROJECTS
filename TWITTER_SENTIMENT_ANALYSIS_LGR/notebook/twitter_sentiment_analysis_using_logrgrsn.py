# -*- coding: utf-8 -*-
"""TWITTER_SENTIMENT_ANALYSIS_USING_LOGRGRSN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17jtlyt-yCdgWRnXzLJvB528vJ7MFkrJx
"""

'''!pip install kaggle

!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#!/bin/bash
!kaggle datasets download kazanova/sentiment140

from zipfile import ZipFile
file_name = "/content/sentiment140.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')'''

#import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

import nltk
nltk.download('stopwords')

#loading the data
twitter_data=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',names=['target','id','date','flag','user','text'],encoding='latin-1')
twitter_data.head()

twitter_data.shape

df = twitter_data.iloc[:, [0, -1]]
df.columns=["sentiment","Tweet"]
df.head()

df.shape

df.isnull().sum()

#check distributions
df.sentiment.value_counts()

# prompt: write code to create new dataset new_df from the existing df dataset extracting 10000 values where sentiment is 0 and other 10000 values where sentiment id 1

# Assuming df is already defined as in your previous code

# Create new_df with 10000 samples where sentiment is 0 and 10000 where sentiment is 1
new_df = pd.concat([
    df[df['sentiment'] == 0].sample(40000, random_state=42),  # Random state for reproducibility
    df[df['sentiment'] == 4].sample(40000, random_state=42)   # Assuming sentiment 4 corresponds to 1
])

# Reset the index of new_df
new_df = new_df.reset_index(drop=True)
new_df.sentiment.value_counts()
df=new_df

#covert 4 to 1 to understand as positive sentiment
df.sentiment.replace(4,1,inplace=True)
df

#USING PORTER STEMMER FOR STEMMING
ps = PorterStemmer()

def stemming(content):

      stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
      stemmed_content = stemmed_content.lower()
      stemmed_content = stemmed_content.split()

      stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
      stemmed_content = ' '.join(stemmed_content)
      return stemmed_content

df['stemmed_tweet'] = df['Tweet'].apply(stemming)

df.head()

X= df['stemmed_tweet'].values
y= df['sentiment'].values

y

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

print(len(X_train),len(Y_train))

#convert the textual data to the numerical data

vectorizer= TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

#Model Training using Logistic Regression
model = LogisticRegression()
model.fit(X_train,Y_train)

"""Model Evaluation"""

model.score(X_train,Y_train)

pred=model.predict(X_train)
train_acc=accuracy_score(Y_train,pred)
print('Training Accuracy:',train_acc)

test_pred=model.predict(X_test)
test_acc=accuracy_score(Y_test,test_pred)
print('Testing Accuracy:',test_acc)

import pickle

filename="trained_model.sav"
pickle.dump(model,open(filename,'wb'))

loaded_model=pickle.load(open(filename,'rb'))

X_new=X_test[2922]
print(X_new)
print(Y_test[2922])

prediction = loaded_model.predict(X_new)
print(prediction)

import pickle

# ... (your existing code) ...

# Assuming 'vectorizer' is your TfidfVectorizer object
with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)