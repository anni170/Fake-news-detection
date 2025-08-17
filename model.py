import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # Fixed: 'porterStemer' -> 'PorterStemmer'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Fixed: 'LogicticRegression' -> 'LogisticRegression'
from sklearn.metrics import accuracy_score  # Fixed: 'skearn' -> 'sklearn'
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier

news_df_Fake = pd.read_csv('Fake.csv')
news_df_True = pd.read_csv('True.csv')

news_df_Fake['label'] = 0
news_df_True['label'] = 1

news_df_merge = pd.concat([news_df_Fake,news_df_True],axis=0)

news_df_merge['content'] = news_df_merge['subject']+ ' ' + news_df_merge['text']
news_df_merge = news_df_merge.sample(frac=1)
news_df_merge.reset_index(inplace=True)
news_df_merge.drop(['index'],axis=1,inplace=True)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+','',text)
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\n',' ',text)
    return text

news_df_merge['text'] = news_df_merge ['text'].apply(wordopt)

x = news_df_merge['text']
y = news_df_merge['label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))

joblib.dump(LR, 'LR.pkl')
joblib.dump(vectorization, 'tfidf_vectorizer.pkl')

