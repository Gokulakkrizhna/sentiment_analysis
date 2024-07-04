import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import mysql.connector

# Ignore specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def remove_stopwords(text):
    en_stopwords = stopwords.words('english')
    result = []
    for token in text:
        if token.lower() not in en_stopwords:
            result.append(token)

    return result

def remove_punct(text):
    tokenizer = RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(' '.join(text))
    return lst

def keep_alphabetical_only(sentiment_list):
    return [word for word in sentiment_list if word.isalpha()]

def remove_single_letters(text):
    return [word for word in text if len(word) > 1]


def data_preprocess(a):
    a = a.lower()
    a = " ".join(a.split())
    a = word_tokenize(a)
    a = remove_stopwords(a)
    a = remove_punct(a)
    a = keep_alphabetical_only(a)
    a = remove_single_letters(a)
    a = " ".join(a)
    with open('/home/ec2-user/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    a = vectorizer.transform([a])

    return a

def streamlit():
    tab1, tab2 = st.tabs(["Home", "Sentiment Analysis"])
    with tab1:
        st.markdown('<h1 style="text-align: center; color: red;">GUVI SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)
        name = st.text_input("Please enter your name", "")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        if st.button("Login"):
            mycursor.execute(
                "INSERT INTO cust_info (name, login) VALUES (%s, %s)",
                (name, formatted_datetime)
            )
            st.success('Data migrated to RDS-Mysql server!', icon="✅")

    with tab2:
        sentence = st.text_input("Please enter the sentence", "")

        sentence = data_preprocess(sentence)

        if st.button("Sentiment Analysis"):
            with open('/home/ec2-user/rf.pkl', 'rb') as file:
                model = pickle.load(file)
            
            y_pred = model.predict(sentence)

            if y_pred[0] == 1:
                st.subheader(":red[Sentiment of the sentence is] Neutral")
            elif y_pred[0] == 0:
                st.subheader(":red[Sentiment of the sentence is] Negative")
            else:
                st.subheader(":red[Sentiment of the sentence is] Positive")



mydb = mysql.connector.connect(
    host="please provide rds host id",
    user="admin",
    password="please provide rds password ",
    port="please provide port number",
    )
mycursor = mydb.cursor(buffered=True)

mycursor.execute("create database if not exists sentimentanalysis")
mycursor.execute("use sentimentanalysis")
mycursor.execute("create table if not exists cust_info (name varchar(255) ,login DATETIME)")

streamlit()