# Deploying GUVI Sentiment Analysis Model Using AWS Services
## Overview
The GUVI Sentiment Analysis project aims to analyze the sentiment of text input and classify it as Positive, Negative, or Neutral. This project leverages Natural Language Processing (NLP) techniques and Machine Learning models to provide accurate sentiment predictions. This project deploys its final model on AWS EC2, enabling it to fetch files from an AWS S3 bucket. Additionally, it stores basic information in an AWS RDS server.
## Table of Contents
- [Key Technologies and Skills](#key-technologies-and-skills)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Approach](#approach)
- [Contact](#contact)
# Key Technologies and Skills
- Python Scripting
- Streamlit
- Data Cleaning and analysis
- Pandas-Dataframe
- Numpy
- Scikit-Learn
- Pickle
- TFID Vectorizer
- Label encoder
- AWS EC2
- AWS S3
- AWS RDS
# Installation
To run this project, please install below python packages as prerequisites in AWS EC2.

```bash
sudo yum install -y python3-pip
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install boto3
pip install mysql-connector-python
```
# Usage
To use this project, Please follow the below steps in AWS EC2.
- To clone this repository: ```git clone https://github.com/Gokulakkrizhna/sentiment_analysis.git```
- Install the required packages: ```pip install -r requirements.txt ```
- To run the Streamlit app in AWS EC2: ```python3 /home/ec2/aws_app.py```
# Features
- Fetch Twitter data from Github
- Data Cleaning and pre-processing
- Perform Machine Leanrning analysis
- User-friendly interface powered by Streamlit
- Deployment on AWS EC2
# Approach
```Data Collection```: Fetch opinion on GUVI from Github. 

```Data Cleaning```: Perform pre-processing methods like Data handling is applied to the collected data.

```Setup the Streamlit app```: Streamlit is a user-friendly web development tool that simplifies the process of creating intuitive interfaces.

```Data Analysis```: Cleaned data has been analyzed in Streamlit through Pandas DataFrame.

```Machine Learning```: Cleaned data has been applied in different machine learning algorithm to classify the sentiment.

The provided code utilizes Python scripting along with various libraries to fetch data from GitHub. NLP cleaning has been applied to the collected data, and the cleaned data is used to train the ML model. The finalized model is implemented in Streamlit and deployed on AWS EC2.

Here's a breakdown of what the code does:
- Importing all the neccessary libraries includes ```Streamlit``` which creates UI to interact with user and display the analysed data, ```Pandas``` which helps to display the analysed data in Streamlit web,```numpy``` which will help in mathematical conversion, ```LabelEncoder``` used to convert categorical values to numerical data,```pickle``` is used to load the trained model,```nltk``` used to cleaning the NLP data, ```sql-connector``` is used to connect MySql server, ```boto3``` which helps to connect S3 bucket and EC2.
```bash
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mysql.connector
import boto3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
```
- ```Data_preprocessing``` is responsible for cleaning the NLP data by removing stopwords, punctuation, and single letters, and performing lemmatization. It also prepares the data for training the ML models.
- ```app```  collects the user's name and sentence input for sentiment analysis, stores the user's name and login time in the AWS RDS server, and displays the sentiment analysis results for the input sentence.**Note: Replace your id,password,number in```host,password,port```**
```bash
mydb = mysql.connector.connect(
    host="please provide rds host id",
    user="admin",
    password="please provide rds password ",
    port="please provide port number",
    )
```
- ```aws_app``` manages the connection between an S3 bucket and an EC2 instance, facilitating the download of files from S3 to EC2 for execution.**Note: Replace your bucket name,id,access key in ```s3_bucket,aws_access_key_id,aws_secret_access_key```**
```bash
  s3_bucket = 'please provide bucket name'
  aws_access_key_id = 'please provide s3 access id'
  aws_secret_access_key = 'please provide s3 access key'
```
- Two separate tabs have been implemented in the Streamlit web application to facilitate user interaction.
```bash
tab1, tab2, tab3, tab4= st.tabs(["Home", "Sentiment Analysis"])
```
- In Tab1 of the Streamlit web application,users provide their username, which is then stored in the RDS server.
- In Tab2, users input a sentence, and the sentiment of that sentence is displayed.

- This project fetches data from GitHub, cleans and processes it using NLP techniques, trains a model to analyze sentiment, and finally deploys this model on AWS EC2 for practical use.

# Contact
📧 Email: [gokulakkrizhna@gmail.com](mailto:gokulakkrizhna@gmail.com)

🌐 LinkedIn: [linkedin.com/in/gokulakkrizhna-s-241562159](https://www.linkedin.com/in/gokulakkrizhna-s-241562159/)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.
  
