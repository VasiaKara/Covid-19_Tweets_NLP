# Coronavirus NLP

## Summary
This project addressed a complex NLP task concerning text data from twitter users. The [dataset](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification) 
consists of 44.955 records and the classification task we perform aims to distinguish the sentiment of the comment. The initial comments were
classified as 'Extremely Negative', 'Negative', 'Neutral', 'Positive' and 'Extremely Positive'. For this project purpose we concatenate 'Extremely Negative' with 'Negative' comments 
'Extremely Positive' with 'Positive' comments while we discard all 'Neutral' comments. This jupyter notebook presents a text data analysis, text data preprocessing and finally several model and the corresponding
results for certain metrics(i.e. accuracy, precission, recall, f1-score).

** The data has been subjected to NLTK preprocessing in order to be applied in deep learning models. For this NLTK preprocessing we use a [pretrained vectorized dictionary](https://github.com/allenai/spv2/blob/master/model/glove.6B.100d.txt.gz) so as to 
form the embedding layer for our models.

Except from the jupyter notebook, this repo contains the implementation of a user friendly interface to test if a comment has negative or positive sentiment along with the final report of this project in greek.

## Introduction

**Natural Language Processing**, or NLP for short, is broadly defined as the automatic manipulation
of natural language, like speech and text, by software.More specifically,the area of **sentiment analysis**
, which is also known as opinion mining, is the computational study of people’s opinions, sentiments, 
emotions, appraisals, and attitudes towards entities such as products, services, organizations, 
individuals, issues, events, topics, and their attributes.

Essentially, sentiment analysis or sentiment classification fall into the broad category of text classification tasks where you are supplied with a phrase, or a list of phrases and your classifier is supposed to tell if the sentiment behind that is positive, negative or neutral.

**Dataset:**Our data source contains several tweets concerning comments relevant to covid-19 pandemic. The dataset consists of usernames, screenId(screen name), location, date, tweet text and the sentiment for each tweet. In our project the general purpose is to develop a model that is capable of correctly classify a new inserted tweet text either as positive or negative.  

## Data Preprocessing
- Check for NAN values
- Remove most frequent words
- Normalize words to its true root(Stemming and Lemmatization)
- Split the dataframe
- Create bag of words with count vectorizer
- Split target and independent variables

For deep learning models we follow different data preprocessing which is:
- Tokenization
- Padding
- Form embedding layer


## Model Fitting
- Logistic Regression
- Decission Tree
- Support Vector Machine
- Convolutional Neural Net


