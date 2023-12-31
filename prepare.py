#standard imports
import pandas as pd

#scrape/html imports
import requests
from requests import get
from bs4 import BeautifulSoup
from pprint import pprint

#sleep timer
import time

#import regex
import re

#import file
import os
import json

#prepare imports
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

print(f'imports loaded successfully, awaiting commands...')

# -----------------------------------------------------------------PREPARE-----------------------------------------------------------------

#create lowercase function
def lower_everything(string):
    return string.str.lower()

def normalize_everything(string):
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    return string

#create removal of specials function
def specials_removed(string):
    string = re.sub(r'[^a-z\'\s]', ' ', string)
    return string

def basic_clean(string):
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    string = re.sub(r'[^a-z\'\s]', ' ', string)
    return string

def token_it_up(string):
    tokenize = nltk.tokenize.ToktokTokenizer()
    string = tokenize.tokenize(string, return_str=True)
    return string

def lemmad(string):
    wnl = nltk.stem.WordNetLemmatizer()
    string = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(string)
    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    
    ADDITIONAL_STOPWORDS = ['&#9;', 'www', 'github', 'com', 'http', 'td', 'c', 'e', 'org', 'http ', 'b', ' http', 'io']
    
    sls = stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    sls = set(sls) - set(exclude_words)
    sls = sls.union(set(extra_words))
    
    words = string.split()
    filtered = [word for word in words if word not in sls]
    string = ' '.join(filtered)
    return string

def clean_df(df, col_to_clean, exclude_words=[], extra_words=[]):
    '''
    send in df, returns df with repo and author, language, and clean readme data
    '''
    df['clean'] = df[col_to_clean].apply(basic_clean).apply(token_it_up).apply(remove_stopwords)
    df['readme'] = df.clean.apply(lemmad)
    
    df = df.drop(columns={'clean', col_to_clean})
    
    #assign languages to keep
    languages = ['Python', 'JavaScript', 'HTML', 'Shell', 'Java', 'Go']

    #edit the languages but keep the data
    df['language'] = df.apply(lambda row: row['language'] if row['language'] in languages else 'other', axis=1)
    df['language'] = lower_everything(df['language'])
                                      
    
    return df

def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore') #most frequently used for base text creation - works great with SQL
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words