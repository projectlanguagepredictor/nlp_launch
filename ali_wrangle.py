#standard imports
import pandas as pd

#my module
import env

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
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    return string

def basic_clean(string):
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    return string

def token_it_up(string):
    tokenize = nltk.tokenize.ToktokTokenizer()
    string = tokenize.tokenize(string, return_str=True)
    return string

def stemmer(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    return string

def lemmad(string):
    wnl = nltk.stem.WordNetLemmatizer()
    string = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(string)
    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    sls = stopwords.words('english')
    
    sls = set(sls) - set(exclude_words)
    sls = sls.union(set(extra_words))
    
    words = string.split()
    filtered = [word for word in words if word not in sls]
    string = ' '.join(filtered)
    return string

def clean_df(df, exclude_words=[], extra_words=[]):
    '''
    send in df with columns: title and original,
    returns df with original, clean, stemmed, and lemmatized data
    '''
    df['clean'] = df.original.apply(basic_clean).apply(token_it_up).apply(remove_stopwords)
    df['stem'] = df.clean.apply(stemmer)
    df['lemma'] = df.clean.apply(lemmad)
    return df

def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    #assigning additional stopwords
    ADDITIONAL_STOPWORDS = ['r', 'u', '2', '4', 'ltgt']
    
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
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words