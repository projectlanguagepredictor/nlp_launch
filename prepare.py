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

def lemmad(string):
    wnl = nltk.stem.WordNetLemmatizer()
    string = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(string)
    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    
    ADDITIONAL_STOPWORDS = ['Python', 'python', 'makefile', 'Makefile', 'Java', 'java', '&#9;'\
                           , 'HTML', 'Shell', 'CSS', 'TypeScript', 'Go', 'JavaScript']
    
    sls = stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    sls = set(sls) - set(exclude_words)
    sls = sls.union(set(extra_words))
    
    words = string.split()
    filtered = [word for word in words if word not in sls]
    string = ' '.join(filtered)
    return string

def clean_df(df, col_to_clean, exclude_words=[], extra_words=[]):
    '''
    send in df, returns df with original, clean, and lemmatized data
    '''
    df['clean'] = df[col_to_clean].apply(basic_clean).apply(token_it_up).apply(remove_stopwords)
    df['lemma'] = df.clean.apply(lemmad)
    return df