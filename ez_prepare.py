from sklearn.feature_extraction.text import FfidVectorizer

def clean(text: str) -> list: #shows expectations (string will turn into a list)
    """A simple function to cleanup text data"""
    
    #remove non-ascii characters & lower
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
    
    #remove special characters
    words = re.sub(r'[^\w\s]', '', text).split()
    
    #build the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    #getting all stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def stem(text):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    text = ' '.join(stems)
    return text

def lemmatize(text):
    nltk.download('all')
    wn1 = nltk.stem.WordNetLemmatizer()
    lemmas = [wn1.lemmatize(word) for word in text.split()]
    text_lemma = ' '.join(lemmas)
    return text

def remove_stopwords(text, extra_words=[], exclude_words=[]):
    stopwords_ls = stopwords.words('english')
    stopwords_ls.extend(extra_words)
    words = text.split()
    stopword_list = [word for word in words if word not in stopwords_ls]
    filtered_words = [word for word in words if word not in stopwords_ls]
    return ' '.join(text)

