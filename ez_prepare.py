def basic_clean(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\'\s]', '', text)
    return text

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

#mine, failed to add ', extra_words=[], exclude_words=[]'
def remove_stopwords(text, extra_words=[], exclude_words=[]):
    #nltk.download('stopwords')
    stopwords_ls = stopwords.words('english')
    stopwords_ls.extend(extra_words)
    #to add words to stopwords_ls.append('word(s) here')
    words = text.split()
    stopword_list = [word for word in words if word not in stopwords_ls]
    filtered_words = [word for word in words if word not in stopwords_ls]
    return ' '.join(text)

