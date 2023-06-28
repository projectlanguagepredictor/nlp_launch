import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

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

def represent_text(texts):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix

'''# Example usage
document1 = "This is an example document for text preprocessing."
document2 = "Text representation is important for NLP analysis."

# Preprocess the text
preprocessed_doc1 = preprocess_text(document1)
preprocessed_doc2 = preprocess_text(document2)

print("Preprocessed Document 1:", preprocessed_doc1)
print("Preprocessed Document 2:", preprocessed_doc2)

# Represent the text using TF-IDF
texts = [document1, document2]
tfidf_matrix = represent_text(texts)

print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())'''