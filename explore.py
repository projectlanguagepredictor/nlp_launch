#module
import prepare as pp

#standard imports
import pandas as pd

#text
import re
import unicodedata
import nltk



# -----------------------------------------------------------------EXPLORE-----------------------------------------------------------------

def show_counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1).round(2)
    labels.columns = ['n', 'percent']
    labels
    return labels

def get_words(df):
    '''
    this function extracts and counts words from a df based on different programming languages.
    returns a filtered df containing the top 4000 words.
    '''
    #assinging all words to proper labels
    python_words = pp.clean(' '.join(df[df.language == 'python'].lemma))
    javascript_words = pp.clean(' '.join(df[df.language == 'javascript'].lemma))
    html_words = pp.clean(' '.join(df[df.language == 'html'].lemma))
    shell_words = pp.clean(' '.join(df[df.language == 'shell'].lemma))
    java_words = pp.clean(' '.join(df[df.language == 'java'].lemma))
    go_words = pp.clean(' '.join(df[df.language == 'go'].lemma))
    other_words = pp.clean(' '.join(df[df.language == 'other'].lemma))
    all_words = pp.clean(' '.join(df.lemma))
    
    
    #grabbing frequencies of occurences
    python_freq = pd.Series(python_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    html_freq = pd.Series(html_words).value_counts()
    shell_freq = pd.Series(shell_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    go_freq = pd.Series(go_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    #combine into df to see all words and languages together
    word_counts = (pd.concat([all_freq, python_freq, javascript_freq, html_freq, shell_freq, java_freq, go_freq, other_freq], axis=1, sort=True)
                .set_axis(['all', 'python', 'javascript', 'html', 'shell', 'java', 'go', 'other'], axis=1)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    #filtering out the garbage 
    filtered_word_counts = word_counts.sort_values(by='all',ascending=False)[:4000]
    
    print(f"Unfiltered Data:{word_counts.shape[0]} words  Filtered Data: {filtered_word_counts.shape[0]} words")
    print()
    
    return filtered_word_counts
    