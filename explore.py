#module
import prepare as pp

#standard imports
import pandas as pd

#text
import re
import unicodedata
import nltk

import stats_conclude

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
    python_words = pp.clean(' '.join(df[df.language == 'python'].readme))
    javascript_words = pp.clean(' '.join(df[df.language == 'javascript'].readme))
    html_words = pp.clean(' '.join(df[df.language == 'html'].readme))
    shell_words = pp.clean(' '.join(df[df.language == 'shell'].readme))
    java_words = pp.clean(' '.join(df[df.language == 'java'].readme))
    go_words = pp.clean(' '.join(df[df.language == 'go'].readme))
    other_words = pp.clean(' '.join(df[df.language == 'other'].readme))
    all_words = pp.clean(' '.join(df.readme))
    
    
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
    

def plot_unique_words_and_compare(df):
    def plot_unique_words_per_language(df):
        word_counts = df.groupby('language')['text'].transform(lambda x: len(set(x.str.split().sum())))
        word_counts = word_counts.drop(columns={'all', 'other'})
        word_counts.nunique().plot.barh()
        plt.title('Unique Words Used Per Language (excluding other)')
        plt.xlabel('Count')
        plt.ylabel('Language')
        plt.show()

    plot_unique_words_per_language(df)
    sc.compare_categorical_continuous(df['language'], df['count'], df)

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Call the function
plot_unique_words_and_compare(df)
