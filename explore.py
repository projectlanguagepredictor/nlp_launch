#module
import prepare as pp
import stats_conclude as sc

#standard imports
import pandas as pd

#text
import re
import unicodedata
import nltk

#for viz
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

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
    '''
    this function uses the nested function to perform a stats test 
    '''
    def plot_unique_words_per_language(df):
        '''
        plots the count of unique words used per language (excluding 'all' and 'other') and compares them.
        '''
        word_counts = df.groupby('language')['text'].transform(lambda x: len(set(x.str.split().sum())))
        word_counts = word_counts.drop(columns={'all', 'other'})
        word_counts.nunique().plot.barh()
        plt.title('Unique Words Used Per Language (excluding other)')
        plt.xlabel('Count')
        plt.ylabel('Language')
        plt.show()

    plot_unique_words_per_language(df)
    sc.compare_categorical_continuous(df['language'], df['count'], df)



def plot_language_distribution(df):
    '''
    generates word clouds for the top 20 most frequently used words in each language.
    '''
    plt.figure(figsize=(10, 6))
    sns.countplot(y='language', data=df, order=df['language'].value_counts().index, color='skyblue')
    plt.title('Distribution of Languages')
    plt.xlabel('Count')
    plt.ylabel('Language')
    plt.show()


def generate_language_wordclouds(df):
    # Group the DataFrame by the 'language' column and join all the readme files for each language into a single string
    grouped = df.groupby('language')['readme'].apply(' '.join).reset_index()

    # Initialize a CountVectorizer
    cv = CountVectorizer(stop_words='english')

    # Ccreate a word cloud of the top 20 most frequently used words, for each language in df
    for i, row in grouped.iterrows():
        # Count the frequency of each word in the readme files for the current language
        word_count = cv.fit_transform([row['readme']])
        words = cv.get_feature_names_out()
        word_freq = word_count.toarray().sum(axis=0)
        word_freq_dict = dict(zip(words, word_freq))

        # Get the top 20 most frequently used words and their frequencies
        top_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        top_words_dict = dict(top_words)

        # Create a word cloud
        wc = WordCloud(width=800, height=400, max_words=20, background_color='white').generate_from_frequencies(top_words_dict)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Top 20 Words for {row['language']}")
        plt.show()

def calculate_average_letter_count(df):
    # Calculate the letter count for each row
    df['letter_count'] = df['readme'].apply(lambda x: len(x))
    # Group by language and calculate the average letter count
    grouped_data = df.groupby('language').agg('mean')
    print(grouped_data)
    # Create a bar plot
    plt.bar(grouped_data.index, grouped_data.letter_count)
    plt.xlabel('Language')
    plt.ylabel('Average Letter Count')
    plt.title('Average Letter Count by Language')
    plt.show()
    sc.compare_categorical_continuous(df['language'], df['letter_count'], df)        