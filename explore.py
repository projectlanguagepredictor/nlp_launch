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