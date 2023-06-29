#standard imports
import pandas as pd
import numpy as np

#vizzes
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing imports
from sklearn.model_selection import train_test_split

#classification imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

random_state=123

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------- SPLITTING WITH STRATIFICATION -------------------------------------------------------------

def split_data_strat(df, target):
    '''
    take in a df and target variable. split into train, validate, and test dfs; stratified. 60/20/20
    ---
    Format: train, validate, test = function()
    '''
    #create variables
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    #reset index
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    #returns shapes of df's
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')

    return train, validate, test

#------------------------------------------------------------- CLASSIFICATION -------------------------------------------------------------

def assign_variables(train, validate, test):
    '''
    this function takes in train, validate, and test dfs and assigns the modeling variables using hardcoded variables
    ---
    format: X_train, y_train, X_validate, y_validate, X_test, y_test = function()
    '''
    #assign variables
    X_train = train.readme #prepared text
    y_train = train.language #the label we are trying to predict for classification

    X_validate = validate.readme
    y_validate = validate.language

    X_test = test.readme
    y_test = test.language
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def make_bag_o_words(df, col):
    '''
    this function makes a bag of words from df
    ---
    format: bag_of_words = function()
    '''
        #make the count vectorizer thing to create the bag of words to prepare for modeling
    cv = CountVectorizer()

    #fit and use the thing on the entire corpus
    bag_of_words = cv.fit_transform(df[col]) #everything is getting transformed on the same data set because we haven't split yet
    return bag_of_words

def make_variable_bag_o_words(X_train, X_validate, X_test):
    '''
    this function makes bags of words from variables
    ---
    format: X_bow, X_validate_bow, X_test_bow = function()
    '''
    #make my bag of words
    cv = CountVectorizer()

    #fit and transform my bag of words on X_train!
    X_bow = cv.fit_transform(X_train)

    #only transform on validate and test!
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)
    
    return X_bow, X_validate_bow, X_test_bow

def model_one(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses Decision Tree Classifier with a depth of 3, returns print statements of the accuracies
    '''
    #build model 1
    tree3 = DecisionTreeClassifier(max_depth=3) #because decision tree likes to overfit

    #fit on bag_of_words
    tree3.fit(X_bow, y_train)

    #use on bag of words
    train_acc = tree3.score(X_bow, y_train).round(2)
    validate_acc = tree3.score(X_validate_bow, y_validate).round(2)
                    #^^^ this is the accuracy score!!

    print(f'Decision Tree / Max Depth = 3: `train` accuracy: {train_acc:.2%}')
    print(f'Decision Tree / Max Depth = 3: `validate` accuracy: {validate_acc:.2%}')

def model_two(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses KNN with neighbors of 5, returns print statements of the accuracies
    '''
    #build model 2
    knn5 = KNeighborsClassifier(5)

    #fit on bag_of_words
    knn5.fit(X_bow, y_train)

    #use on train and validate
    train_acc = knn5.score(X_bow, y_train).round(2)
    validate_acc = knn5.score(X_validate_bow, y_validate).round(2)
      #^^^ this is the accuracy score!!

    print(f'KNN / Neighbors = 5: `train` accuracy: {train_acc:.2%}')
    print(f'KNN / Neighbors = 5: `validate` accuracy: {validate_acc:.2%}')

def model_three(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses KNN with neighbors of 9, returns print statements of the accuracies
    '''
    #build model 2
    knn9 = KNeighborsClassifier(9)

    #fit on bag_of_words
    knn9.fit(X_bow, y_train)

    #use on train and validate
    train_acc = knn9.score(X_bow, y_train).round(2)
    validate_acc = knn9.score(X_validate_bow, y_validate).round(2)
      #^^^ this is the accuracy score!!

    print(f'KNN / Neighbors = 9: `train` accuracy: {train_acc:.2%}')
    print(f'KNN / Neighbors = 9: `validate` accuracy: {validate_acc:.2%}')
                
def test_model(X_bow, y_train, X_test_bow, y_test):
    '''
    this function tests the best model on the test dataset
    '''
    #build model 3
    knn9 = KNeighborsClassifier(9)

    #fit on bag_of_words
    knn9.fit(X_bow, y_train)

    #use on train and validate
    test_acc = knn9.score(X_test_bow, y_test).round(2)
      #^^^ this is the accuracy score!!

    print(f'baseline accuracy: 80.68%')
    print(f'KNN / Neighbors = 9: `test` accuracy: {test_acc:.2%}')
