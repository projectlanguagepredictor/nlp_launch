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

def make_bag_o_words(X_train, X_validate, X_test):
    '''
    this function makes bags of words
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
    print(f'KNN/Neighbors=9: `test` accuracy: {test_acc:.2%}')

#------------------------------------------------------------- VIZZES -------------------------------------------------------------

# def plot_target(df, col1):
#     """This function plots the target variable"""
#     # plot the target variable
#     plt.figure(figsize=(10, 10))
#     sns.histplot(df[col1], bins=60, color='purple')
#     plt.axvline(df[col1].mean(), label='Average Profit', color='black', linewidth=2)
#     plt.legend()
#     plt.title('Distribution of Target Variable: Profit');

# def plot_model_predictions(p1, p2, p3, y, baseline):
#     """Plots the selected models predictions with baseline"""
#     plt.scatter(p1, y, label='Linear Regression',  color="red", alpha=.6)
#     plt.scatter(p2, y, label='LassoLars', color='green', alpha=.6)
#     plt.scatter(p3, y, label='GLM', color='blue', alpha=.6)
#     plt.plot(y, y, label='_nolegend_', color='grey')

#     plt.axhline(baseline, ls=':', color='grey')
#     plt.annotate("Baseline", (65, 81))

#     plt.title("Where are predictions more extreme? More modest? Overfit?")
#     plt.ylabel("Actual Profit")
#     plt.xlabel("Predicted Profit")
#     plt.legend()
#     plt.show()
    
# def plot_model_residuals(p1, p2, p3, y):
#     """Plots the selected models residuals with baseline"""
#     plt.axhline(label="No Error")

#     plt.scatter(y, p1 - y, alpha=.5, color="red", label="Linear Regression")
#     plt.scatter(y, p2 - y, alpha=.5, color="green", label="LassoLars")
#     plt.scatter(y, p3 - y, alpha=.5, color="blue", label="GLM")

#     plt.legend()
#     plt.title("Do the size of errors change as the actual value changes?")
#     plt.xlabel("Actual Profit")
#     plt.ylabel("Residual: Predicted Profit - Actual Profit")
#     plt.show()

# def plot_model_actual_predicted(y, p1):
#     """Plots the selected models predictions with actual"""
#     plt.hist(y, color='purple', alpha=.4, label="Actual")
#     plt.hist(p1, color='green', alpha=.9, label="LassoLars")
#     # plt.hist(pred_glm, color='orange', alpha=.7, label='GLM')
#     # plt.hist(pred_lars, color='yellow', alpha=.2, label='LassoLars')

#     plt.xlabel("Profit Size")
#     plt.ylabel("Count")
#     plt.title("Comparing the Distribution of Actual to Predicted Profit Size")
#     plt.legend()
#     plt.show()