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

#------------------------------------------------------------- BASELINE -------------------------------------------------------------

def baseline_accuracy(train, mode_variable, target_col):
    '''
    this function creates a baseline prediction column on the train dataset used for classification and returns the baseline accuracy rating
    ---
    Format: baseline_accuracy = function()
    '''
    #establish baseline accuracy
    train['baseline_prediction'] = {mode_variable}
    baseline_accuracy = (train.baseline_prediction == train[target_col]).mean()
    print(f'The baseline accuracy is {baseline_accuracy:.2%}')

    return baseline_accuracy

#------------------------------------------------------------- CLASSIFICATION -------------------------------------------------------------



#------------------------------------------------------------- VIZZES -------------------------------------------------------------

def plot_target(df, col1):
    """This function plots the target variable"""
    # plot the target variable
    plt.figure(figsize=(10, 10))
    sns.histplot(df[col1], bins=60, color='purple')
    plt.axvline(df[col1].mean(), label='Average Profit', color='black', linewidth=2)
    plt.legend()
    plt.title('Distribution of Target Variable: Profit');

def plot_model_predictions(p1, p2, p3, y, baseline):
    """Plots the selected models predictions with baseline"""
    plt.scatter(p1, y, label='Linear Regression',  color="red", alpha=.6)
    plt.scatter(p2, y, label='LassoLars', color='green', alpha=.6)
    plt.scatter(p3, y, label='GLM', color='blue', alpha=.6)
    plt.plot(y, y, label='_nolegend_', color='grey')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81))

    plt.title("Where are predictions more extreme? More modest? Overfit?")
    plt.ylabel("Actual Profit")
    plt.xlabel("Predicted Profit")
    plt.legend()
    plt.show()
    
def plot_model_residuals(p1, p2, p3, y):
    """Plots the selected models residuals with baseline"""
    plt.axhline(label="No Error")

    plt.scatter(y, p1 - y, alpha=.5, color="red", label="Linear Regression")
    plt.scatter(y, p2 - y, alpha=.5, color="green", label="LassoLars")
    plt.scatter(y, p3 - y, alpha=.5, color="blue", label="GLM")

    plt.legend()
    plt.title("Do the size of errors change as the actual value changes?")
    plt.xlabel("Actual Profit")
    plt.ylabel("Residual: Predicted Profit - Actual Profit")
    plt.show()

def plot_model_actual_predicted(y, p1):
    """Plots the selected models predictions with actual"""
    plt.hist(y, color='purple', alpha=.4, label="Actual")
    plt.hist(p1, color='green', alpha=.9, label="LassoLars")
    # plt.hist(pred_glm, color='orange', alpha=.7, label='GLM')
    # plt.hist(pred_lars, color='yellow', alpha=.2, label='LassoLars')

    plt.xlabel("Profit Size")
    plt.ylabel("Count")
    plt.title("Comparing the Distribution of Actual to Predicted Profit Size")
    plt.legend()
    plt.show()