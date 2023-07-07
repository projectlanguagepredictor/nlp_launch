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
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier

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

def model_mbsmooth(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses multinomial naive boyes modeling with smoothing on train and validate
    '''
    #make the thing
    nb = MultinomialNB()

    #fit and use on bow train
    nb.fit(X_bow, y_train)
    train_acc = nb.score(X_bow, y_train)

    #use on validate
    validate_acc = nb.score(X_validate_bow, y_validate)

    print(f'Multinomial Naive Bayes: `train` accuracy: {train_acc:.2%}')
    print(f'Multinomial Naive Bayes: `validate` accuracy: {validate_acc:.2%}')
    
def model_mbnosmooth(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses multinomial naive boyes modeling with no smoothing on train and validate
    '''
    #make the thing
    nbc = MultinomialNB(alpha=0)

    #fit and use on bow train
    nbc.fit(X_bow, y_train)
    train_acc_c = nbc.score(X_bow, y_train)

    #use on validate
    validate_acc_c = nbc.score(X_validate_bow, y_validate)

    print(f'Multinomial Naive Bayes: `train` accuracy: {train_acc_c:.2%}')
    print(f'Multinomial Naive Bayes: `validate` accuracy: {validate_acc_c:.2%}')

def model_rf10(X_bow, y_train, X_validate_bow, y_validate, baseline_accuracy, x=20):
    ''' This function is to calculate the best random forest decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]

    for x in range(1,11):
        rf = RandomForestClassifier(max_depth = x)
        rf.fit(X_bow, y_train)
        train_acc = rf.score(X_bow, y_train)
        print(f"For depth of {x:2}, the accuracy is {train_acc:.2%}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc = rf.score(X_validate_bow, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

    print(f'The baseline accuracy is {baseline_accuracy:.2%}')
          
    # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()

def model_dt3(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses Decision Tree Classifier with a depth of 3, returns print statements of the accuracies
    '''
    #build model 1
    tree3 = DecisionTreeClassifier(max_depth=3) #because decision tree likes to overfit

    #fit on bag_of_words
    tree3.fit(X_bow, y_train)

    #use on bag of words
    train_acc = tree3.score(X_bow, y_train)
    validate_acc = tree3.score(X_validate_bow, y_validate)
                    #^^^ this is the accuracy score!!

    print(f'Decision Tree / Max Depth = 3: `train` accuracy: {train_acc:.2%}')
    print(f'Decision Tree / Max Depth = 3: `validate` accuracy: {validate_acc:.2%}')

def model_knn5(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses KNN with neighbors of 5, returns print statements of the accuracies
    '''
    #build model 2
    knn5 = KNeighborsClassifier(5)

    #fit on bag_of_words
    knn5.fit(X_bow, y_train)

    #use on train and validate
    train_acc = knn5.score(X_bow, y_train)
    validate_acc = knn5.score(X_validate_bow, y_validate)
      #^^^ this is the accuracy score!!

    print(f'KNN / Neighbors = 5: `train` accuracy: {train_acc:.2%}')
    print(f'KNN / Neighbors = 5: `validate` accuracy: {validate_acc:.2%}')

def model_knn9(X_bow, y_train, X_validate_bow, y_validate):
    '''
    this functions uses KNN with neighbors of 9, returns print statements of the accuracies
    '''
    #build model 2
    knn9 = KNeighborsClassifier(9)

    #fit on bag_of_words
    knn9.fit(X_bow, y_train)

    #use on train and validate
    train_acc = knn9.score(X_bow, y_train)
    validate_acc = knn9.score(X_validate_bow, y_validate)
      #^^^ this is the accuracy score!!

    print(f'KNN / Neighbors = 9: `train` accuracy: {train_acc:.2%}')
    print(f'KNN / Neighbors = 9: `validate` accuracy: {validate_acc:.2%}')
                
def test_model(X_bow, y_train, X_test_bow, y_test, baseline_accuracy):
    '''
    this function tests the best model on the test dataset
    '''
    #make the thing
    rf = RandomForestClassifier(max_depth = 4)

    #fit the thing
    rf.fit(X_bow, y_train)

    #use the thing
    test_acc = rf.score(X_test_bow, y_test)

    #display results
    print(f'The baseline accuracy is {baseline_accuracy:.2%}')
    print(f'The test accuracy is {test_acc:.2%}')