#! /usr/bin/env python3
# coding=utf-8

'''
Title: Disaster Response Project: ML Pipeline
Author: Akshita Gupta

'''


import pandas as pd
import numpy as np
import nltk
import logging
import sys

from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus.reader.wordnet import NOUN, ADJ, VERB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


nltk.download(['punkt', 'wordnet', 'stopwords', 'words'])
stop_words = set(stopwords.words('english'))


def load_data(database_filepath: str):
    '''
    Function to load data from database
    
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: np.array, dataset of features.
        Y: np.array, dataset with dependent feature 
        category_names: Y labels, a list.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    '''
    Function to tokenize words from input sentences

    input:
        text: Message data; str.
    output:
        clean_tokens: list of cleaned token after tokenization.
    '''
    
    
    word_tokenizer = RegexpTokenizer(r'\w+')
    tokens = word_tokenizer.tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    tokens = [word.lower() for word in tokens]
    filtered_tokens = filter(lambda x: x not in stop_words, tokens)

    for token in filtered_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_token = lemmatizer.lemmatize(clean_token, pos=NOUN)
        clean_token = lemmatizer.lemmatize(clean_token, pos=ADJ)
        clean_token = lemmatizer.lemmatize(clean_token, pos=VERB)
        clean_tokens.append(clean_token)

    return clean_tokens

def build_model():
    '''
    Machine Learning classification model function that executes following steps:
      1. Building Machine Learning pipeline
      2. Running GridSearchCV for Hyper-parameter tunning
      
      input: None
    output: GridSearch best model.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_extract', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())]))])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=42), n_jobs=-1))
    ])
    
    # Taking the n_estimator to 10 as faced error: "remote: warning: File models/classifier.pkl is 86.78 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB" while pushing to github repository
    
    parameters = {
                'features__text_extract__vect__ngram_range': [(1, 1), (1, 2)],
                'features__text_extract__tfidf__smooth_idf':[True, False],
                'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__min_samples_split': [2, 3, 4],
                'clf__estimator__criterion': ['gini']
             }
             
            
    clf_grid_best = GridSearchCV(pipeline, param_grid= parameters, verbose=5, n_jobs=-1, scoring= 'f1_macro', cv = 3, refit=True, return_train_score=True)
    
    return clf_grid_best

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Funtion that evaluates and logs model performance
    
    Input
    ----------
    model : best GridsearchCV multiclassification model
    X_test: numpy.ndarray, test dataset of features.
    Y_test: numpy.ndarray, test dataset with dependent feature 
    category_names: list, Y labels.
    
    Output: None
    '''
    reports = []
    Y_pred = model.predict(X_test)

    for idx, target in enumerate(category_names):
        reports.append(classification_report(Y_test[:, idx].tolist(),
                                             Y_pred[:, idx].tolist(),
                                             labels=[0],
                                             target_names=[target]))

    for report in reports:
        print('***********************************')
        print(report)
        
    

def save_model(model, model_filepath):
    '''
    Save joblib file of modwel
    input:
        model - trained GridsearchCV multiclassification model, 
        model_filepath - path to store the pickle serialized model
    
    output:  None
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('...Model Building...')
        model = build_model()

        print('...Training of Model...')
        model.fit(X_train, Y_train)

        print('...Evaluating the Model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Saved serialized Trained Model!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()