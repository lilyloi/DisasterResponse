# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import string 
import sys 

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB

def load_data(databse_filepath):
	"""
	Load database and get dataset
	Args: 
		database_filepath (str): file path of sqlite database

	Return:
		X (pandas dataframe): Features
		y (pandas dataframe): Targets/ Labels
	"""
	engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DS_messages', engine)
    X = df['message']
    y = df[df.columns[4:]]
    return X, y

def tokenize(text):
    # normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmed

 def build_model():
    """Returns the GridSearchCV model
    Args:
        None
    Returns:
        cv: Grid search model object
    """

    clf = RandomForestClassifier(n_estimators=100)

    # The pipeline has tfidf, dimensionality reduction, and classifier
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('best', TruncatedSVD(n_components=100)),
                    ('clf', MultiOutputClassifier(clf))
                      ])

    # Parameters for GridSearchCV
    param_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': [0.8, 1.0],
        'tfidf__max_features': [None, 10000],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid, cv=3, verbose=10), n_jobs=-1)

    return cv

def model_evaluation(model, X_test, Y_test, categories):
	"""
	Print multi-ouput classification results
	Args: 
		model (dataframe): the selected model in step above 
		X_test: features of test set 
		y_test: labels of test set 
		categories: list of categorical labels
	Return:
		None
	"""

	# Predict the model based on test set 
	y_pred = model.predict(X_test)

	# Print out model evaluation metrics via classification report
	print(classification_report(y_test, y_pred, target_names=categories))


def save_model(model, model_filepath):
	"""

	"""

def main():
	if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
