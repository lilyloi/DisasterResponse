import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB

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

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

	def starting_verb(self, text):
		sentence_list = nltk.sent_tokenize(text)
		for sentence in sentence_list:
			pos_tags = nltk.pos_tag(tokenize(sentence))
			first_word, first_tag = pos_tag[0]
			if first_tag in ['VB', 'VBP'] or first_word == 'RT':
				return True
			return False

	def fit(self, x, y=None):
		return self

	def transform(self, X):
		X_tagged = pd.Series(X).apply(self.starting_verb)
		return pd.DataFrame(X_tagged)

# Build model pipeline 
def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(BernoulliNB()))
    ])

    parameters = {
    	'features__text_pipeline__vect__max_df': (0.5),
    	'features__text_pipeline__vect__ngram_range': (1,2),
    	'features__text_pipeline__vect__max_features': (10000),
    	'features__text_pipeline__tfidf__use_idf': (True, False),
    	'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

