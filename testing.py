import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

# allows us to select a column by name from a data frame and return it as a nparray of type string


class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def transform(self, data):
        return np.asarray(data[self.column]).astype(str)

    # not used
    def fit(self, *_):
        return self

        # allows us to select a column by name from a data frame and return it as a nparray of the specified type


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column, dtype):
        self.dtype = dtype
        self.column = column

    def transform(self, data):
        data = np.asarray(data[self.column]).astype(self.dtype)

        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        # reshapes from row vector to column vector
        data = data.reshape(data.size, 1)
        return data

    # not used
    def fit(self, *_):
        return self

###########################################################################
###########################################################################

comments = pd.read_pickle('toxic_comment_data/combined_train_data.csv')

char_ngram_features = Pipeline([
    ('extractor', TextExtractor('Comment')),
    ('vectorizer', CountVectorizer(analyzer='char')),
    ('tfidf', TfidfTransformer())
])

word_ngram_features = Pipeline([
    ('extractor', TextExtractor('Comment')),
    ('vectorizer', CountVectorizer(analyzer='word')),
    ('tfidf', TfidfTransformer())
])

# the features combined
features = FeatureUnion([
    ('char_ngrams', char_ngram_features),
    ('word_ngrams', word_ngram_features),
    ('norm_length', ColumnExtractor('Norm_True_Length',float)),
    ('weekday', ColumnExtractor('Weekday',int)),
    ('day', ColumnExtractor('Day',int)),
    ('month', ColumnExtractor('Month',int)),
    ('year', ColumnExtractor('Year',int)),
    ('hour', ColumnExtractor('Hour',int)),
    ('minute', ColumnExtractor('Minute',int)),
    ('second', ColumnExtractor('Second',int))
])

# set the classifier
# --** this is where to change the classifier **--
classifier = SVC()

# set the feature selection (these params can also be grid searched)
select = SelectPercentile(score_func=chi2, percentile=25)

pipeline = Pipeline([
    ('union', features),
    ('select', select),
    ('classifier', classifier)
])

parameters = {'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
              'classifier__C': [1, 10],
              'union__char_ngrams__vectorizer__ngram_range': [(1, 1), (1, 2)],
              'union__char_ngrams__vectorizer__stop_words': [None, 'english'],  # remove stopwords or not
              'union__char_ngrams__tfidf__use_idf': (True, False),  # use idf or not
              'union__word_ngrams__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'union__word_ngrams__vectorizer__stop_words': [None, 'english'],
              'union__word_ngrams__tfidf__use_idf': (True, False),
              # 'classifier__gamma': [1, 5],
              'classifier__degree': [3, 5],
              'select__percentile': [1, 5, 10, 20, 50]  # percentage of best features to select
              }
# {'union__char_ngrams__vectorizer__ngram_range': [(1, 1), (1, 2)], # the range of the n-grams
#               'kernel': ('rbf', 'linear')
#              }
# the names are important, e.g. vect__ corresponds to the name given to the class in the pipeline,
# while "ngram_range" is a parameter for the class, i.e. CountVectorizer

gs_clf = GridSearchCV(pipeline, parameters)   # run the grid search on the chosen pipeline
print("Fitting...")
gs_clf = gs_clf.fit(comments, comments.Insult)

print(gs_clf.best_score_)
print(gs_clf.best_params_)