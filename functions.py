import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import re

def import_data(full=False):
    """ Import the test tweets, the train tweets positive and the train
        tweets negative. Import small sets of full sets depending if the
        'full' parameter is set to True of False.
        Return pandas dataframe for each of the 3 sets."""
    if(full):
        tweet_pos = pd.read_csv('data/train_pos_full.txt', header = None, sep = "\r\n", engine = 'python')
        tweet_neg = pd.read_csv('data/train_neg_full.txt', header = None, sep = "\r\n", engine = 'python')
    else:
        tweet_pos = pd.read_csv('data/train_pos.txt', header = None, sep = "\r\n", engine = 'python')
        tweet_neg = pd.read_csv('data/train_neg.txt', header = None, sep = "\r\n", engine = 'python')
    tweet_test = pd.read_csv('data/test_data.txt', header = None, sep = "\r\n", engine = 'python')
    return tweet_pos, tweet_neg, tweet_test

def construct_train_set(tweet_pos, tweet_neg):
    """ Construct a dataframe of all tweets from tweet_pos
        and tweet_neg with the associated sentiment prediction
        0 if negative and 1 if positive.
        Return the dataframe tweet_TR."""
    tweet_pos['pred'] = 1
    tweet_neg['pred'] = 0
    tweet_pos.columns = ['tweet', 'pred']
    tweet_neg.columns = ['tweet', 'pred']
    all_tweets = tweet_neg.append(tweet_pos)
    tweet_TR = all_tweets.reset_index().drop(['index'], axis = 1)
    return tweet_TR

def construct_test_set(tweet_test):
    """ Construct a list of all tweets from tweet_test.
        Return the list tweet_TE."""
    tweet_clean = clean_data(tweet_test.values)
    np.reshape(tweet_clean, (10000,))
    tweet_TE = tweet_clean.flatten()
    return tweet_TE

def clean_data(array):
    """ Clean the data by deleting the id
        placed in the front of the tweet."""
    ret = np.zeros(len(array))
    for i in range(len(array)):
        drop_id = len(str(i+1)) + 1
        array[i, 0] = array[i, 0][int(drop_id):]
    return array

def tokenize(t):
    """ Customized tokenizer called by TfidfVectorizer.
        Tokenize tweets using TweetTokenizer and lemmatize
        each token with WordNetLemmatizer."""
    tweet_tok = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tweet_tok.tokenize(t)
    wnl = WordNetLemmatizer()
    stems = []
    for item in tokens:
        stems.append(wnl.lemmatize(item))
    return stems

def param_selection(X, Y, nfolds, param_grid, classifier):
    """ Given the features and the predicitons, the number of cross validation,
        the parameter grid and the classifier, return the best parameters."""
    grid_search = GridSearchCV(classifier, param_grid, cv = nfolds)
    grid_search.fit(X, Y)
    grid_search.best_params_
    return grid_search.best_params_

def zero_to_neg(array):
    """ Given an array of 0 and 1, transform it into
        an array of -1 and 1. """
    ret = np.ones(len(array))
    for i in range(len(array)):
        if(array[i] == 0):
            ret[i] = -1
    return ret

def build_submission(y_pred, id_submission):
    """ Build submission and save it into the
        folder 'prediction' with id 'id_submission'."""
    y_pred_ = zero_to_neg(y_pred)
    ret = np.ones((len(y_pred_), 2))
    for i in range(len(y_pred_)):
        ret[i] = np.array([i+1, y_pred_[i]])
    ret = ret.astype(int)
    sub = pd.DataFrame(data = ret)
    sub.columns = ['Id', 'Prediction']
    sub.to_csv('pred_' + id_submission + '.csv', index=None)
    
def get_prediction_from_score(score):
    """ This function returns the prediction given the
        compound score of the tweet. """
    if(score >= 0.03):
        return 'Positive'
    elif(score <= -0.03):
        return 'Negative'
    else:
        return 'Neutral'