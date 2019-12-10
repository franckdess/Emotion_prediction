import sklearn
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from tokenizer import tokenizer
from sklearn.linear_model import *
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from functions import *

# Import the small dataset
print('Importing small datasets\n')
tweet_pos, tweet_neg, tweet_test = import_data(full = False)

# Import stopwords
print('Importing stopwords\n')
stopw = pd.read_csv('data/twitter-stopwords.txt').values.flatten().tolist()

# Construct train and test sets
print('Building train and test sets\n')
tweet_TR = construct_train_set(tweet_pos, tweet_neg)
tweet_TE = construct_test_set(tweet_test)

# Create TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopw, tokenizer=tokenize, ngram_range=(1,4), min_df=10)

# Apply TfidfVectorizer to the small train set
print('Applying TfidfVectorizer\n')
X = vectorizer.fit_transform(tweet_TR.values[:, 0])
Y = tweet_TR.values[:, 1].astype(int)

# Split the small train set for local accuracy computation
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Define a standard classifier
print('Dataset: Small\nClassifier: LinearSVC\nParameters: Standard')
clf = LinearSVC(random_state=42)
clf.fit(x_train, y_train)

# Compute the predicitions of x_test
y_pred = clf.predict(x_test)

# Compute the standard classifier's accuracy (LinearSVC)
print('Accuracy: {:2.2%}\n'.format(accuracy_score(y_pred, y_test)))

# Define the list of parameters to test
losses = ['hinge', 'squared_hinge']
tols = [1e-5, 1e-4, 1e-3]
Cs = [0.1, 1, 10]

# Create the parameter grid
param_grid = {'loss': losses, 'tol': tols, 'C': Cs}

# Find the best parameters
print('5-fold GridSearchCV for optimal parameters\n')
best_parameters = param_selection(x_train, y_train, 5, param_grid, clf)

# Apply the best parameters
loss_opt = best_parameters['loss']
tol_opt = best_parameters['tol']
C_opt = best_parameters['C']

# Print best parameters
print('Best parameters:\nloss: {}\ntol: {}\nC: {}\n'.format(loss_opt, tol_opt, C_opt))                       

# Create a new classifier with the optimal parameters
print('Dataset: Small\nClassifier: LinearSVC\nParameters: Optimal')
clf_optimal = LinearSVC(C=C_opt, tol=tol_opt, loss=loss_opt, random_state=42)
model_optimal = clf_optimal.fit(x_train, y_train)

# Compute the predictions of x_test
y_pred_optimal = model_optimal.predict(x_test)

# Compute the optimal classifier's accuracy (LinearSVC)
print('Accuracy: {:2.2%}\n'.format(accuracy_score(y_pred_optimal, y_test)))

# Import the full dataset
print('Importing full dataset\n')
tweet_pos_full, tweet_neg_full, tweet_test_full = import_data(full = True)

# Construct train and test set
print('Building train and test sets\n')
tweet_TR_full = construct_train_set(tweet_pos_full, tweet_neg_full)
tweet_TE_full = construct_test_set(tweet_test_full)

# Apply TfidfVectorizer on the full train set
print('Applying TfidfVectorizer\n')
X_full = vectorizer.fit_transform(tweet_TR_full.values[:, 0])
Y_full = tweet_TR_full.values[:, 1].astype(int)

# Split the full train set for local accuracy computation
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(X_full, Y_full, test_size = 0.2)

# Apply the optimal classifier on the full dataset
print('Dataset: Full\nClassifier: LinearSVC\nParameters: Optimal')
model_optimal_full = clf_optimal.fit(x_train_full, y_train_full)

# Compute the predictions of x_test
y_pred_full = model_optimal_full.predict(x_test_full)

# Compute the optimal classifier's accuracy (LinearSVC)
print('Accuracy: {:2.2%}\n'.format(accuracy_score(y_pred_full, y_test_full)))

print('Building final submission')
# Apply TfidfVectorizer to the test set
X_TE = vectorizer.transform(tweet_TE)

# Apply the optimal classifier to the test set
y_pred_TE = model_optimal.predict(X_TE)

# Build submission to be submitted on CrowdAi
build_submission(y_pred_TE, 'final_submission')