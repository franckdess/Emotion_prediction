# Emotion prediction

This project is divided into three parts.

#### First part

The goal of this part is to predict whether the tweets present in the tweet_test.txt file reflect a positive or a negative sentiment. Here we have access to train datasets and test datasets.

#### Second part

The goal of the second part, is the same as the first one, i.e. to predict whether a tweet reflects a positive, negative or neutral sentiment. Here we won't have access to train datasets. We will reuse the train dataset from the previous part only to gather the tweets, but not the prediction.

#### Third part

The third part aims to extract five emotions from the tweets dataset.

#### Getting Started

The repository contains everything needed to run the .py and .ipynb files directly. It contains:

- data folder: it contains:
	- train_pos.txt:			train set of positive tweets - 100000 tweets
	- train_neg.txt: 			train set of negative tweets - 100000 tweets
	- train_pos_full.txt:		bigger train set of positive tweets - 1250000 tweets
	- train_neg_full.txt:		bigger train set of negative tweets - 1250000 tweets
	- test_data.txt:			test set of tweets for which we want to predict the sentiment - 10000 tweets
	- twitter-stopwords.txt:	list of stop words related to social media context
- functions.py: implementation of all functions used in the .ipynb file.
- emotion_prediction.ipynb: a documented jupyter notebook where the three parts described above are implemented.

#### Prerequisites

Before running the script described above, make sure that the following libraries are correctly installed:
- sklearn
- pandas
- nltk
- numpy
- tokenizer
- vaderSentiment

If this is not the case, run the following command:
	`$pip install library_name`
where `library_name` is the name of the missing library.

#### Understanding the code

In both functions.py and emotion_prediction.ipynb files, the implementation of every function as well as every step are precisely described. Please refer to the corresponding file to have the complete explanation.
