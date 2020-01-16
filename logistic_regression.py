import regex as re
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
from pandas import *

def read_lines(file):
    """
    Creates a list of sentence from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences

def split_lines(lines, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    for line in lines:
        rows = line.split('\n')
        line = [dict(zip(column_names, row.split('\t'))) for row in rows]
    return line

def read_lines2(file):
    """
    Creates a list of sentence from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    return f

def split_lines2(lines, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    rows = lines.split('\n')
    line = [dict(zip(column_names, row.split(','))) for row in rows]

    return line


def one_hot_encode(list_of_tweets, token_index):
    max_length = len(token_index)
    results = np.zeros(shape=(len(list_of_tweets), max(token_index.values()) + 1))

    for i, tweet in enumerate(list_of_tweets):
        for word in tweet.split():
            index = token_index.get(word)
            results[i, index] = 1.
    return results

train_file = 'olid-training-v1.0.tsv' #set the train file path
test_file = 'testset-levela.tsv' #set the test file path
test_labels_a = 'labels-levela.csv' #set the test labels file path
column_names = ['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c']
column_names_test = ['id', 'tweet']
lines = read_lines(train_file)
formatted_corpus = split_lines(lines, column_names)  #list of dictionaries where each dictionary is the info of one tweet
formatted_corpus = formatted_corpus[1:] #delete column names

lines = read_lines(test_file)
formatted_corpus_test = split_lines(lines, column_names_test)  #list of dictionaries where each dictionary is the info of one tweet
formatted_corpus_test = formatted_corpus_test[1:] #delete column names

lines = read_lines2(test_labels_a)
formatted_corpus_labels_a = split_lines2(lines, ['id', 'label'])  #list of dictionaries where each dictionary is the info of one tweet

# TRAINING DATA
list_of_tweets = []
y = []

for tweet in formatted_corpus:
    sentence = tweet['tweet'].lower()
    list_of_tweets.append(sentence)
    if tweet['subtask_a'] == 'OFF':
        y.append(1)
    else:
        y.append(0)

token_index = {} #index of unique words
for tweet in list_of_tweets:
    for word in tweet.split():
        if word not in token_index:
            token_index[word] = len(token_index)

# TESTING DATA
list_of_tweets_test = []

for tweet in formatted_corpus_test:
    sentence = tweet['tweet'].lower()
    list_of_tweets_test.append(sentence)

for tweet in list_of_tweets_test:
    for word in tweet.split():
        if word not in token_index:
            token_index[word] = len(token_index)

y_true_a = []

for tweet in formatted_corpus_labels_a:
    if tweet['label'] == 'NOT':
        y_true_a.append(0)
    else:
        y_true_a.append(1)

X = one_hot_encode(list_of_tweets, token_index) #x_train
X_test = one_hot_encode(list_of_tweets_test, token_index) #x_test

classifier = LogisticRegression(penalty='l2', dual=True, solver='liblinear', verbose=1)
model = classifier.fit(X, y)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_true_a, y_pred, target_names = ['NOT', 'OFF'], digits=4))
print(metrics.confusion_matrix(y_true_a, y_pred))
