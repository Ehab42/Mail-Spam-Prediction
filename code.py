from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target'] == 'spam', 1, 0)


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)


def add_feature(feature, train):
    X_train_dtm = hstack((train, feature))
    return X_train_dtm


def answer_one():

    num_of_spam_entries = len(spam_data[spam_data['target'] == 1])
    spam_percentage = 100 * (num_of_spam_entries/len(spam_data))
    return spam_percentage


# print(answer_one())


def answer_two():

    vect = CountVectorizer().fit(X_train)
    longest_token = max(vect.get_feature_names(), key=len)
    return longest_token


# print(answer_two())


def answer_three():

    # Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
    count_vect = CountVectorizer().fit(X_train)
    count_vect_transformed = count_vect.transform(X_train)

    # fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`
    model = MultinomialNB(alpha=0.1).fit(count_vect_transformed, y_train)

    # Find the area under the curve (AUC) score
    predictions = model.predict(count_vect.transform(X_test))
    auc = roc_auc_score(y_test, predictions)
    return auc


# print(answer_three())

def answer_five():

    # Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
    tfidf = TfidfVectorizer(min_df=3).fit(X_train)
    tfidf_vectorized = tfidf.transform(X_train)

    # fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`
    model = MultinomialNB(alpha=0.1).fit(tfidf_vectorized, y_train)

    # Find the area under the curve (AUC) score
    predictions = model.predict(tfidf.transform(X_test))
    auc = roc_auc_score(y_test, predictions)
    return auc


# print(answer_five())

def answer_six():

    avg_len_nospam = (spam_data[spam_data['target']
                                == 0]['text'].str.len()).mean()
    avg_len_spam = (spam_data[spam_data['target'] == 1]
                    ['text'].str.len()).mean()
    return avg_len_nospam, avg_len_spam


# print(answer_six())

def answer_seven():

    # Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
    tfidf = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_tf = tfidf.transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    # Add Document length as a feature
    X_train_tf = add_feature(X_train.str.len().values[:, None], X_train_tf)
    X_test_tf = add_feature(X_test.str.len().values[:, None], X_test_tf)

    # Fit a Support Vector Classification model with regularization C=10000
    model = SVC(C=10000).fit(X_train_tf, y_train)

    # Compute the area under the curve (AUC) score using the transformed test data
    predictions = model.predict(X_test_tf)
    auc = roc_auc_score(y_test, predictions)

    return auc


answer_seven()


def answer_eight():

    return spam_data[spam_data['target'] == 0]['text'].str.count(r'\d').mean(), spam_data[spam_data['target'] == 1]['text'].str.count(r'\d').mean()


answer_eight()


def answer_nine():

    # Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5
    #  and using word n-grams from n=1 to n=3
    tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train)
    X_train_tf = tfidf.transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    # # # Add Document length as a feature
    # X_train_tf = add_feature(X_train.str.len().values[:, None], X_train_tf)
    # X_test_tf = add_feature(X_test.str.len().values[:, None], X_test_tf)

    # Add Number of digits per document as a feature
    X_train_tf = add_feature(X_train.str.count(
        r'\d').values[:, None], X_train_tf)
    X_test_tf = add_feature(X_test.str.count(
        r'\d').values[:, None], X_test_tf)

    # Fit a Logistic Regression model with regularization C=100
    model = LogisticRegression(C=100).fit(X_train_tf, y_train)

    # Find the area under the curve (AUC) score    predictions = model.predict(X_test_tf)
    predictions = model.predict(X_test_tf)
    auc = roc_auc_score(y_test, predictions)

    return auc


answer_nine()


def answer_ten():

    return spam_data[spam_data['target'] == 0]['text'].str.count(r'\W').mean(), spam_data[spam_data['target'] == 1]['text'].str.count(r'\W').mean()


answer_ten()
