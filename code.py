from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target'] == 'spam', 1, 0)


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)


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


def answer_four():

    vect = TfidfVectorizer().fit(X_train)
    # A list of all feature names
    feature_names = np.array(vect.get_feature_names())
    tfidf_values = vect.transform(X_train).max(
        0).toarray()[0]  # Sorted Tfidf values
    smallest_20tfidf_features = feature_names[tfidf_values.argsort()[
        :20]]
    largest_20tfidf_features = feature_names[tfidf_values.argsort()[
        :-21:-1]]
    smallest_tfidfs_series = pd.Series(
        smallest_20tfidf_features, index=sorted(tfidf_values)[0:20])
    largest_tfidfs_series = pd.Series(
        largest_20tfidf_features, index=sorted(tfidf_values)[-21:-1])
    return smallest_tfidfs_series, largest_tfidfs_series


print(answer_four())
