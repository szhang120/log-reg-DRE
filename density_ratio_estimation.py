import numpy as np
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, X_test):
    combined_X = np.concatenate((X_train, X_test)).reshape(-1, 1)
    labels = np.concatenate((np.ones(len(X_train)), np.zeros(len(X_test))))
    classifier = LogisticRegression()
    classifier.fit(combined_X, labels)
    return classifier

def estimate_density_ratio(x, classifier):
    prob_Q1 = classifier.predict_proba(x.reshape(-1, 1))[0, 1]
    return prob_Q1 / (1 - prob_Q1)

