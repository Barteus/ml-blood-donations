from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import commons

# %%
data = commons.get_data()

# %%
data = commons.preprocessing(data)
data = commons.feature_engineering(data)
# %%
X_train, X_test, y_train, y_test = commons.prepare_ds(data)
X_train, y_train = commons.oversampling(X_train, y_train)


# %% SVM method
def score(X_train, X_test, y_train, y_test):
    print("----------GaussianNB----------")
    clf =  MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    commons.score_model(y_test, y_pred)


# %%
score(X_train, X_test, y_train, y_test)
