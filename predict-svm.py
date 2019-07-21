from sklearn import svm

import commons

# %%
data = commons.get_data()

# %%
commons.feature_engineering(data)
# %%
X_train, X_test, y_train, y_test = commons.prepare_ds(data)
X_train, y_train = commons.oversampling(X_train, y_train)


# %% SVM method
def svc(X_train, X_test, y_train, y_test):
    print("----------SVC----------")
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    commons.score_model(y_test, y_pred)


# %%
svc(X_train, X_test, y_train, y_test)
