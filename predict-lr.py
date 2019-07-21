from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import commons

# %%
data = commons.get_data()

# %%
X_train, X_test, y_train, y_test = commons.prepare_ds(data)
X_train, y_train = commons.oversampling(X_train, y_train)

# %%
params = {
    "C": range(1, 10, 1),
    # "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "penalty": ['l1', 'l2']
}
est = LogisticRegression()
clf = GridSearchCV(est, params, cv=5, verbose=5, n_jobs=3)
clf.fit(X_train, y_train)
# %%
# svm(X_train, X_test, y_train, y_test)
