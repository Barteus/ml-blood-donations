from sklearn.linear_model import LogisticRegression

import commons

# %%
data = commons.get_data()

# %%
data = commons.preprocessing(data)
data = commons.feature_engineering(data)
# %%
X_train, X_test, y_train, y_test = commons.prepare_ds(data)
X_train, y_train = commons.oversampling(X_train, y_train)

# %%
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# %%
commons.score_model(y_test, y_pred)
