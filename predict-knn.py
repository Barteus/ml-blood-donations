from sklearn.neighbors import KNeighborsClassifier

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
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# %%
commons.score_model(y_test, y_pred)
