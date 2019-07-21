# TODO improve by using xgboost

from sklearn.ensemble import RandomForestClassifier

import commons

# %%
data = commons.get_data()

# %%
X_train, X_test, y_train, y_test = commons.prepare_ds(data)
X_train, y_train = commons.oversampling(X_train, y_train)

# %%
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %%
commons.score_model(y_test, y_pred)
