# TODO improve by using xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import make_dataset
from src.models import eval

# %%
data = make_dataset.get_processed_data("transfusion_2_oversampled.csv")
# data = make_dataset.get_processed_data("transfusion_1.csv")

# %%

x = data.drop('Donation', axis=1)
y = data.Donation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# %%
def rf(x_train, x_test, y_train, y_test):
    classifier = RandomForestClassifier(random_state=0, n_estimators=200)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    eval.score_model(y_test, y_pred)
    return y_pred


rf(x_train, x_test, y_train, y_test)
