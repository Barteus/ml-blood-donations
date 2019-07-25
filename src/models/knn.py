from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.data import make_dataset
from src.features import clean
from src.models import eval

# %%
data = make_dataset.get_processed_data("transfusion_1.csv")

# %%

x = data.drop('Donation', axis=1)
y = data.Donation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x_train, y_train = clean.oversampling(x_train, y_train)


# %%
def knc(x_train, x_test, y_train, y_test):
    knc = KNeighborsClassifier(n_neighbors=4)
    knc.fit(x_train, y_train)
    y_pred = knc.predict(x_test)
    eval.score_model(y_test, y_pred)
    return y_pred


knc(x_train, x_test, y_train, y_test)
