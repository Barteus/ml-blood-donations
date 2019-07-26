from sklearn import svm
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


# %% SVM method
def svc(x_train, x_test, y_train, y_test):
    print("----------SVC----------")
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    eval.score_model(y_test, y_pred)
    # TODO create nice printing of errors for analysis
    # errors = eval.get_error_examples(pd.DataFrame(x_test), y_test, y_pred)
    # print(errors)
    return y_pred


# %%
svc(x_train, x_test, y_train, y_test)
