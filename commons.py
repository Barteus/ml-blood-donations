import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


def get_data():
    column_names = ["Recency", "Frequency", "Amount", "Times", "Donation"]
    return pd.read_csv("transfusion.data", names=column_names, header=0)


def transform(train_data, test_data):
    transformer = PowerTransformer()
    X_train = transformer.fit_transform(train_data)
    X_test = transformer.transform(test_data)
    return X_train, X_test, transformer


def oversampling(X, y):
    os = SMOTE(random_state=42, sampling_strategy='minority')
    # os = BorderlineSMOTE(random_state=42)
    return os.fit_resample(X, y)

# TODO add feature importance
def feature_engineering(data):
    data['TimesBeforeLast']=data.apply(lambda x : x['Times']-x['Recency'], axis=1)
    data['FrequencyBeforeLast']=data.apply(lambda x : (x['Times']-x['Recency'])/x['Frequency'], axis=1)
    data['AvgAmount']=data.apply(lambda x : x['Amount']/x['Frequency'], axis=1)

def prepare_ds(data):
    X = data.iloc[:, 0:4].values
    y = data.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, _ = transform(X_train, X_test)
    return X_train, X_test, y_train, y_test


def score_model(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(f"accuracy_score={accuracy_score(y_true, y_pred)}")
    print(classification_report(y_true, y_pred))
