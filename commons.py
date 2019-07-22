import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

import preprocessing as pp


def get_data():
    column_names = ["Recency", "Frequency", "Amount", "Times", "Donation"]
    return pd.read_csv("transfusion.data", names=column_names, header=0)


def preprocessing(data):
    data = pp.outliners(data)
    return data


def transform(train_data, test_data):
    transformer = PowerTransformer(standardize=False)
    # transformer = StandardScaler()
    x_train = transformer.fit_transform(train_data)
    x_test = transformer.transform(test_data)
    return x_train, x_test, transformer


def oversampling(X, y):
    os = SMOTE(random_state=42, k_neighbors=3)
    return os.fit_resample(X, y)


# TODO add feature importance
def feature_engineering(data):
    data['TimesBeforeLast'] = data.apply(lambda x: x['Times'] - x['Recency'], axis=1)
    data['FrequencyBeforeLast'] = data.apply(lambda x: (x['Times'] - x['Recency']) / x['Frequency'], axis=1)
    return data


def prepare_ds(data):
    X = data.iloc[:, [0, 1, 3]].values
    y = data.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    x_train, x_test, _ = transform(x_train, x_test)
    return x_train, x_test, y_train, y_test


def score_model(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
