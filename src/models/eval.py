import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def score_model(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def get_error_examples(data, y_true, y_pred):
    idxs = np.add(np.where(np.logical_xor(y_pred, y_true) == True),1)
    return data.iloc[np.hstack(idxs),:]
