from imblearn.over_sampling import SMOTE


def oversampling(X, y):
    os = SMOTE(random_state=42, k_neighbors=4)
    return os.fit_resample(X, y)
