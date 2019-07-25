import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def model(data, target):
    pca_model = PCA(n_components=2)
    principal_components = pca_model.fit_transform(data)
    df_pca = pd.DataFrame(principal_components, columns=['x', 'y'])
    df_pca['Donation'] = target
    return df_pca


def plot(df_pca):
    sns.lmplot(x='x', y='y', hue='Donation', data=df_pca, fit_reg=True, scatter_kws={'alpha': 0.5})
