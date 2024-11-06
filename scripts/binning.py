import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, labels=None, include_lowest=True, right=False):
        self.bins = bins
        self.labels = labels
        self.include_lowest = include_lowest
        self.right = right

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_binned = pd.cut(
            X.iloc[:, 0],
            bins=self.bins,
            labels=self.labels,
            include_lowest=self.include_lowest,
            right=self.right,
        )
        return X_binned.to_frame()
