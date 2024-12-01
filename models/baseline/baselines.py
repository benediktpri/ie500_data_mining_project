from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle


class PCANNModel:
    def __init__(self, n_components=1, n_neighbors=5):
        """
        Initialize the PCA + KNN model.
        """
        self.pca = PCA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        """
        Fit PCA and KNN on the training data.
        """
        # Apply PCA
        self.X_train_pca = self.pca.fit_transform(X_train)
        # Train K-NN on the PCA-transformed data
        self.knn.fit(self.X_train_pca, y_train)

    def predict(self, X):
        """
        Predict labels for new data.
        """
        # Transform data using PCA
        X_pca = self.pca.transform(X)
        # Predict using K-NN
        return self.knn.predict(X_pca)

    def evaluate(self, X, y_true):
        """
        Evaluate the model on validation data.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return accuracy, report

    def explained_variance(self):
        """
        Get the explained variance ratio of the PCA components.
        """
        return self.pca.explained_variance_ratio_


import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


class OneFeatureModel:
    def __init__(self, feature, threshold):
        """
        Initialize with the selected feature and threshold.
        """
        self.feature = feature
        self.threshold = threshold

    def predict(self, X):
        """
        Predict labels based on the feature and threshold.
        """
        return X[self.feature].apply(lambda x: 1 if x >= self.threshold else 0)

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on validation data.
        """
        y_val_pred = self.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        report = classification_report(y_val, y_val_pred)
        return accuracy, report
