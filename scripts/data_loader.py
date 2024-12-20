import pandas as pd
import os


class DataLoader:
    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # Get the directory of the current script

    train_file = os.path.join(base_path, "../data/preprocessed/dataset_train.csv")
    test_file = os.path.join(base_path, "../data/preprocessed/dataset_test.csv")
    validation_file = os.path.join(base_path, "../data/preprocessed/dataset_val.csv")

    train_file_pca = os.path.join(base_path, "../data/pca/dataset_train_pca.csv")
    test_file_pca = os.path.join(base_path, "../data/pca/dataset_test_pca.csv")
    validation_file_pca = os.path.join(base_path, "../data/pca/dataset_val_pca.csv")

    train_file_oversampling_random = os.path.join(
        base_path, "../data/resampling/dataset_train_oversampled.csv"
    )
    train_file_oversampling_smote = os.path.join(
        base_path, "../data/resampling/dataset_train_oversampled_smote.csv"
    )
    train_file_undersampling_random = os.path.join(
        base_path, "../data/resampling/dataset_train_undersampled.csv"
    )
    train_file_resampling_smote_tomek = os.path.join(
        base_path, "../data/resampling/dataset_train_smote_tomek.csv"
    )

    def __init__(self):
        self.train_file = DataLoader.train_file
        self.test_file = DataLoader.test_file
        self.validation_file = DataLoader.validation_file
        self.train_file_pca = DataLoader.train_file_pca
        self.test_file_pca = DataLoader.test_file_pca
        self.validation_file_pca = DataLoader.validation_file_pca
        self.train_file_oversampling_random = DataLoader.train_file_oversampling_random
        self.train_file_oversampling_smote = DataLoader.train_file_oversampling_smote
        self.train_file_undersampling_random = (
            DataLoader.train_file_undersampling_random
        )
        self.train_file_resampling_smote_tomek = (
            DataLoader.train_file_resampling_smote_tomek
        )

    def load_data(self, file_path, separate_target=True):
        data = pd.read_csv(file_path)
        if separate_target:
            X = data.drop(columns=["Diabetes"])
            y = data["Diabetes"]
            return X, y
        else:
            return data

    @property
    def training_data(self):
        return self.load_data(self.train_file)

    @property
    def test_data(self):
        return self.load_data(self.test_file)

    @property
    def validation_data(self):
        return self.load_data(self.validation_file)

    @property
    def training_dataframe(self):
        return self.load_data(self.train_file, separate_target=False)

    @property
    def test_dataframe(self):
        return self.load_data(self.test_file, separate_target=False)

    @property
    def validation_dataframe(self):
        return self.load_data(self.validation_file, separate_target=False)

    def load_data_pca(self, file_path, n=None, separate_target=True):
        data = pd.read_csv(file_path)
        if separate_target:
            X = data.drop(columns=["Diabetes"])
            y = data["Diabetes"]
            if n:
                X = X.iloc[:, :n]
            return X, y
        else:
            if n:
                columns = data.columns[:n].tolist()
                if "Diabetes" not in columns:
                    columns.append("Diabetes")
                data = data[columns]
            return data

    def training_data_pca(self, n=None):
        return self.load_data_pca(self.train_file_pca, n)

    def test_data_pca(self, n=None):
        return self.load_data_pca(self.test_file_pca, n)

    def validation_data_pca(self, n=None):
        return self.load_data_pca(self.validation_file_pca, n)

    def training_dataframe_pca(self, n=None):
        return self.load_data_pca(self.train_file_pca, n, separate_target=False)

    def test_dataframe_pca(self, n=None):
        return self.load_data_pca(self.test_file_pca, n, separate_target=False)

    def validation_dataframe_pca(self, n=None):
        return self.load_data_pca(self.validation_file_pca, n, separate_target=False)

    @property
    def training_data_oversampling_random(self):
        return self.load_data(self.train_file_oversampling_random)

    @property
    def training_data_oversampling_smote(self):
        return self.load_data(self.train_file_oversampling_smote)

    @property
    def training_data_undersampling_random(self):
        return self.load_data(self.train_file_undersampling_random)

    @property
    def training_dataframe_oversampling_random(self):
        return self.load_data(
            self.train_file_oversampling_random, separate_target=False
        )

    @property
    def training_dataframe_oversampling_smote(self):
        return self.load_data(self.train_file_oversampling_smote, separate_target=False)

    @property
    def training_dataframe_undersampling_random(self):
        return self.load_data(
            self.train_file_undersampling_random, separate_target=False
        )

    @property
    def training_data_resampling_smote_tomek(self):
        return self.load_data(self.train_file_resampling_smote_tomek)

    @property
    def training_dataframe_resampling_smote_tomek(self):
        return self.load_data(
            self.train_file_resampling_smote_tomek, separate_target=False
        )
