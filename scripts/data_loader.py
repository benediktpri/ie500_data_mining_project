import pandas as pd


class DataLoader:
    train_file = "../data/preprocessed/dataset_train.csv"
    test_file = "../data/preprocessed/dataset_test.csv"
    validation_file = "../data/preprocessed/dataset_val.csv"

    train_file_pca = "../data/pca/dataset_train_pca.csv"
    test_file_pca = "../data/pca/dataset_test_pca.csv"
    validation_file_pca = "../data/pca/dataset_val_pca.csv"

    def __init__(self):
        self.train_file = DataLoader.train_file
        self.test_file = DataLoader.test_file
        self.validation_file = DataLoader.validation_file
        self.train_file_pca = DataLoader.train_file_pca
        self.test_file_pca = DataLoader.test_file_pca
        self.validation_file_pca = DataLoader.validation_file_pca

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

    @property
    def training_data_pca(self):
        return self.load_data(self.train_file_pca)

    @property
    def test_data_pca(self):
        return self.load_data(self.test_file_pca)

    @property
    def validation_data_pca(self):
        return self.load_data(self.validation_file_pca)

    @property
    def training_dataframe_pca(self):
        return self.load_data(self.train_file_pca, separate_target=False)

    @property
    def test_dataframe_pca(self):
        return self.load_data(self.test_file_pca, separate_target=False)

    @property
    def validation_dataframe_pca(self):
        return self.load_data(self.validation_file_pca, separate_target=False)
