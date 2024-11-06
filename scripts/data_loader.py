import pandas as pd


class DataLoader:
    train_file = "../data/preprocessed/dataset_train.csv"
    test_file = "../data/preprocessed/dataset_test.csv"
    validation_file = "../data/preprocessed/dataset_val.csv"

    def __init__(self):
        self.train_file = DataLoader.train_file
        self.test_file = DataLoader.test_file
        self.validation_file = DataLoader.validation_file

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
