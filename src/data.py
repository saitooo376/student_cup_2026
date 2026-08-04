import pandas as pd

from src.paths import DATA_DIR
from src.features import create_features

def load_train():
    train = pd.read_csv(DATA_DIR/"train.csv")
    train = create_features(train)
    return train

def load_test():
    test = pd.read_csv(DATA_DIR/"test.csv")
    test = create_features(test)
    return test


# preprocess.pyに切り分けることも検討
def convert_category(train, test, cat_cols):

    for col in cat_cols:

        categories = train[col].astype("category").cat.categories

        train[col] = pd.Categorical(
            train[col],
            categories=categories
        )

        test[col] = pd.Categorical(
            test[col],
            categories=categories
        )

    return train, test
