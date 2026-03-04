import os
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

@dataclass
class Source:
    # General Parsing Info
    url: str                
    sep: str                ## CSV Sep
    header: int = None      ## Header row number.
    nan_type: str = None    ## Which chars to replace with NaN

    # Specific split information
    id_col: int = None      ## ID column to remove it
    target_col: int = -1    ## On which column is the target
    one_hot: bool = False   ## Semeion has its target as one-hot

UCI_DATASETS = {
    "abalone":              Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
                                   sep=",",    nan_type=None),

    "anneal":               Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data",
                                   sep=",",    nan_type="?"),

    "arrhythmia":           Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
                                   sep=",",    nan_type="?"),

    "bridges-version1":     Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version1",
                                   sep=",",    nan_type="?",   id_col=0),

    "car":                  Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                                   sep=",",    nan_type=None),

    "cylinder-bands":       Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data",
                                   sep=",",    nan_type="?"),

    #"cylinder-bands":       Source(url=str(PROJECT_ROOT / "data" / "UCI" / "cylinder_bands.csv"),
    #                               sep=",",    nan_type="?"),

    "glass":                Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
                                   sep=",",    nan_type=None,   id_col=0),

    "hepatitis":            Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
                                   sep=",",    nan_type="?",   target_col=0),

    "iris":                 Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                                   sep=",",    nan_type=None),

    "semeion":              Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
                                   sep=r"\s+", nan_type=None,  one_hot=True),

    "shuttle_landing":      Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/shuttle-landing-control/shuttle-landing-control.data",
                                   sep=",",    nan_type="*",   target_col=0),

    "winequality-red":      Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                                   sep=";",    header=0, nan_type=None),

    "winequality-white":    Source(url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                                   sep=";",    header=0, nan_type=None),
}


def load_uci_dataset(source: Source):

    # Read CSV
    df = pd.read_csv(
        source.url,
        header=source.header,
        sep=source.sep,
        na_values=source.nan_type,
    )

    id_col = source.id_col
    target_col = source.target_col
    one_hot_encoding = source.one_hot

    # Special case: Semeion
    if one_hot_encoding:

        # Get Features
        X = df.iloc[:,:-10]

        # Last 10 cols are encoded as one-hot
        y_onehot = df.iloc[:,-10:]

        # Transform from one-hot into ordinal
        Y = np.argmax(y_onehot.to_numpy(), axis=1)

        # Encode target labels
        Y = LabelEncoder().fit_transform(Y.astype(str))
        return X, Y

    # --- Others ---
    # Remove ID
    if not(id_col is None):
        df = df.drop(df.columns[id_col], axis=1)

    # Split target & features
    target_col_name = df.columns[target_col]
    X = df.drop(target_col_name, axis=1)
    Y = df[target_col_name]

    # Encode target labels
    Y = LabelEncoder().fit_transform(Y.astype(str))

    return X, Y


def first_process():

    # Download all UCI datasets
    for name, source in UCI_DATASETS.items():

        print("Downloading:", name)
        X, y = load_uci_dataset(source)

        # Get dimensions
        n, m = X.shape

        # Make column names
        cols = [f"feature_{i:03d}" for i in range(m)] + ["target"]

        # Make pandas df
        data = np.hstack((X, y.reshape(-1,1)))
        df = pd.DataFrame(data, columns=cols)

        # Save into data path as csv
        new_name = name.replace("-", "_") + ".csv"
        data_path = Path.cwd() / "data" / new_name
        print("Saving to:", data_path)
        df.to_csv(data_path, index=False)


def main():

    # First, process
    first_process()

    bc = load_breast_cancer(return_X_y=True)
    X, y = bc[0], bc[1]

    # Get dimensions
    n, m = X.shape

    # Make column names
    cols = [f"feature_{i:03d}" for i in range(m)] + ["target"]

    # Make pandas df
    data = np.hstack((X, y.reshape(-1,1)))
    df = pd.DataFrame(data, columns=cols)

    # Save into data path as csv
    new_name = "breast_cancer.csv"
    data_path = Path.cwd() / "data" / new_name
    print("Saving to:", data_path)
    df.to_csv(data_path, index=False)


if __name__ == "__main__":
    main()
