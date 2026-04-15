import math
import numpy as np
from sklearn.model_selection import train_test_split


def can_stratify(y, test_size=0.2):
    y = np.asarray(y)

    if len(y) < 2:
        return False

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return False

    # Need at least 2 members per class for stratified splitting
    if counts.min() < 2:
        return False

    # Validation set must be large enough to hold at least one sample per class
    n_test = max(1, math.ceil(len(y) * test_size))
    if n_test < len(classes):
        return False

    return True


def safe_train_test_split(X, y, test_size=0.2, random_state=42, prefer_stratify=True):
    y_arr = np.asarray(y)

    use_stratify = prefer_stratify and can_stratify(y_arr, test_size=test_size)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None
    )