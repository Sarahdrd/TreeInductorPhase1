"""Authoritative recovered definitions; never execute full notebook/evaluations.

Source: Methodology_Only_Tree.ipynb, Cross-Model Validation, cell 14 (zero-based).
Function/class bodies are unchanged. Only imports and module initialization added.
"""
import importlib.metadata
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone, is_classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from .preprocessing import build_preprocessor

CROSS_VERSIONS = {}

SVM_CONFIG = dict(kernel="rbf", C=1.0, gamma="scale", class_weight=None)

try:
    import wittgenstein
    CROSS_VERSIONS["wittgenstein"] = importlib.metadata.version("wittgenstein")
    RIPPER_CONFIG = wittgenstein.RIPPER(random_state=42).get_params()
    RIPPER_IMPORT_ERROR = ""
except ImportError as exc:
    RIPPER_CONFIG = None
    CROSS_VERSIONS["wittgenstein"] = "unavailable"
    RIPPER_IMPORT_ERROR = repr(exc)

class RipperPreprocessor(TransformerMixin, BaseEstimator):
    """Fold-local imputation preserving numeric and nominal feature types."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.numeric_ = set(X.select_dtypes(include=[np.number]).columns)
        self.fill_ = {}
        for c in self.columns_:
            values = X[c].dropna()
            self.fill_[c] = ((values.median() if len(values) else 0.0) if c in self.numeric_
                else str(values.astype(str).mode().iloc[0]) if len(values) else "__MISSING__")
        return self
    def transform(self, X):
        out = X.loc[:, self.columns_].copy()
        for c in self.columns_:
            out[c] = (pd.to_numeric(out[c]).fillna(self.fill_[c]).astype(float) if c in self.numeric_
                else out[c].astype(object).where(out[c].notna(), self.fill_[c]).astype(str))
        return out

if not RIPPER_IMPORT_ERROR:
    class SklearnRIPPER(ClassifierMixin, wittgenstein.RIPPER, BaseEstimator):
        """Only sklearn metadata: inherit RIPPER's constructor, fit and predictions."""
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.classifier_tags.multi_class = False  # native RIPPER is binary; OvR handles multiclass
            return tags

def make_svm_model(X):
    return Pipeline([("preprocess", build_preprocessor(X, for_linear=True)),
                     ("model", SVC(**SVM_CONFIG))])

def make_cross_model(model, X, y):
    if model == "svm_rbf":
        return make_svm_model(X)
    if model != "ripper":
        raise ValueError(f"Unknown model: {model}")
    if RIPPER_IMPORT_ERROR:
        raise ImportError(RIPPER_IMPORT_ERROR)
    estimator = SklearnRIPPER(random_state=42)
    if len(np.unique(y)) > 2:
        estimator = OneVsRestClassifier(estimator, n_jobs=1)
    return Pipeline([("preprocess", RipperPreprocessor()), ("model", estimator)])

def check_cross_predictions(y, predicted):
    predicted = np.asarray(predicted)
    if predicted.ndim != 1 or len(predicted) != len(y) or not np.isin(predicted, np.unique(y)).all():
        raise ValueError("Invalid prediction shape or unknown class")
    score = float(f1_score(y, predicted, average="macro", zero_division=0))
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid macro-F1")
    return score

def check_ripper_compatibility():
    caught = []
    try:
        if RIPPER_IMPORT_ERROR:
            raise ImportError(RIPPER_IMPORT_ERROR)
        raw = SklearnRIPPER(random_state=42)
        if not is_classifier(raw) or not is_classifier(clone(raw)):
            raise TypeError("SklearnRIPPER/clone missing classifier tags")
        if raw.get_params() != wittgenstein.RIPPER(random_state=42).get_params():
            raise ValueError("RIPPER hyperparameters changed")
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True, as_frame=True)
        X["nominal"] = np.where(np.arange(len(X)) % 2, "a", "b")
        X.loc[0, X.columns[0]] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for target in [(y > 0).astype(int).to_numpy(), y.to_numpy()]:
                pipe = clone(make_cross_model("ripper", X, target))
                if not is_classifier(pipe):
                    raise TypeError("Installed sklearn does not recognize RIPPER as a classifier")
                train, test = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                  random_state=42).split(X, target))
                pipe.fit(X.iloc[train], target[train])
                check_cross_predictions(target[test], pipe.predict(X.iloc[test]))
                probabilities = np.asarray(pipe.predict_proba(X.iloc[test]))
                expected = (len(test), len(np.unique(target)))
                if (probabilities.shape != expected or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1.0)):
                    raise ValueError(f"Invalid predict_proba output; expected {expected}")
                fitted = pipe.named_steps["model"]
                members = fitted.estimators_ if isinstance(fitted, OneVsRestClassifier) else [fitted]
                if not all(is_classifier(m) and is_classifier(clone(m)) for m in members):
                    raise TypeError("Fitted RIPPER/OvR member missing classifier tags")
        return dict(status="complete", error="", warnings=[str(w.message) for w in caught])
    except Exception:
        return dict(status="compatibility_error", error=traceback.format_exc(),
                    warnings=[str(w.message) for w in caught])
