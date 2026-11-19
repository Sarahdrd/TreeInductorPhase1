"""Construct unfitted baseline pipelines; no evaluation or fitting on import."""
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from .infrastructure import config
from .preprocessing import build_preprocessor


def make_model(name, X, y=None):
    params = config()['models'].get(name)
    if params is None:
        raise ValueError(f'No frozen adapter for {name}')
    if name in ('ripper', 'svm_rbf'):
        from .recovered_cross_models import make_cross_model
        if name == 'ripper' and y is None:
            raise ValueError('RIPPER requires encoded y to select the original binary/OvR strategy')
        model = make_cross_model(name, X, y)
        estimator = model.named_steps['model']
        if name == 'ripper' and hasattr(estimator, 'estimator'):
            estimator = estimator.estimator
        observed = estimator.get_params(deep=False)
        # sklearn 1.9's default sentinel resolves to False in BaseLibSVM.fit.
        # Keep the recovered factory unchanged; compare effective semantics.
        if name == 'svm_rbf' and observed.get('probability') == 'deprecated' and params.get('probability') is False:
            observed = {**observed, 'probability': False}
        if observed != params:
            raise ValueError(f'Recovered {name} defaults differ from the frozen historical configuration')
        return model
    if name.startswith('dummy_'):
        return DummyClassifier(**params)
    if name.startswith('decision_tree_'):
        estimator = DecisionTreeClassifier(**params)
    elif name.startswith('logistic_regression'):
        estimator = LogisticRegression(**params)
    else:
        raise ValueError(f'Unsupported model: {name}')
    scaled = name.startswith('logistic_regression') or name == 'svm_rbf'
    return Pipeline([('preprocess', build_preprocessor(X, for_linear=scaled)), ('model', estimator)])


# Future adapters must provide sklearn-compatible fit/predict/get_params and
# return predictions in the original encoded label space. Their preprocessing
# belongs inside fit on the supplied training indices. No new inducer integrated.
