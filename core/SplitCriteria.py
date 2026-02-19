# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:17:23 2025

"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.special import gammaln
from sklearn.preprocessing import LabelEncoder
from math import exp, log2
from scipy.stats import hypergeom


# ---------- UTILITIES ----------
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def gini_split(y, X, t):
    left, right = X <= t, X > t
    if not left.any() or not right.any():
        return 0.0
    n = len(y)
    nL = left.sum()
    nR = right.sum()
    return gini(y) - (nL/n)*gini(y[left]) - (nR/n)*gini(y[right])



def safe_log2(x):
    return np.log2(x, out=np.zeros_like(x), where=(x > 0))


# ---------- SPLIT CRITERIA ----------
def information_gain(y, left_idx, right_idx):
    p = len(left_idx) / (len(left_idx) + len(right_idx))
    return entropy(y) - (p * entropy(y[left_idx]) + (1 - p) * entropy(y[right_idx]))


def g_stat(y, X, t):
    y = np.asarray(y).ravel().astype(int)
    classes = np.unique(y)
    C, V = len(classes), 2
    N = len(X)

    left, right = X <= t, X > t
    if not left.any() or not right.any():
        return 0.0

    n_c = np.bincount(y)
    m_v = np.array([left.sum(), right.sum()])
    f_cv = np.array([[np.sum(y[left] == c), np.sum(y[right] == c)] for c in classes])

    log_num = np.sum(gammaln(n_c + 1)) + np.sum(gammaln(m_v + 1))
    log_den = gammaln(N + 1) + np.sum(gammaln(f_cv + 1))
    log_ratio = np.clip(log_num - log_den, -700, 700)
    return 1 - np.exp(log_ratio)


def mantaras(y, X, t):
    y = LabelEncoder().fit_transform(np.ravel(y))
    X = np.asarray(X).ravel()
    left, right = X <= t, X > t

    if not left.any() or not right.any():
        return 0.0

    left_counts = np.bincount(y[left], minlength=np.max(y) + 1)
    right_counts = np.bincount(y[right], minlength=np.max(y) + 1)
    observed = np.array([left_counts, right_counts])

    total = observed.sum()
    I_PC = -np.sum((observed.sum(axis=0) / total) * safe_log2(observed.sum(axis=0) / total))
    I_PV = -np.sum((observed.sum(axis=1) / total) * safe_log2(observed.sum(axis=1) / total))
    joint = observed / total
    I_PC_PV = -np.sum(joint * safe_log2(joint))
    distance = (I_PC_PV - I_PV) + (I_PC_PV - I_PC)
    return 1 - (distance / I_PC_PV)

def hg_distribution(y, X_col, t, eps=1e-12):
    # Encode classes as integers
    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    N = len(y)
    n_left = left_mask.sum()

    score = 0.0
    classes = np.unique(y)

    for c in classes:
        Kc = np.sum(y == c)                 # total class count
        kc_left = np.sum(y[left_mask] == c) # class count in left child

        # Hypergeometric probability
        p = hypergeom.pmf(kc_left, N, Kc, n_left)

        # Penalize unlikely events (good splits)
        score -= np.log(p + eps)
    return score


def chv_criterion(y, X_col, t, normalize=True):
    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    n_classes = len(np.unique(y))

    observed = np.array([
        np.bincount(y[left_mask], minlength=n_classes),
        np.bincount(y[right_mask], minlength=n_classes)
    ])

    # Guard against empty expected frequencies
    if np.any(observed.sum(axis=0) == 0):
        return 0.0

    chi2, _, _, _ = chi2_contingency(observed, correction=False)

    if normalize:
        return chi2 / len(y)
    else:
        return chi2
    

def dcsm(y, X_col, t):
    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    n = len(y)
    nL = left_mask.sum()
    nR = right_mask.sum()

    n_classes = len(np.unique(y))

    def dcs(y_subset):
        counts = np.bincount(y_subset, minlength=n_classes)
        probs = counts / counts.sum()
        return np.sum(probs ** 2)

    dcs_parent = dcs(y)
    dcs_left = dcs(y[left_mask])
    dcs_right = dcs(y[right_mask])

    return dcs_parent - (nL / n) * dcs_left - (nR / n) * dcs_right



def chi_square(y, X, t):
    y = LabelEncoder().fit_transform(np.ravel(y))
    left, right = X <= t, X > t
    if not left.any() or not right.any():
        return 0.0

    observed = np.array([
        np.bincount(y[left], minlength=len(np.unique(y))),
        np.bincount(y[right], minlength=len(np.unique(y)))
    ])
    chi2, _, _, _ = chi2_contingency(observed, correction=True)
    return chi2


def mpi(y, X_col, t):
    # mpi > 0 means improvement in purity
    # mpi < 0 means worsening in purity and classification
    # mpi = 0 means no change
    
    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    n = len(y)
    nL = left_mask.sum()
    nR = right_mask.sum()

    n_classes = len(np.unique(y))

    def purity(y_subset):
        counts = np.bincount(y_subset, minlength=n_classes)
        return np.max(counts) / counts.sum()

    purity_parent = purity(y)
    purity_left = purity(y[left_mask])
    purity_right = purity(y[right_mask])

    # MPI = weighted child purity - parent purity
    return (nL / n) * purity_left + (nR / n) * purity_right - purity_parent


def ort_criterion(y, X_col, t, eps=1e-12):

    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    n_classes = len(np.unique(y))

    # Class probability vectors
    p_left = np.bincount(y[left_mask], minlength=n_classes)
    p_right = np.bincount(y[right_mask], minlength=n_classes)

    p_left = p_left / p_left.sum()
    p_right = p_right / p_right.sum()

    # Cosine similarity
    dot = np.dot(p_left, p_right)
    norm_left = np.linalg.norm(p_left)
    norm_right = np.linalg.norm(p_right)

    cos_theta = dot / (norm_left * norm_right + eps)

    # ORT score
    ort = 1.0 - cos_theta

    return ort



def twoing(y, X, t):
    left, right = X <= t, X > t
    if not left.any() or not right.any():
        return 0.0
    n_total, nL, nR = len(y), left.sum(), right.sum()
    PL, PR = nL / n_total, nR / n_total

    diff_sum = sum(abs((np.sum(y[left] == c) / nL) - (np.sum(y[right] == c) / nR))
                   for c in np.unique(y))
    return PL * PR * (diff_sum ** 2)


def cair(y, X_col, t):

    y = LabelEncoder().fit_transform(np.ravel(y))

    left_mask = X_col <= t
    right_mask = X_col > t

    if not left_mask.any() or not right_mask.any():
        return 0.0

    n_total = len(y)
    nL = left_mask.sum()
    nR = right_mask.sum()

    n_classes = len(np.unique(y))

    cair_score = 0.0
    for c in range(n_classes):
        # Class prior
        p_c = np.sum(y == c) / n_total

        # Conditional probabilities
        p_cL = np.sum(y[left_mask] == c) / nL
        p_cR = np.sum(y[right_mask] == c) / nR

        cair_score += p_c * abs(p_cL - p_cR)

    return cair_score



def gain_ratio(y, X, t):
    ent_init = entropy(y)
    left, right = X <= t, X > t
    if not left.any() or not right.any():
        return 0.0

    nL, nR = left.sum(), right.sum()
    total = nL + nR
    info_gain = ent_init - (nL / total) * entropy(y[left]) - (nR / total) * entropy(y[right])
    split_info = - (nL / total) * log2(nL / total) - (nR / total) * log2(nR / total)
    return info_gain / split_info if split_info else 0.0




# ---------- MAIN CLASS ----------
class SplitCriterion:
    def __init__(self, criterion="gini"):
        self.criterion = criterion

    def calculate(self, y, X_column, threshold):
        methods = {
            "information_gain": lambda: information_gain(y, np.where(X_column <= threshold)[0], np.where(X_column > threshold)[0]),
            "gini": lambda: gini_split(y, X_column, threshold),
            "g_stat": lambda: g_stat(y, X_column, threshold),
            "mantaras": lambda: mantaras(y, X_column, threshold),
            "hg_distribution": lambda: hg_distribution(y, X_column, threshold),
            "chv_criterion": lambda: chv_criterion(y, X_column, threshold),
            "dcsm": lambda: dcsm(y, X_column, threshold),
            "chi_square": lambda: chi_square(y, X_column, threshold),
            "mpi": lambda: mpi(y, X_column, threshold),
            "ort": lambda: ort_criterion(y, X_column, threshold),
            "twoing": lambda: twoing(y, X_column, threshold),
            "cair": lambda: cair(y, X_column, threshold),
            "gain_ratio": lambda: gain_ratio(y, X_column, threshold)
        }
        if self.criterion not in methods:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        return methods[self.criterion]()
