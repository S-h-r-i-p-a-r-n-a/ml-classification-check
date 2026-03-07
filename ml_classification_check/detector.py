"""
Problem Type Auto-Detector
──────────────────────────
Automatically figures out whether the user's problem is:
- Binary Classification     (exactly 2 unique classes)
- Multiclass Classification (3 or more unique classes)
- Multilabel Classification (y is a 2D matrix)

This means the user never has to manually specify their
problem type — the package figures it out automatically.
"""

import numpy as np
import pandas as pd


def detect_problem_type(y_train):
    """
    Detects classification problem type from y_train.

    Parameters
    ----------
    y_train : array-like, Series, or DataFrame
        Training labels

    Returns
    -------
    str : "binary", "multiclass", or "multilabel"
    """

    # Convert to numpy for consistent handling
    y = np.array(y_train)

    # ── Check for Multilabel ──────────────────────────
    # Multilabel means y is a 2D matrix
    # Example: [[1,0,1], [0,1,1], [1,1,0]]
    # Each row = one sample, each column = one label
    if isinstance(y_train, pd.DataFrame):
        return "multilabel"

    if y.ndim == 2 and y.shape[1] > 1:
        return "multilabel"

    # ── Check for Binary ─────────────────────────────
    # Binary means exactly 2 unique values in target
    # Example: [0,1,0,1,1,0] or ["yes","no","yes"]
    unique_classes = np.unique(y)

    if len(unique_classes) == 2:
        return "binary"

    # ── Default to Multiclass ────────────────────────
    # 3 or more unique values = multiclass
    # Example: [0,1,2,3] or ["cat","dog","bird"]
    return "multiclass"


def validate_problem_type(y_train, declared_type):
    """
    Checks if user's declared problem type matches
    what the data actually looks like.

    For example if user calls check_binary_classification()
    but their y_train has 5 classes — this catches it.

    Parameters
    ----------
    y_train      : array-like training labels
    declared_type: "binary", "multiclass", or "multilabel"

    Returns
    -------
    dict with keys: valid (bool), detected (str), message (str)
    """

    detected = detect_problem_type(y_train)

    if detected == declared_type:
        return {
            "valid"    : True,
            "detected" : detected,
            "message"  : f"✅ Problem type confirmed: {declared_type}"
        }

    return {
        "valid"    : False,
        "detected" : detected,
        "message"  : (
            f"⚠️  WARNING: You called check_{declared_type}_classification()\n"
            f"   but your y_train looks like a {detected} problem.\n"
            f"   Detected: {len(np.unique(np.array(y_train)))} unique classes.\n"
            f"   Consider using check_{detected}_classification() instead."
        )
    }


def get_class_info(y_train):
    """
    Extracts useful class information from y_train.
    Used by the dataset summary printer.

    Returns
    -------
    dict with class names, counts, and distribution
    """

    y = np.array(y_train)

    # Handle multilabel separately
    if isinstance(y_train, pd.DataFrame):
        return {
            "classes"      : list(y_train.columns),
            "n_classes"    : y_train.shape[1],
            "distribution" : None,
            "type"         : "multilabel"
        }

    unique, counts = np.unique(y, return_counts=True)
    total          = len(y)
    distribution   = {
        str(cls): f"{round((cnt/total)*100, 1)}%"
        for cls, cnt in zip(unique, counts)
    }

    return {
        "classes"      : list(unique),
        "n_classes"    : len(unique),
        "distribution" : distribution,
        "type"         : detect_problem_type(y_train)
    }
