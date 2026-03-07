"""
ml-classification-check
────────────────────────
Automatic sanity checks for classification models.
Catches data leakage, class imbalance, feature issues
and more — before you waste time training broken models.

Install:
    pip install ml-classification-check

Usage:
    from ml_classification_check import check_binary_classification
    check_binary_classification(X_train, X_test, y_train, y_test)
"""

import pandas as pd
import numpy as np

from ml_classification_check.detector      import (
    detect_problem_type,
    validate_problem_type,
    get_class_info
)
from ml_classification_check.reporter      import (
    print_summary,
    print_report,
    print_type_warning
)

__version__ = "0.1.0"
__author__  = "Shriparna Prasad"


# ─────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────

def _prepare_inputs(X_train, X_test, y_train, y_test):
    """
    Converts all inputs to pandas DataFrames/Series
    so every checker works with consistent data types
    regardless of what format the user passed in.
    Accepts numpy arrays, lists, or pandas objects.
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test  = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(np.array(y_train).flatten()) \
              if not isinstance(y_train, pd.DataFrame) \
              else y_train.reset_index(drop=True)
    y_test  = pd.Series(np.array(y_test).flatten()) \
              if y_test is not None and \
              not isinstance(y_test, pd.DataFrame) \
              else y_test
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────
# PUBLIC API — Three functions users call
# ─────────────────────────────────────────────────────

def check_binary_classification(
    X_train, X_test, y_train, y_test=None
):
    """
    Run all sanity checks for a Binary Classification problem.

    Parameters
    ----------
    X_train : DataFrame or array — training features
    X_test  : DataFrame or array — test features
    y_train : Series or array   — training labels (0/1)
    y_test  : Series or array   — test labels (optional)

    Returns
    -------
    dict → critical, warnings, passed, results

    Example
    -------
    from ml_classification_check import check_binary_classification
    check_binary_classification(X_train, X_test, y_train, y_test)
    """

    # Prepare inputs
    X_train, X_test, y_train, y_test = _prepare_inputs(
        X_train, X_test, y_train, y_test
    )

    # Validate problem type
    validation = validate_problem_type(y_train, "binary")
    print_type_warning(validation)

    # Print dataset summary
    class_info = get_class_info(y_train)
    print_summary(
        X_train, X_test, y_train, y_test,
        "binary", class_info
    )

    # Run all checks — populated in Phase 2, 3, 6
    results = _run_binary_checks(X_train, X_test, y_train, y_test)

    # Print full report and return
    return print_report(results, "binary")


def check_multiclass_classification(
    X_train, X_test, y_train, y_test=None
):
    """
    Run all sanity checks for a Multiclass Classification problem.

    Parameters
    ----------
    X_train : DataFrame or array — training features
    X_test  : DataFrame or array — test features
    y_train : Series or array   — training labels (0,1,2,3...)
    y_test  : Series or array   — test labels (optional)

    Returns
    -------
    dict → critical, warnings, passed, results

    Example
    -------
    from ml_classification_check import check_multiclass_classification
    check_multiclass_classification(X_train, X_test, y_train, y_test)
    """

    # Prepare inputs
    X_train, X_test, y_train, y_test = _prepare_inputs(
        X_train, X_test, y_train, y_test
    )

    # Validate problem type
    validation = validate_problem_type(y_train, "multiclass")
    print_type_warning(validation)

    # Print dataset summary
    class_info = get_class_info(y_train)
    print_summary(
        X_train, X_test, y_train, y_test,
        "multiclass", class_info
    )

    # Run all checks — populated in Phase 2, 4, 6
    results = _run_multiclass_checks(X_train, X_test, y_train, y_test)

    # Print full report and return
    return print_report(results, "multiclass")


def check_multilabel_classification(
    X_train, X_test, y_train, y_test=None
):
    """
    Run all sanity checks for a Multilabel Classification problem.

    Parameters
    ----------
    X_train : DataFrame or array     — training features
    X_test  : DataFrame or array     — test features
    y_train : DataFrame or 2D array  — training labels matrix
    y_test  : DataFrame or 2D array  — test labels (optional)

    Returns
    -------
    dict → critical, warnings, passed, results

    Example
    -------
    from ml_classification_check import check_multilabel_classification
    check_multilabel_classification(X_train, X_test, y_train, y_test)
    """

    # For multilabel y stays as DataFrame — don't flatten
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test  = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_test  = pd.DataFrame(y_test).reset_index(drop=True) \
              if y_test is not None else None

    # Validate problem type
    validation = validate_problem_type(y_train, "multilabel")
    print_type_warning(validation)

    # Print dataset summary
    class_info = get_class_info(y_train)
    print_summary(
        X_train, X_test, y_train, y_test,
        "multilabel", class_info
    )

    # Run all checks — populated in Phase 2, 5, 6
    results = _run_multilabel_checks(X_train, X_test, y_train, y_test)

    # Print full report and return
    return print_report(results, "multilabel")


# ─────────────────────────────────────────────────────
# INTERNAL CHECK RUNNERS
# These will be filled in as we build Phase 2, 3, 4, 5, 6
# For now they return empty lists — structure is ready
# ─────────────────────────────────────────────────────

def _run_binary_checks(X_train, X_test, y_train, y_test):
    """Runs all checks for binary classification."""
    results = []
    # Phase 2 checks added here
    # Phase 3 checks added here
    # Phase 6 checks added here
    return results


def _run_multiclass_checks(X_train, X_test, y_train, y_test):
    """Runs all checks for multiclass classification."""
    results = []
    # Phase 2 checks added here
    # Phase 4 checks added here
    # Phase 6 checks added here
    return results


def _run_multilabel_checks(X_train, X_test, y_train, y_test):
    """Runs all checks for multilabel classification."""
    results = []
    # Phase 2 checks added here
    # Phase 5 checks added here
    # Phase 6 checks added here
    return results