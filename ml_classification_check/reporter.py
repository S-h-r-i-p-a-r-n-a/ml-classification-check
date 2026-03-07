"""
Reporter Module
───────────────
Handles all terminal output for ml-classification-check.
Prints the complete 4-section report:
  Section 1 → Dataset Summary
  Section 2 → Check Results
  Section 3 → Final Score
  Section 4 → Fix Code
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────
# SECTION 1 — Dataset Summary Printer
# ─────────────────────────────────────────────────────

def print_summary(X_train, X_test, y_train, y_test, problem_type, class_info):
    """
    Prints a clean dataset summary before any checks run.
    Gives the user a complete picture of what they passed in.

    Parameters
    ----------
    X_train      : training features
    X_test       : test features
    y_train      : training labels
    y_test       : test labels (optional, can be None)
    problem_type : "binary", "multiclass", or "multilabel"
    class_info   : dict from detector.get_class_info()
    """

    # Problem type display name
    type_display = {
        "binary"      : "Binary Classification",
        "multiclass"  : "Multiclass Classification",
        "multilabel"  : "Multilabel Classification"
    }.get(problem_type, problem_type)

    # Header
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  ML CLASSIFICATION CHECK — {type_display:<26}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Dataset info
    print("📋 DATASET SUMMARY")
    print(f"   Problem Type   : {type_display}")
    print(f"   Train samples  : {len(X_train)}"
          f"  |  Test samples : {len(X_test)}")
    print(f"   Features       : {X_train.shape[1]} columns")

    # Class info
    if problem_type == "multilabel":
        print(f"   Labels         : {class_info['n_classes']} labels")
        print(f"   Label names    : {class_info['classes']}")
    else:
        print(f"   Classes        : {class_info['classes']}")
        if class_info.get("distribution"):
            dist = "  |  ".join(
                [f"Class {k} = {v}"
                 for k, v in class_info["distribution"].items()]
            )
            print(f"   Distribution   : {dist}")

    # Feature info
    print(f"   Feature names  : {list(X_train.columns)[:5]}"
          f"{'...' if X_train.shape[1] > 5 else ''}")

    # Missing values quick glance
    total_missing = X_train.isnull().sum().sum()
    if total_missing > 0:
        print(f"   Missing values : {total_missing} found in training set")
    else:
        print(f"   Missing values : None found in training set")

    print()


# ─────────────────────────────────────────────────────
# SECTION 2 + 3 + 4 — Full Report Printer
# ─────────────────────────────────────────────────────

def print_report(results, problem_type):
    """
    Prints sections 2, 3, and 4 of the report.
    Called after all checks have been run.

    Parameters
    ----------
    results      : list of result dicts from all checkers
    problem_type : "binary", "multiclass", or "multilabel"
    """

    # ── Section 2 — Check Results ─────────────────────
    print("🔍 RUNNING CHECKS...")
    print()

    # Data Integrity Group
    data_results = [r for r in results if r.get("group") == "data_integrity"]
    if data_results:
        print("── DATA INTEGRITY ───────────────────────────────────")
        for r in data_results:
            print(r["message"])
            print()

    # Classification Health Group
    clf_results = [r for r in results if r.get("group") == "classification"]
    if clf_results:
        print("── CLASSIFICATION HEALTH ────────────────────────────")
        for r in clf_results:
            print(r["message"])
            print()

    # Feature Quality Group
    feat_results = [r for r in results if r.get("group") == "features"]
    if feat_results:
        print("── FEATURE QUALITY ──────────────────────────────────")
        for r in feat_results:
            print(r["message"])
            print()

    # ── Section 3 — Final Score ───────────────────────
    critical = sum(1 for r in results if r["severity"] == "critical")
    warnings = sum(1 for r in results if r["severity"] == "warning")
    passed   = sum(1 for r in results if r["severity"] == "ok")

    print("─" * 54)
    print()
    print("📊 FINAL SCORE")
    print(f"   ❌ Critical  : {critical}   → Fix before training")
    print(f"   ⚠️  Warnings  : {warnings}   → Fix for better results")
    print(f"   ✅ Passed    : {passed}   → Looking good")
    print()

    if critical > 0:
        print("🚨 DO NOT TRAIN YET — Fix critical issues first!")
    elif warnings > 0:
        print("⚠️  Warnings found — fix for best results.")
    else:
        print("✅ All checks passed — safe to train your model!")

    print("─" * 54)

    # ── Section 4 — Fix Code ──────────────────────────
    fixes = [r for r in results if r.get("fix_code")]

    if fixes:
        print()
        print("💡 SUGGESTED FIXES — Copy & Paste into your notebook")
        print("─" * 54)

        for i, r in enumerate(fixes, 1):
            print()
            print(f"[FIX {i}] {r['check']}")
            print("─" * 40)
            print(r["fix_code"])

        print()
        print("─" * 54)
        print("✅ After applying fixes re-run the check to verify!")

    print()

    return {
        "critical" : critical,
        "warnings" : warnings,
        "passed"   : passed,
        "results"  : results
    }


# ─────────────────────────────────────────────────────
# VALIDATION WARNING PRINTER
# ─────────────────────────────────────────────────────

def print_type_warning(validation_result):
    """
    Prints a warning if user called the wrong
    check function for their problem type.
    """
    if not validation_result["valid"]:
        print()
        print("─" * 54)
        print(validation_result["message"])
        print("─" * 54)
        print()
