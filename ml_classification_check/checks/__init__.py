class BaseChecker:
    """
    Parent class for all checkers.
    Every checker must implement the check() method
    and return a result dictionary with these keys:
    - passed   : bool
    - severity : "critical", "warning", or "ok"
    - check    : name of the check
    - group    : "data_integrity", "classification", "features"
    - message  : what was found
    - fix_code : paste-ready code to fix the issue (or None)
    """

    def check(self):
        raise NotImplementedError(
            "Every checker must implement the check() method."
        )

    def _result(self, passed, severity, check, group, message, fix_code=None):
        """
        Standard result format returned by every checker.
        """
        return {
            "passed"   : passed,
            "severity" : severity,
            "check"    : check,
            "group"    : group,
            "message"  : message,
            "fix_code" : fix_code
        }
