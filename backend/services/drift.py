import numpy as np
from scipy.stats import ks_2samp


class DriftDetector:
    """
    Detects drift between reference and current distributions.
    """

    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def ks_drift(self, reference: list[float], current: list[float]) -> dict:
        """
        Kolmogorov-Smirnov test for distribution drift.
        """
        stat, p_value = ks_2samp(reference, current)

        return {
            "drift_detected": p_value < self.p_value_threshold,
            "p_value": float(p_value),
            "statistic": float(stat),
        }
