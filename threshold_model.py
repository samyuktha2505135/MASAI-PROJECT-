"""Serializable estimator wrapper with a training-only chosen decision threshold."""
import numpy as np
class ThresholdedModel:
    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = float(threshold)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int64)
