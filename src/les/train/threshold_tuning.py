"""
Threshold Tuning Module for Multi-Class Classification
========================================================
Optimizes decision thresholds for each class to boost recall and precision.

This module handles threshold optimization for 3-class churn prediction:
- Class 0: Early Churn (0-3 months)
- Class 1: Medium Churn (3-6 months)
- Class 2: No Churn (retention)

Strategies:
1. Lower thresholds for churn classes (0,1) to BOOST RECALL (catch more churners)
2. Adjust precision vs recall tradeoff for each class
3. Maintain No-Churn class accuracy
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdTuner:
    """
    Optimizes decision thresholds for multi-class classification.

    For binary and multi-class problems with probability outputs,
    this class finds optimal thresholds to maximize a given metric
    while maintaining balance between precision and recall.
    """

    def __init__(self, n_classes: int = 3):
        """
        Initialize ThresholdTuner.

        Args:
            n_classes: Number of classes in the problem (default: 3 for churn)
        """
        self.n_classes = n_classes
        self.thresholds = {}
        self.metrics_history = []

    def apply_optimal_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        target_classes: List[int] = None,
        metric: str = "balanced_precision",
        recall_threshold: float = 0.70,
        precision_threshold: float = 0.60,
        **kwargs
    ) -> Dict[str, float]:
        """
        Find optimal thresholds for each class to maximize specified metric.

        Args:
            y_prob: Probability predictions (n_samples, n_classes)
            y_true: True labels (n_samples,)
            target_classes: List of class indices to optimize (default: [0, 1])
            metric: Metric to optimize - "f1", "balanced_precision", "recall", "precision"
            recall_threshold: Minimum recall to maintain (default: 0.70)
            precision_threshold: Minimum precision to maintain (default: 0.60)

        Returns:
            Dictionary mapping class index to optimal threshold value
        """
        if target_classes is None:
            target_classes = [0, 1]  # Default: optimize for churn classes

        logger.info(f"🎯 Finding optimal thresholds for classes: {target_classes}")
        logger.info(f"   Metric: {metric}")
        logger.info(f"   Min Recall: {recall_threshold:.2f}, Min Precision: {precision_threshold:.2f}")

        # Initialize thresholds: default is max probability per sample
        thresholds = {}

        # For each target class, find optimal threshold
        for class_idx in target_classes:
            threshold = self._find_optimal_threshold_for_class(
                y_prob=y_prob,
                y_true=y_true,
                class_idx=class_idx,
                metric=metric,
                recall_threshold=recall_threshold,
                precision_threshold=precision_threshold
            )
            thresholds[class_idx] = threshold

            logger.info(f"   Class {class_idx}: threshold = {threshold:.4f}")

        # Set default threshold for non-target classes
        for class_idx in range(self.n_classes):
            if class_idx not in thresholds:
                thresholds[class_idx] = 0.0  # Default: any positive probability

        self.thresholds = thresholds
        return thresholds

    def _find_optimal_threshold_for_class(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        class_idx: int,
        metric: str = "balanced_precision",
        recall_threshold: float = 0.70,
        precision_threshold: float = 0.60
    ) -> float:
        """
        Find optimal threshold for a single class using grid search.

        Args:
            y_prob: Probability predictions (n_samples, n_classes)
            y_true: True labels
            class_idx: Target class index
            metric: Metric to optimize
            recall_threshold: Minimum recall to maintain
            precision_threshold: Minimum precision to maintain

        Returns:
            Optimal threshold value for this class
        """
        # Get probabilities for this class
        class_probs = y_prob[:, class_idx]
        is_class = (y_true == class_idx).astype(int)

        best_threshold = 0.5
        best_score = -1
        candidate_thresholds = np.linspace(0.0, 1.0, 101)  # 101 candidate thresholds

        for threshold in candidate_thresholds:
            # Predict: assign to class if prob >= threshold
            y_pred_class = (class_probs >= threshold).astype(int)

            # Skip if no positive predictions
            if y_pred_class.sum() == 0:
                continue

            # Calculate metrics
            try:
                precision = precision_score(is_class, y_pred_class, zero_division=0)
                recall = recall_score(is_class, y_pred_class, zero_division=0)
                f1 = f1_score(is_class, y_pred_class, zero_division=0)
            except:
                continue

            # Check constraints
            if recall < recall_threshold or precision < precision_threshold:
                continue

            # Score based on metric
            if metric == "f1":
                score = f1
            elif metric == "balanced_precision":
                score = 0.6 * recall + 0.4 * precision  # Prefer recall for churn detection
            elif metric == "recall":
                score = recall
            elif metric == "precision":
                score = precision
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_threshold = threshold

        # If no valid threshold found, use conservative default
        if best_score == -1:
            best_threshold = 0.5
            logger.warning(f"   ⚠️  No valid threshold found for class {class_idx}, using 0.5")

        return best_threshold

    def apply_thresholds_to_predictions(
        self,
        y_prob: np.ndarray,
        threshold_config: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply thresholds to probability predictions to get final class predictions.

        For multi-class problems:
        1. For each class, check if its probability meets the threshold
        2. Assign to class with highest probability among those meeting their thresholds
        3. Fallback to argmax if no class meets its threshold

        Args:
            y_prob: Probability predictions (n_samples, n_classes)
            threshold_config: Dictionary mapping class index to threshold

        Returns:
            Predicted class labels (n_samples,)
        """
        n_samples = y_prob.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Get probabilities for this sample
            probs = y_prob[i, :]

            # Check which classes meet their thresholds
            valid_classes = []
            for class_idx in range(self.n_classes):
                threshold = threshold_config.get(class_idx, 0.0)
                if probs[class_idx] >= threshold:
                    valid_classes.append(class_idx)

            # Assign to class with highest probability among valid ones
            if valid_classes:
                # Among valid classes, pick the one with highest probability
                best_class = max(valid_classes, key=lambda c: probs[c])
                y_pred[i] = best_class
            else:
                # Fallback: pick class with highest probability
                y_pred[i] = np.argmax(probs)

        return y_pred

    def optimize_multi_class_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        metric: str = "balanced_f1",
        weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Optimize thresholds for all classes simultaneously.

        This method adjusts thresholds to maximize overall weighted metric
        while maintaining per-class performance bounds.

        Args:
            y_prob: Probability predictions (n_samples, n_classes)
            y_true: True labels
            metric: Metric to optimize ("balanced_f1", "macro_f1", "weighted_f1")
            weights: Optional per-class weights for the metric

        Returns:
            Optimized threshold configuration
        """
        if weights is None:
            weights = {0: 1.5, 1: 1.5, 2: 1.0}  # Boost churn classes

        logger.info(f"🔧 Optimizing multi-class thresholds with metric: {metric}")

        best_config = {i: 0.5 for i in range(self.n_classes)}
        best_score = -1

        # Grid search over threshold combinations (simplified)
        threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        # Only search over churn classes (class 0, 1) - too expensive otherwise
        for t0 in threshold_candidates:
            for t1 in threshold_candidates:
                config = {0: t0, 1: t1, 2: 0.5}
                y_pred = self.apply_thresholds_to_predictions(y_prob, config)

                # Calculate weighted metric
                try:
                    if metric == "balanced_f1":
                        scores = [
                            f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
                            for i in range(self.n_classes)
                        ]
                        weighted_score = sum(w * s for w, s in zip(weights.values(), scores))
                    else:
                        weighted_score = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_config = config
                except:
                    continue

        self.thresholds = best_config
        logger.info(f"   Best config: {best_config} (score: {best_score:.4f})")

        return best_config

    def get_threshold_summary(self) -> str:
        """Return a summary of the current thresholds."""
        summary = "🎯 THRESHOLD SUMMARY:\n"
        class_names = {
            0: "Early Churn (0-3mo)",
            1: "Medium Churn (3-6mo)",
            2: "No Churn"
        }

        for class_idx, threshold in self.thresholds.items():
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            summary += f"   {class_name}: {threshold:.4f}\n"

        return summary
