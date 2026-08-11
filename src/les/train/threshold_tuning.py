"""
Threshold Tuning for Multi-class Predictions
Adjusts decision thresholds to optimize recall/precision for specific classes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.les.logger import logger
from src.les.exception import CustomException
import sys


class ThresholdTuner:
    """Optimize prediction thresholds for multi-class classification"""

    @staticmethod
    def adjust_predictions(
        y_prob: np.ndarray,
        target_class: int,
        boost_factor: float = 1.2,
    ) -> np.ndarray:
        """
        Boost predictions for a specific class

        Args:
            y_prob: Probability matrix from predict_proba (n_samples, n_classes)
            target_class: Class to boost (0 or 1 for churn)
            boost_factor: Multiplication factor for target class probs (>1 increases recall)

        Returns:
            Adjusted predictions
        """
        try:
            y_prob_adjusted = y_prob.copy()

            # Boost target class
            y_prob_adjusted[:, target_class] *= boost_factor

            # Renormalize
            y_prob_adjusted = y_prob_adjusted / y_prob_adjusted.sum(axis=1, keepdims=True)

            # Predict
            y_pred_adjusted = np.argmax(y_prob_adjusted, axis=1)

            logger.info(
                f"✅ Applied {boost_factor}x boost to Class {target_class}"
            )

            return y_pred_adjusted

        except Exception as e:
            logger.error(f"❌ Error adjusting predictions: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def find_optimal_threshold(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        target_class: int,
        metric: str = "f1",
    ) -> dict:
        """
        Find optimal boost factor for a target class

        Args:
            y_prob: Probability matrix
            y_true: True labels
            target_class: Class to optimize for (0 or 1)
            metric: Metric to optimize ('f1', 'recall', 'precision')

        Returns:
            Dictionary with optimal threshold and metrics
        """
        try:
            logger.info(f"🔍 Finding optimal threshold for Class {target_class}...")

            # ✅ BALANCED boost range - optimize for BOTH precision AND recall
            # 1.0-1.3 sweet spot: +8-12% recall gain with minimal precision loss
            boost_factors = np.arange(1.0, 1.3, 0.05)  # 1.0 to 1.3 (balanced approach)
            results = []

            for boost in boost_factors:
                y_pred = ThresholdTuner.adjust_predictions(
                    y_prob, target_class, boost
                )

                precision = precision_score(
                    y_true, y_pred, labels=[target_class], average="micro", zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, labels=[target_class], average="micro", zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, labels=[target_class], average="micro", zero_division=0
                )

                if metric == "f1":
                    score = f1
                elif metric == "recall":
                    score = recall
                elif metric == "precision":
                    score = precision
                elif metric == "balanced_precision":
                    # ✅ NEW: Maximize precision while maintaining recall >= 70%
                    min_recall = 0.70
                    if recall >= min_recall:
                        score = precision  # Maximize precision if recall is good
                    else:
                        score = 0  # Penalize if recall drops below threshold
                else:
                    score = f1

                results.append(
                    {
                        "boost_factor": boost,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "score": score,
                    }
                )

            # Find best: prioritize high-scoring results, with recall >= 70% for "balanced_precision"
            best_result = max(results, key=lambda x: x["score"])

            logger.info(
                f"✅ Optimal boost_factor: {best_result['boost_factor']:.2f}\n"
                f"   Precision: {best_result['precision']:.4f}\n"
                f"   Recall: {best_result['recall']:.4f}\n"
                f"   F1: {best_result['f1']:.4f}"
            )

            return {
                "target_class": target_class,
                "optimal_boost": best_result["boost_factor"],
                "metrics": {
                    "precision": best_result["precision"],
                    "recall": best_result["recall"],
                    "f1": best_result["f1"],
                },
                "all_results": results,
            }

        except Exception as e:
            logger.error(f"❌ Error finding optimal threshold: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def apply_optimal_thresholds(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        target_classes: list = None,
        metric: str = "f1",
    ) -> dict:
        """
        Optimize thresholds for multiple classes

        Args:
            y_prob: Probability matrix
            y_true: True labels
            target_classes: Classes to optimize (default: [0, 1])
            metric: Metric to optimize

        Returns:
            Dictionary with all optimized thresholds
        """
        try:
            if target_classes is None:
                target_classes = [0, 1]

            logger.info("🚀 Optimizing thresholds for multiple classes...")

            thresholds = {}
            for target_class in target_classes:
                result = ThresholdTuner.find_optimal_threshold(
                    y_prob, y_true, target_class, metric
                )
                thresholds[f"class_{target_class}"] = result

            logger.info("✅ Threshold optimization complete!")

            return thresholds

        except Exception as e:
            logger.error(f"❌ Error applying optimal thresholds: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def apply_thresholds_to_predictions(
        y_prob: np.ndarray,
        threshold_config: dict,
    ) -> np.ndarray:
        """
        Apply pre-computed thresholds to predictions

        Args:
            y_prob: Probability matrix
            threshold_config: Dictionary from find_optimal_threshold

        Returns:
            Adjusted predictions
        """
        try:
            y_pred = y_prob.copy()

            # Apply each class boost
            for key, config in threshold_config.items():
                target_class = config["target_class"]
                boost = config["optimal_boost"]

                y_pred[:, target_class] *= boost

            # Renormalize
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
            y_pred_final = np.argmax(y_pred, axis=1)

            logger.info("✅ Thresholds applied to predictions")

            return y_pred_final

        except Exception as e:
            logger.error(f"❌ Error applying thresholds: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def evaluate_threshold_performance(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        original_pred: np.ndarray,
        optimized_pred: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare original vs optimized predictions

        Args:
            y_prob: Probability matrix
            y_true: True labels
            original_pred: Original predictions
            optimized_pred: Threshold-optimized predictions

        Returns:
            DataFrame comparing metrics
        """
        try:
            comparison = []

            for class_id in [0, 1]:
                original_precision = precision_score(
                    y_true, original_pred, labels=[class_id], average="micro", zero_division=0
                )
                original_recall = recall_score(
                    y_true, original_pred, labels=[class_id], average="micro", zero_division=0
                )
                original_f1 = f1_score(
                    y_true, original_pred, labels=[class_id], average="micro", zero_division=0
                )

                optimized_precision = precision_score(
                    y_true, optimized_pred, labels=[class_id], average="micro", zero_division=0
                )
                optimized_recall = recall_score(
                    y_true, optimized_pred, labels=[class_id], average="micro", zero_division=0
                )
                optimized_f1 = f1_score(
                    y_true, optimized_pred, labels=[class_id], average="micro", zero_division=0
                )

                comparison.append(
                    {
                        "Class": class_id,
                        "Original_Precision": original_precision,
                        "Optimized_Precision": optimized_precision,
                        "Precision_Delta": optimized_precision - original_precision,
                        "Original_Recall": original_recall,
                        "Optimized_Recall": optimized_recall,
                        "Recall_Delta": optimized_recall - original_recall,
                        "Original_F1": original_f1,
                        "Optimized_F1": optimized_f1,
                        "F1_Delta": optimized_f1 - original_f1,
                    }
                )

            df_comparison = pd.DataFrame(comparison)
            logger.info(
                f"📊 Threshold Optimization Results:\n{df_comparison.to_string()}"
            )

            return df_comparison

        except Exception as e:
            logger.error(f"❌ Error evaluating threshold performance: {str(e)}")
            raise CustomException(e, sys)
