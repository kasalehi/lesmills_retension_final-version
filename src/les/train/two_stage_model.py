"""
Phase 3b: Two-Stage Ensemble Model for Churn Prediction
═════════════════════════════════════════════════════════════════════════════

Addresses the fundamental challenge: Class 0 and Class 1 are hard to distinguish.

Solution: Two simpler models instead of one complex 3-class model:

  Stage 1: BINARY CHURN DETECTION
  ──────────────────────────────
  Question: Is member at risk of churning (within 6 months)?
  Classes:  {Churn, No-Churn}
  Target:   High recall (catch most churners!)
  Expected: 85%+ recall on churn detection

  Stage 2: CHURN TIMING PREDICTION
  ────────────────────────────────
  Question: WHEN will they churn? Early or Medium?
  Classes:  {Class 0 (0-3mo), Class 1 (3-6mo)}
  Input:    ONLY members predicted as churners from Stage 1
  Target:   Good accuracy on timing prediction
  Expected: 75%+ accuracy on early vs medium churn

Overall Flow:
  1. Stage 1 predicts "No-Churn" → Output Class 2
  2. Stage 1 predicts "Churn" → Go to Stage 2
  3. Stage 2 predicts "Early" → Output Class 0
  4. Stage 2 predicts "Medium" → Output Class 1

Benefits:
  ✓ Simpler problems = better solutions
  ✓ Stage 1 focuses on easy binary task (85%+ recall achievable)
  ✓ Stage 2 focuses only on distinguishing similar classes
  ✓ More interpretable ("at-risk" → "likely timeframe")

Expected Results:
  Current:  64% Class 0 recall, 66% Class 1 recall
  After:    82-85% Class 0 recall, 83-85% Class 1 recall
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    balanced_accuracy_score, roc_auc_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib

from src.les.logger import logging

logger = logging.getLogger(__name__)


@dataclass
class TwoStageModelConfig:
    """Configuration for two-stage model"""

    # Stage 1: Binary Churn Detector
    stage1_params: dict = field(default_factory=lambda: {
        "clf__n_estimators": [300, 400, 500],
        "clf__max_depth": [4, 5, 6],
        "clf__learning_rate": [0.02, 0.05, 0.1],
    })

    # Stage 2: Timing Predictor (trained only on churn members)
    stage2_params: dict = field(default_factory=lambda: {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.05, 0.1],
    })

    # Class weights for each stage
    stage1_weights: dict = field(default_factory=lambda: {0: 10, 1: 1})  # 10x weight on churn
    stage2_weights: dict = field(default_factory=lambda: {0: 5, 1: 5})   # Balanced


class TwoStageEnsembleModel:
    """
    Two-stage churn prediction model.

    Stage 1: Detects churn (binary classification)
    Stage 2: Predicts churn timing (early vs medium)
    """

    def __init__(self, config: TwoStageModelConfig = None):
        """Initialize two-stage model"""
        self.config = config or TwoStageModelConfig()
        self.stage1_model = None
        self.stage2_model = None
        self.preprocessing_pipeline = None

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        preprocessor: Pipeline
    ) -> dict:
        """
        Train both stages of the ensemble model.

        Args:
            x_train: Training features
            y_train: Training labels (0, 1, 2)
            x_test: Test features
            y_test: Test labels
            preprocessor: Scikit-learn preprocessor pipeline

        Returns:
            Dictionary with results for both stages
        """

        logger.info("=" * 80)
        logger.info("TWO-STAGE ENSEMBLE MODEL TRAINING")
        logger.info("=" * 80)

        self.preprocessing_pipeline = preprocessor
        results = {}

        # =====================================================================
        # STAGE 1: BINARY CHURN DETECTOR
        # =====================================================================
        logger.info("\n📊 STAGE 1: BINARY CHURN DETECTOR (Churn vs No-Churn)")
        logger.info("─" * 80)

        # Create binary target: 0=Churn (classes 0+1), 1=No-Churn (class 2)
        y_train_binary = (y_train == 2).astype(int)  # 0=Churn, 1=No-Churn
        y_test_binary = (y_test == 2).astype(int)

        logger.info(f"Training data: {len(y_train_binary)} samples")
        logger.info(f"  Churn: {(y_train_binary == 0).sum()}")
        logger.info(f"  No-Churn: {(y_train_binary == 1).sum()}")

        # Train Stage 1
        self.stage1_model, stage1_results = self._train_stage(
            x_train, y_train_binary,
            x_test, y_test_binary,
            stage=1,
            params=self.config.stage1_params,
            weights=self.config.stage1_weights
        )

        results['stage1'] = stage1_results

        logger.info(f"✅ Stage 1 Complete: Recall={stage1_results['recall']:.4f}")

        # =====================================================================
        # STAGE 2: CHURN TIMING PREDICTOR
        # =====================================================================
        logger.info("\n📊 STAGE 2: CHURN TIMING PREDICTOR (Early vs Medium Churn)")
        logger.info("─" * 80)

        # Filter to only churn members (y_train ∈ {0, 1})
        churn_mask_train = y_train != 2
        churn_mask_test = y_test != 2

        x_train_churn = x_train[churn_mask_train]
        y_train_timing = y_train[churn_mask_train]

        x_test_churn = x_test[churn_mask_test]
        y_test_timing = y_test[churn_mask_test]

        logger.info(f"Training data (churn members only): {len(y_train_timing)} samples")
        logger.info(f"  Class 0 (0-3mo): {(y_train_timing == 0).sum()}")
        logger.info(f"  Class 1 (3-6mo): {(y_train_timing == 1).sum()}")

        if len(y_train_timing) > 0:
            # Train Stage 2
            self.stage2_model, stage2_results = self._train_stage(
                x_train_churn, y_train_timing,
                x_test_churn, y_test_timing,
                stage=2,
                params=self.config.stage2_params,
                weights=self.config.stage2_weights
            )

            results['stage2'] = stage2_results
            logger.info(f"✅ Stage 2 Complete: Recall={stage2_results['recall']:.4f}")
        else:
            logger.warning("⚠️  No churn members in training set - skipping Stage 2")
            results['stage2'] = None

        # =====================================================================
        # COMBINED PREDICTIONS ON TEST SET
        # =====================================================================
        logger.info("\n📊 COMBINED TWO-STAGE PREDICTIONS")
        logger.info("─" * 80)

        y_pred_combined = self._predict_combined(x_test, preprocessor)

        # Calculate combined metrics
        combined_results = {
            'accuracy': accuracy_score(y_test, y_pred_combined),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_combined),
            'f1_weighted': f1_score(y_test, y_pred_combined, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_combined),
            'report': classification_report(y_test, y_pred_combined, zero_division=0)
        }

        results['combined'] = combined_results

        logger.info(f"\n✅ Combined Model Results:")
        logger.info(f"   Accuracy: {combined_results['accuracy']:.4f}")
        logger.info(f"   Balanced Accuracy: {combined_results['balanced_accuracy']:.4f}")
        logger.info(f"   F1-Weighted: {combined_results['f1_weighted']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{combined_results['confusion_matrix']}")
        logger.info(f"\nClassification Report:\n{combined_results['report']}")

        return results

    def _train_stage(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        stage: int,
        params: dict,
        weights: dict
    ) -> tuple:
        """
        Train a single stage of the ensemble.

        Args:
            x_train, y_train, x_test, y_test: Data
            stage: Stage number (1 or 2)
            params: Hyperparameter grid
            weights: Class weights

        Returns:
            (trained_model, results_dict)
        """

        # Prepare sample weights
        sample_weights = np.array([weights[y] for y in y_train])

        # Create pipeline
        xgb_model = XGBClassifier(
            random_state=42,
            objective="binary:logistic" if stage == 1 else "binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0
        )

        pipe = Pipeline([
            ("preprocess", self.preprocessing_pipeline),
            ("clf", xgb_model)
        ])

        # Grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=cv,
            n_jobs=2,
            scoring="f1_weighted",
            verbose=1
        )

        logger.info(f"Fitting Stage {stage} with GridSearchCV (5-fold, {len(params)} combinations)...")
        grid.fit(x_train, y_train, clf__sample_weight=sample_weights)

        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"CV Score: {grid.best_score_:.4f}")

        # Evaluate on test set
        y_pred = grid.predict(x_test)
        y_prob = grid.predict_proba(x_test)[:, 1]  # Probability of class 1

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        try:
            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = np.nan
        except:
            roc_auc = np.nan

        results = {
            'model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }

        logger.info(f"Stage {stage} Test Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Weighted: {f1:.4f}")

        return grid.best_estimator_, results

    def _predict_combined(self, x_test: np.ndarray, preprocessor: Pipeline) -> np.ndarray:
        """
        Make combined predictions using both stages.

        Stage 1 determines churn (0) or no-churn (2)
        Stage 2 determines timing (0 vs 1) for churn cases
        """

        n_samples = x_test.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        # Stage 1 predictions
        stage1_probs = self.stage1_model.predict_proba(x_test)
        stage1_pred = self.stage1_model.predict(x_test)  # 0=Churn, 1=No-Churn

        for i in range(n_samples):
            if stage1_pred[i] == 1:  # No-Churn
                y_pred[i] = 2
            else:  # Churn (will use Stage 2)
                if self.stage2_model is not None:
                    stage2_pred = self.stage2_model.predict(x_test[i:i+1])
                    y_pred[i] = stage2_pred[0]
                else:
                    # Fallback: predict Class 0 if Stage 2 not available
                    y_pred[i] = 0

        return y_pred

    def save_models(self, artifact_dir: Path) -> dict:
        """Save both stage models"""
        stage1_path = artifact_dir / f"stage1_churn_detector.pkl"
        stage2_path = artifact_dir / f"stage2_timing_predictor.pkl"

        joblib.dump(self.stage1_model, stage1_path)
        if self.stage2_model is not None:
            joblib.dump(self.stage2_model, stage2_path)

        logger.info(f"✅ Models saved:")
        logger.info(f"   Stage 1: {stage1_path}")
        logger.info(f"   Stage 2: {stage2_path}")

        return {
            'stage1_path': str(stage1_path),
            'stage2_path': str(stage2_path) if self.stage2_model else None
        }

    def load_models(self, artifact_dir: Path):
        """Load both stage models"""
        stage1_path = artifact_dir / f"stage1_churn_detector.pkl"
        stage2_path = artifact_dir / f"stage2_timing_predictor.pkl"

        self.stage1_model = joblib.load(stage1_path)
        if stage2_path.exists():
            self.stage2_model = joblib.load(stage2_path)

        logger.info(f"✅ Models loaded from {artifact_dir}")
