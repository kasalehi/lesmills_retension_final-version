"""
Complete Improved Training Workflow
Demonstrates how to use all improvement utilities together
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.les.logger import logger
from src.les.exception import CustomException
from src.les.train.feature_engineering import FeatureEngineer
from src.les.train.threshold_tuning import ThresholdTuner
from src.les.train.feature_importance import FeatureImportanceAnalyzer
import sys


class ImprovedTrainingWorkflow:
    """
    Complete workflow incorporating all improvements:
    1. Feature Engineering
    2. Model Training (with enhanced hyperparameters)
    3. Threshold Tuning
    4. Feature Importance Analysis
    """

    def __init__(self, data_path: str, output_dir: str = None):
        """
        Initialize workflow

        Args:
            data_path: Path to input parquet file
            output_dir: Directory to save results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir) if output_dir else Path("./artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.df_enhanced = None

    def step_1_add_features(self) -> pd.DataFrame:
        """
        Step 1: Add engineered features to data
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 1: FEATURE ENGINEERING")
            logger.info("=" * 80)

            logger.info(f"📥 Loading data from {self.data_path}...")
            self.df = pd.read_parquet(self.data_path)
            logger.info(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

            # Apply feature engineering
            engineer = FeatureEngineer()
            self.df_enhanced = engineer.create_all_features(self.df)

            logger.info(
                f"✅ Feature engineering complete!\n"
                f"   Columns before: {len(self.df.columns)}\n"
                f"   Columns after: {len(self.df_enhanced.columns)}\n"
                f"   New features: {len(self.df_enhanced.columns) - len(self.df.columns)}"
            )

            # Save enhanced data
            enhanced_path = self.output_dir / "df_enhanced_v2.parquet"
            self.df_enhanced.to_parquet(enhanced_path, index=False)
            logger.info(f"💾 Saved enhanced data to {enhanced_path}")

            return self.df_enhanced

        except Exception as e:
            logger.error(f"❌ Error in feature engineering step: {str(e)}")
            raise CustomException(e, sys)

    def step_2_prepare_for_training(self) -> tuple:
        """
        Step 2: Prepare data for training
        Separate features and target
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 2: DATA PREPARATION")
            logger.info("=" * 80)

            if self.df_enhanced is None:
                logger.warning("Enhanced data not found. Running feature engineering...")
                self.step_1_add_features()

            # Identify target column
            target_col = "Churned"
            if target_col not in self.df_enhanced.columns:
                raise CustomException(f"Target column '{target_col}' not found", sys)

            # Separate features and target
            y = self.df_enhanced[target_col]
            X = self.df_enhanced.drop(columns=[target_col])

            # Drop non-numeric columns that shouldn't be in model
            X = X.select_dtypes(include=[np.number])

            logger.info(f"✅ Data prepared:\n" f"   Features: {X.shape[1]}\n" f"   Target classes: {y.unique()}")

            return X, y

        except Exception as e:
            logger.error(f"❌ Error in data preparation step: {str(e)}")
            raise CustomException(e, sys)

    def step_3_analyze_model_results(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict:
        """
        Step 3: Analyze model performance and features
        (Called after training in train.py)
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 3: MODEL ANALYSIS")
            logger.info("=" * 80)

            # Feature importance analysis
            analyzer = FeatureImportanceAnalyzer()
            importance_report = analyzer.generate_feature_importance_report(
                model=model.named_steps["clf"],
                X_test=X_test,
                y_test=y_test,
                output_dir=str(self.output_dir / "importance"),
            )

            logger.info(
                f"✅ Feature analysis complete:\n"
                f"   Top features: {importance_report['top_5_features']}\n"
                f"   Churn drivers: {importance_report['churn_drivers']}"
            )

            return importance_report

        except Exception as e:
            logger.error(f"❌ Error in model analysis step: {str(e)}")
            raise CustomException(e, sys)

    def step_4_optimize_thresholds(
        self,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        y_pred_original: np.ndarray,
    ) -> dict:
        """
        Step 4: Optimize prediction thresholds
        (Called after training to improve class 0 & 1 recall/precision)
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 4: THRESHOLD OPTIMIZATION")
            logger.info("=" * 80)

            tuner = ThresholdTuner()

            # Find optimal thresholds for classes 0 & 1
            thresholds = tuner.apply_optimal_thresholds(
                y_prob=y_prob,
                y_true=y_test,
                target_classes=[0, 1],
                metric="f1",
            )

            # Apply thresholds
            y_pred_optimized = tuner.apply_thresholds_to_predictions(
                y_prob=y_prob,
                threshold_config=thresholds,
            )

            # Compare performance
            comparison = tuner.evaluate_threshold_performance(
                y_prob=y_prob,
                y_true=y_test,
                original_pred=y_pred_original,
                optimized_pred=y_pred_optimized,
            )

            logger.info(f"✅ Threshold optimization results:\n{comparison.to_string()}")

            # Save comparison
            comparison.to_csv(
                self.output_dir / "threshold_comparison.csv", index=False
            )

            return {
                "thresholds": thresholds,
                "optimized_predictions": y_pred_optimized,
                "comparison": comparison,
            }

        except Exception as e:
            logger.error(f"❌ Error in threshold optimization step: {str(e)}")
            raise CustomException(e, sys)

    def run_complete_workflow(self):
        """
        Execute complete workflow
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🚀 STARTING COMPLETE IMPROVED TRAINING WORKFLOW")
            logger.info("=" * 80 + "\n")

            # Step 1: Feature Engineering
            df_enhanced = self.step_1_add_features()

            # Step 2: Data Preparation
            X, y = self.step_2_prepare_for_training()

            logger.info("\n" + "=" * 80)
            logger.info("NEXT STEPS:")
            logger.info("=" * 80)
            logger.info("1. Use the enhanced data (df_enhanced_v2.parquet) for training")
            logger.info("2. Run your training DAG with the updated train.py")
            logger.info("3. After training, call step_3_analyze_model_results()")
            logger.info("4. Optionally call step_4_optimize_thresholds() for post-training optimization")
            logger.info("=" * 80 + "\n")

            return {
                "df_original": self.df,
                "df_enhanced": df_enhanced,
                "X": X,
                "y": y,
                "output_dir": str(self.output_dir),
            }

        except Exception as e:
            logger.error(f"❌ Workflow failed: {str(e)}")
            raise CustomException(e, sys)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_step_by_step():
    """
    Example: Run improvements step by step
    """
    # Initialize workflow
    workflow = ImprovedTrainingWorkflow(
        data_path="include/data/df_merged_balanced.parquet",
        output_dir="./artifacts/improved_workflow",
    )

    # Step 1: Add features
    df_enhanced = workflow.step_1_add_features()

    # Step 2: Prepare data
    X, y = workflow.step_2_prepare_for_training()

    # At this point, you would:
    # - Split into train/test
    # - Run training with improved train.py
    # - Get model results

    # Step 3: Analyze features (after training)
    # importance = workflow.step_3_analyze_model_results(
    #     model=trained_model,
    #     X_test=X_test,
    #     y_test=y_test,
    #     y_pred=y_pred,
    #     y_prob=y_prob
    # )

    # Step 4: Optimize thresholds (after training)
    # optimization = workflow.step_4_optimize_thresholds(
    #     y_test=y_test,
    #     y_prob=y_prob,
    #     y_pred_original=y_pred
    # )


def example_full_integration():
    """
    Example: Integrate with your existing training DAG
    """
    # In your DAG, after ingest step:
    workflow = ImprovedTrainingWorkflow(
        data_path="include/data/df_merged_enhanced.parquet",
        output_dir="include/artifacts",
    )

    results = workflow.run_complete_workflow()

    # Use results for training
    # training_pipeline.fit(results["X"], results["y"])


if __name__ == "__main__":
    # Run workflow
    workflow = ImprovedTrainingWorkflow(
        data_path="include/data/df_merged_balanced.parquet",
        output_dir="./artifacts/workflow_output",
    )

    workflow.run_complete_workflow()
