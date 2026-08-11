"""
Feature Importance Analysis for Class-Specific Performance
Analyzes which features are most important for predicting Classes 0 & 1 (Churn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.les.logger import logger
from src.les.exception import CustomException
import sys


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance for churn prediction"""

    @staticmethod
    def get_xgboost_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        Extract feature importance from XGBoost model

        Args:
            model: Trained XGBoost model (from best_estimator_.named_steps['clf'])
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        try:
            logger.info(f"📊 Extracting XGBoost feature importance (top {top_n})...")

            importance = model.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importance,
                    "Normalized_Importance": importance / importance.sum(),
                }
            )

            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            )

            logger.info(f"✅ Top {top_n} features:\n{importance_df.head(top_n).to_string()}")

            return importance_df.head(top_n)

        except Exception as e:
            logger.error(f"❌ Error extracting feature importance: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def analyze_churn_specific_importance(
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feature_names: list,
    ) -> dict:
        """
        Analyze which features are most important for churn classes

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names

        Returns:
            Dictionary with churn-specific analysis
        """
        try:
            logger.info("🔍 Analyzing churn-specific feature importance...")

            # Get feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            )
            importance_df = importance_df.sort_values("Importance", ascending=False)

            # Identify churn-related features
            churn_keywords = [
                "engagement",
                "visit",
                "attendance",
                "inactivity",
                "access",
                "dropoff",
                "ratio",
                "payment",
                "risk",
                "tenure",
            ]

            churn_features = importance_df[
                importance_df["Feature"].str.lower().str.contains(
                    "|".join(churn_keywords)
                )
            ]

            logger.info(
                f"📌 Churn-related features (top 10):\n"
                f"{churn_features.head(10).to_string()}"
            )

            return {
                "all_importance": importance_df,
                "churn_features": churn_features,
                "top_5_overall": importance_df.head(5)["Feature"].tolist(),
                "top_5_churn_related": churn_features.head(5)["Feature"].tolist(),
            }

        except Exception as e:
            logger.error(f"❌ Error analyzing churn-specific importance: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def visualize_importance(
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        output_path: str = None,
        top_n: int = 20,
    ) -> None:
        """
        Create visualization of feature importance

        Args:
            importance_df: DataFrame with feature importance
            title: Title for plot
            output_path: Path to save figure
            top_n: Number of features to display
        """
        try:
            logger.info(f"📈 Creating feature importance visualization...")

            fig, ax = plt.subplots(figsize=(12, 8))

            top_features = importance_df.head(top_n)

            ax.barh(top_features["Feature"], top_features["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title(title)
            ax.invert_yaxis()

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                logger.info(f"✅ Visualization saved to {output_path}")

            return fig

        except Exception as e:
            logger.error(f"❌ Error creating visualization: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def compare_model_versions(
        importance_dict: dict,
        model_names: list,
    ) -> pd.DataFrame:
        """
        Compare feature importance across different model versions

        Args:
            importance_dict: Dictionary {model_name: importance_df}
            model_names: List of model names to compare

        Returns:
            Comparison DataFrame
        """
        try:
            logger.info("🔄 Comparing feature importance across models...")

            # Merge importance scores from different models
            comparison_df = None

            for model_name in model_names:
                if model_name in importance_dict:
                    imp_df = importance_dict[model_name][["Feature", "Importance"]]
                    imp_df = imp_df.rename(columns={"Importance": model_name})

                    if comparison_df is None:
                        comparison_df = imp_df
                    else:
                        comparison_df = comparison_df.merge(
                            imp_df, on="Feature", how="outer"
                        )

            comparison_df = comparison_df.fillna(0)
            comparison_df = comparison_df.sort_values(by=model_names, ascending=False)

            logger.info(
                f"✅ Model comparison (top 10):\n{comparison_df.head(10).to_string()}"
            )

            return comparison_df

        except Exception as e:
            logger.error(f"❌ Error comparing models: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def generate_feature_importance_report(
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        output_dir: str = None,
    ) -> dict:
        """
        Generate comprehensive feature importance report

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save report

        Returns:
            Dictionary with all analysis results
        """
        try:
            logger.info("🚀 Generating comprehensive feature importance report...")

            feature_names = X_test.columns.tolist()

            # Extract importance
            importance_df = FeatureImportanceAnalyzer.get_xgboost_importance(
                model, feature_names
            )

            # Analyze churn-specific features
            churn_analysis = FeatureImportanceAnalyzer.analyze_churn_specific_importance(
                model, X_test, y_test, feature_names
            )

            # Create visualizations
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Overall importance
                FeatureImportanceAnalyzer.visualize_importance(
                    importance_df,
                    title="Top 20 Features by Overall Importance",
                    output_path=str(output_dir / "feature_importance_overall.png"),
                    top_n=20,
                )

                # Churn-specific
                FeatureImportanceAnalyzer.visualize_importance(
                    churn_analysis["churn_features"],
                    title="Churn-Related Features Importance",
                    output_path=str(output_dir / "feature_importance_churn.png"),
                    top_n=15,
                )

                logger.info(f"✅ Visualizations saved to {output_dir}")

            report = {
                "overall_importance": importance_df,
                "churn_analysis": churn_analysis,
                "top_5_features": importance_df.head(5)["Feature"].tolist(),
                "churn_drivers": churn_analysis["top_5_churn_related"],
            }

            logger.info(
                f"✅ Feature Importance Report:\n"
                f"   Top 5 Overall: {report['top_5_features']}\n"
                f"   Top 5 Churn-Related: {report['churn_drivers']}"
            )

            return report

        except Exception as e:
            logger.error(f"❌ Error generating feature importance report: {str(e)}")
            raise CustomException(e, sys)
