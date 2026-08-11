"""
Feature Engineering for Les Mills Churn Prediction
Adds derived features to improve model performance on Classes 0 & 1 (Churn)
"""

import pandas as pd
import numpy as np
from src.les.logger import logger
from src.les.exception import CustomException
import sys


class FeatureEngineer:
    """Generate enhanced features for better churn prediction"""

    @staticmethod
    def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement-based features to better distinguish churners
        """
        try:
            logger.info("🔧 Adding engagement features...")

            # Engagement rate relative to tenure
            df['Engagement_Rate'] = df['Visits_Last30d'] / (df['TenureDays'] + 1)

            # Recent vs historical engagement
            df['Recent_Activity_Ratio'] = df['Visits_Last30d'] / (df['Visits_Last90d'] + 1)

            # Trend: declining engagement
            df['Declining_Engagement'] = df['Visits_Last90d'] - df['Visits_Last30d']

            # Engagement per month of membership
            df['Monthly_Avg_Visits'] = df['TotalAttendanceToDate'] / (
                (df['TenureDays'] / 30) + 1
            )

            logger.info("✅ Engagement features added")
            return df

        except Exception as e:
            logger.error(f"❌ Error adding engagement features: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial-based features to identify payment-to-engagement mismatches
        """
        try:
            logger.info("🔧 Adding financial features...")

            # Value for money indicator
            df['Payment_to_Attendance_Ratio'] = (
                df['RegularPayment'] / (df['TotalAttendanceToDate'] + 1)
            )

            # Recent value proposition
            df['Recent_Payment_to_Visits'] = (
                df['RegularPayment'] / (df['Visits_Last30d'] + 1)
            )

            logger.info("✅ Financial features added")
            return df

        except Exception as e:
            logger.error(f"❌ Error adding financial features: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features to identify early-stage churn risk
        """
        try:
            logger.info("🔧 Adding tenure features...")

            # Segment tenure into quartiles
            df['Tenure_Quartile'] = pd.qcut(
                df['TenureDays'], q=4, labels=[0, 1, 2, 3], duplicates='drop'
            )

            # Early tenure risk (first 6 months with low engagement)
            df['Early_Churn_Risk'] = (
                (df['TenureDays'] < 180) & (df['DaysSinceLastAccessed'] > 30)
            ).astype(int)

            # Inactivity relative to tenure
            df['Inactivity_Ratio'] = df['DaysSinceLastAccessed'] / (
                df['TenureDays'] + 1
            )

            logger.info("✅ Tenure features added")
            return df

        except Exception as e:
            logger.error(f"❌ Error adding tenure features: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def add_risk_escalation_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that identify escalating risk patterns
        """
        try:
            logger.info("🔧 Adding risk escalation features...")

            # Recent activity drop-off
            df['Attendance_Dropoff'] = (
                df['Visits_Last30d'] < (df['Visits_Last90d'] / 3)
            ).astype(int)

            # Days since last access relative to payment frequency
            df['Access_Gap_Months'] = df['DaysSinceLastAccessed'] / 30

            # Consistent inactivity indicator
            df['High_Inactivity'] = (df['DaysSinceLastAccessed'] > 60).astype(int)

            logger.info("✅ Risk escalation features added")
            return df

        except Exception as e:
            logger.error(f"❌ Error adding risk escalation features: {str(e)}")
            raise CustomException(e, sys)

    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations
        """
        try:
            logger.info("🚀 Starting comprehensive feature engineering...")

            df = cls.add_engagement_features(df)
            df = cls.add_financial_features(df)
            df = cls.add_tenure_features(df)
            df = cls.add_risk_escalation_features(df)

            logger.info(
                f"✅ Feature engineering complete!\n"
                f"   Original features: ~23\n"
                f"   New features added: ~14\n"
                f"   Total features: ~37"
            )

            return df

        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {str(e)}")
            raise CustomException(e, sys)


def apply_features_to_data(df_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Load data, apply features, and save result

    Args:
        df_path: Path to input parquet file
        output_path: Path to save enhanced parquet file

    Returns:
        Enhanced DataFrame
    """
    try:
        logger.info(f"📥 Loading data from {df_path}...")
        df = pd.read_parquet(df_path)

        engineer = FeatureEngineer()
        df_enhanced = engineer.create_all_features(df)

        if output_path:
            logger.info(f"💾 Saving enhanced data to {output_path}...")
            df_enhanced.to_parquet(output_path, index=False)
            logger.info(f"✅ Saved!")

        return df_enhanced

    except Exception as e:
        logger.error(f"❌ Error processing data: {str(e)}")
        raise CustomException(e, sys)
