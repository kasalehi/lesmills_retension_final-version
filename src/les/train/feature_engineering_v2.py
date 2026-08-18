"""
Phase 3a: Advanced Feature Engineering for Churn Prediction
═══════════════════════════════════════════════════════════════════════════════

Creates discriminative features to better distinguish early churn (Class 0)
from medium churn (Class 1). Features are grouped into 4 categories:

1. Time-Based Features: Capture temporal patterns and acceleration
2. Behavioral Features: Capture engagement and payment patterns
3. Interaction Features: Combine multiple signals
4. Statistical Features: Capture variability and trends

Expected improvement: +10-20% recall on churn classes
"""

import numpy as np
import pandas as pd
from src.les.logger import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features for churn prediction.

    Args:
        df: DataFrame with base features (from ingestfortrain.py)

    Returns:
        DataFrame with new features added
    """

    df = df.copy()
    logger.info(f"Creating advanced features from {len(df)} records")
    logger.info(f"Input features: {len(df.columns)}")

    # =========================================================================
    # CATEGORY 1: TIME-BASED FEATURES
    # =========================================================================
    logger.info("Creating Time-Based Features...")

    # Feature 1: Days Since First Alert
    # Measures how long until first warning sign appears
    df['DaysSinceFirstAlert'] = df.apply(
        lambda row: min(
            row['DaysSinceLastAccessed'] if pd.notna(row['DaysSinceLastAccessed']) else 999,
            row['TenureDays'] - row['DaysSinceOriginalStart'] if pd.notna(row['DaysSinceOriginalStart']) else 999
        ),
        axis=1
    )
    df['DaysSinceFirstAlert'] = df['DaysSinceFirstAlert'].clip(lower=0)

    # Feature 2: Churn Acceleration
    # How fast is engagement declining? Early churn often has fast decline
    df['ChurnAcceleration'] = (
        (df['Visits_Last30d'] - df['Visits_Last90d'] / 3).fillna(0) /
        (df['Visits_Last90d'].fillna(1) + 1)
    )
    df['ChurnAcceleration'] = df['ChurnAcceleration'].clip(-10, 10)  # Clip outliers

    # Feature 3: Engagement Trend Direction
    # Is engagement increasing or decreasing? Direction matters for timing
    df['EngagementTrend'] = df.apply(
        lambda row: 1 if (row['Visits_Last30d'] > row['Visits_Last90d'] / 3 * 1.2)
                   else (-1 if row['Visits_Last30d'] < row['Visits_Last90d'] / 3 * 0.8 else 0),
        axis=1
    )

    # Feature 4: Tenure Bucket (in months)
    # Long-term vs new members churn differently
    df['TenureMonths'] = (df['TenureDays'] / 30).astype(int)
    df['TenureBucket'] = pd.cut(
        df['TenureMonths'],
        bins=[0, 3, 6, 12, 24, 999],
        labels=['0-3m', '3-6m', '6-12m', '12-24m', '24m+']
    ).astype(str)

    # =========================================================================
    # CATEGORY 2: BEHAVIORAL FEATURES
    # =========================================================================
    logger.info("Creating Behavioral Features...")

    # Feature 5: Visit Pattern Changed
    # Has engagement suddenly dropped? Indicator of imminent churn
    df['VisitPatternChanged'] = df.apply(
        lambda row: 1 if pd.notna(row['Visits_Last30d']) and pd.notna(row['Visits_Last90d'])
                   and row['Visits_Last30d'] < row['Visits_Last90d'] * 0.5 else 0,
        axis=1
    )

    # Feature 6: Payment Consistency
    # Regular payment = committed member; irregular = higher churn risk
    df['PaymentConsistency'] = df['RegularPayment'].apply(
        lambda x: 1 if pd.notna(x) and x != 'UnDefined' else 0
    )

    # Feature 7: Engagement Level (Quartile)
    # High engagement = lower churn; captures overall activity
    df['EngagementLevel'] = pd.qcut(
        df['TotalAttendanceToDate'].fillna(0),
        q=4,
        labels=['Very Low', 'Low', 'Medium', 'High'],
        duplicates='drop'
    ).astype(str)

    # Feature 8: Recent Activity Level
    # More relevant than total attendance for predicting imminent churn
    df['RecentActivityLevel'] = pd.qcut(
        df['Visits_Last30d'].fillna(0),
        q=4,
        labels=['Very Low', 'Low', 'Medium', 'High'],
        duplicates='drop'
    ).astype(str)

    # Feature 9: Is Member Inactive Recently
    # No visits in last 30 days = high churn risk
    df['IsRecentlyInactive'] = (
        (df['DaysSinceLastAccessed'] > 30) | (df['DaysSinceLastAccessed'].isna())
    ).astype(int)

    # =========================================================================
    # CATEGORY 3: INTERACTION FEATURES
    # =========================================================================
    logger.info("Creating Interaction Features...")

    # Feature 10: Age × Inactivity Interaction
    # Older members who don't visit are higher risk
    df['AgeInactivityInteraction'] = (
        (df['Age'].fillna(df['Age'].median()) / 10) *
        (df['DaysSinceLastAccessed'].fillna(0) / 30)
    ).clip(0, 100)

    # Feature 11: Tenure × Recent Activity Interaction
    # Loyal members (high tenure) who suddenly stop visiting = imminent churn
    df['LoyaltyActivityInteraction'] = (
        (df['TenureDays'].fillna(365) / 365) *
        (1 - df['Visits_Last30d'].fillna(0) / (df['TotalAttendanceToDate'].fillna(1) + 1))
    ).clip(0, 10)

    # Feature 12: Contract Usage Ratio
    # Actual usage vs contract term - low ratio suggests declining interest
    df['ContractUsageRatio'] = (
        df['TotalAttendanceToDate'].fillna(1) /
        (df['TermDays'].fillna(365) + 1)
    ).clip(0, 10)

    # Feature 13: Attendance Rate Change
    # How much faster/slower than historical average?
    df['AttendanceRateChange'] = (
        (df['Visits_Last30d'] / 4.3).fillna(0) -  # ~4.3 weeks per month
        (df['TotalAttendanceToDate'] / df['TenureDays']).fillna(0)
    ).clip(-10, 10)

    # =========================================================================
    # CATEGORY 4: STATISTICAL FEATURES
    # =========================================================================
    logger.info("Creating Statistical Features...")

    # Feature 14: Attendance Volatility
    # High volatility = unstable commitment
    # Estimated from visits_last30 vs visits_last90
    df['AttendanceVolatility'] = (
        df['Visits_Last30d'].fillna(0) - df['Visits_Last90d'].fillna(0) / 3
    ).abs()

    # Feature 15: Engagement Momentum
    # Is engagement accelerating or decelerating?
    # Calculate "momentum" of attendance
    df['EngagementMomentum'] = df.apply(
        lambda row: (row['Visits_Last30d'] - row['Visits_Last90d'] / 3) / (row['Visits_Last90d'] / 3 + 1)
        if pd.notna(row['Visits_Last90d']) and row['Visits_Last90d'] > 0 else 0,
        axis=1
    ).clip(-5, 5)

    # Feature 16: Normalized Attendance
    # Scale attendance by tenure (accounts for new vs old members)
    df['NormalizedAttendance'] = (
        df['TotalAttendanceToDate'].fillna(0) /
        (df['TenureDays'].fillna(365) / 30 + 1)  # Per month of tenure
    )

    # Feature 17: Days Since Consistent Activity
    # How long since member was consistently active?
    df['DaysSinceConsistentActivity'] = df['DaysSinceLastAccessed'].fillna(999)

    # Feature 18: Risk Score (Composite)
    # Combine multiple risk signals
    df['RiskScore'] = (
        (df['IsRecentlyInactive'] * 30) +  # Recent inactivity = high weight
        (df['VisitPatternChanged'] * 20) +  # Sudden drop = high weight
        (df['ChurnAcceleration'].clip(-1, 1) * 15) +  # Acceleration matters
        (df['AgeInactivityInteraction'] * 5)  # Age+inactivity matters
    ).clip(0, 100)

    # =========================================================================
    # CLEANUP & VALIDATION
    # =========================================================================
    logger.info("Validating features...")

    # Replace infinities with NaN, then fill with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Count new features added
    new_features = [
        'DaysSinceFirstAlert', 'ChurnAcceleration', 'EngagementTrend', 'TenureMonths', 'TenureBucket',
        'VisitPatternChanged', 'PaymentConsistency', 'EngagementLevel', 'RecentActivityLevel', 'IsRecentlyInactive',
        'AgeInactivityInteraction', 'LoyaltyActivityInteraction', 'ContractUsageRatio', 'AttendanceRateChange',
        'AttendanceVolatility', 'EngagementMomentum', 'NormalizedAttendance', 'DaysSinceConsistentActivity', 'RiskScore'
    ]

    logger.info(f"✅ Created {len(new_features)} new features:")
    logger.info(f"   Time-Based (4): DaysSinceFirstAlert, ChurnAcceleration, EngagementTrend, TenureMonths")
    logger.info(f"   Behavioral (5): VisitPatternChanged, PaymentConsistency, EngagementLevel, RecentActivityLevel, IsRecentlyInactive")
    logger.info(f"   Interaction (4): AgeInactivityInteraction, LoyaltyActivityInteraction, ContractUsageRatio, AttendanceRateChange")
    logger.info(f"   Statistical (5): AttendanceVolatility, EngagementMomentum, NormalizedAttendance, DaysSinceConsistentActivity, RiskScore")

    logger.info(f"📊 Output features: {len(df.columns)} (was {len(df.columns) - len(new_features)})")
    logger.info(f"📊 Output shape: {df.shape}")
    logger.info(f"📊 Missing values:\n{df.isnull().sum()}")

    return df


def create_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create same features for inference/prediction.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with engineered features
    """
    return create_advanced_features(df)


# For testing
if __name__ == "__main__":
    # Example usage
    print("Feature Engineering V2 Module")
    print("Use via: create_advanced_features(df)")
