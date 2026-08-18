"""
Feature Engineering for SQL Insert
════════════════════════════════════════════════════════════════════════════════

Calculates engineered features from real features for SQL table insertion.
These features help explain churn patterns and improve business insights.

All features derived from:
- Tenure metrics (TenureDays, DaysSinceOriginalStart)
- Engagement metrics (TotalAttendanceToDate, Visits_Last30d, Visits_Last90d)
- Activity metrics (DaysSinceLastAccessed)
- Financial metrics (RegularPayment, Base Amount)
"""

import numpy as np
import pandas as pd
from src.les.logger import logging

logger = logging.getLogger(__name__)


def engineer_sql_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for SQL table insertion.

    Args:
        df: DataFrame with real features

    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()

    logger.info("🔧 Engineering features for SQL insertion...")

    # =========================================================================
    # 1. ENGAGEMENT_RATE: Visits per day of tenure
    # =========================================================================
    df['Engagement_Rate'] = (
        df['TotalAttendanceToDate'] / (df['TenureDays'].fillna(1) + 1)
    ).clip(0, 10)  # Cap at 10 visits/day
    logger.info("✅ Engagement_Rate: visits per day of tenure")

    # =========================================================================
    # 2. RECENT_ACTIVITY_RATIO: Recent visits vs historical average
    # =========================================================================
    # Visits_Last30d vs average of Visits_Last90d
    avg_visits_90d = df['Visits_Last90d'].fillna(0) / 3  # ~30 days per month
    df['Recent_Activity_Ratio'] = (
        df['Visits_Last30d'].fillna(0) / (avg_visits_90d + 0.001)
    ).clip(0, 5)  # Cap at 5x
    logger.info("✅ Recent_Activity_Ratio: recent vs historical visits")

    # =========================================================================
    # 3. DECLINING_ENGAGEMENT: Binary indicator of visit decline
    # =========================================================================
    # If last 30 days visits are <70% of 90-day average, it's declining
    avg_90d_visits = df['Visits_Last90d'].fillna(0) / 3
    df['Declining_Engagement'] = (
        (df['Visits_Last30d'].fillna(0) < avg_90d_visits * 0.7).astype(int)
    )
    logger.info("✅ Declining_Engagement: binary decline indicator")

    # =========================================================================
    # 4. MONTHLY_AVG_VISITS: Average visits per month
    # =========================================================================
    tenure_months = df['TenureDays'].fillna(1) / 30
    df['Monthly_Avg_Visits'] = (
        df['TotalAttendanceToDate'] / (tenure_months + 0.1)
    ).clip(0, 30)  # Cap at 30 visits/month
    logger.info("✅ Monthly_Avg_Visits: average visits per month")

    # =========================================================================
    # 5. PAYMENT_TO_ATTENDANCE_RATIO: Payment efficiency
    # =========================================================================
    # Higher ratio = less visits per payment (less value)
    df['Payment_to_Attendance_Ratio'] = (
        df['RegularPayment'].fillna(0) / (df['TotalAttendanceToDate'].fillna(1) + 1)
    ).clip(0, 100)
    logger.info("✅ Payment_to_Attendance_Ratio: payment per visit")

    # =========================================================================
    # 6. RECENT_PAYMENT_TO_VISITS: Recent payment efficiency
    # =========================================================================
    # Payment efficiency in recent period
    df['Recent_Payment_to_Visits'] = (
        df['RegularPayment'].fillna(0) / (df['Visits_Last30d'].fillna(1) + 1)
    ).clip(0, 100)
    logger.info("✅ Recent_Payment_to_Visits: recent payment per visit")

    # =========================================================================
    # 7. TENURE_QUARTILE: Quartile of tenure (1-4)
    # =========================================================================
    df['Tenure_Quartile'] = pd.qcut(
        df['TenureDays'].fillna(0),
        q=4,
        labels=[1, 2, 3, 4],
        duplicates='drop'
    ).astype('Int64')
    logger.info("✅ Tenure_Quartile: member tenure ranking")

    # =========================================================================
    # 8. EARLY_CHURN_RISK: Combined risk indicator
    # =========================================================================
    # High risk if: low engagement + high inactivity + declining visits
    high_inactivity = df['DaysSinceLastAccessed'].fillna(0) > 90
    low_engagement = df['Engagement_Rate'] < 0.5
    declining = df['Declining_Engagement'] == 1

    df['Early_Churn_Risk'] = (high_inactivity & low_engagement & declining).astype(int)
    logger.info("✅ Early_Churn_Risk: combined risk indicator")

    # =========================================================================
    # 9. INACTIVITY_RATIO: Proportion of tenure spent inactive
    # =========================================================================
    df['Inactivity_Ratio'] = (
        df['DaysSinceLastAccessed'].fillna(0) / (df['TenureDays'].fillna(1) + 1)
    ).clip(0, 10)  # Can exceed 1 if inactive longer than tenure
    logger.info("✅ Inactivity_Ratio: inactive days vs tenure")

    # =========================================================================
    # 10. ATTENDANCE_DROPOFF: Binary indicator of attendance decline
    # =========================================================================
    # If recent visits are <50% of average, it's a dropoff
    avg_monthly_visits = df['TotalAttendanceToDate'] / (tenure_months + 0.1)
    df['Attendance_Dropoff'] = (
        (df['Visits_Last30d'].fillna(0) < avg_monthly_visits * 0.5).astype(int)
    )
    logger.info("✅ Attendance_Dropoff: binary dropoff indicator")

    # =========================================================================
    # 11. ACCESS_GAP_MONTHS: Months since last access
    # =========================================================================
    df['Access_Gap_Months'] = (
        df['DaysSinceLastAccessed'].fillna(0) / 30
    ).clip(0, 36)  # Cap at 3 years
    logger.info("✅ Access_Gap_Months: months since last access")

    # =========================================================================
    # 12. HIGH_INACTIVITY: Binary high inactivity flag
    # =========================================================================
    # >90 days without access
    df['High_Inactivity'] = (
        (df['DaysSinceLastAccessed'].fillna(0) > 90).astype(int)
    )
    logger.info("✅ High_Inactivity: >90 days inactive flag")

    # =========================================================================
    # VALIDATION
    # =========================================================================
    engineered_features = [
        'Engagement_Rate', 'Recent_Activity_Ratio', 'Declining_Engagement',
        'Monthly_Avg_Visits', 'Payment_to_Attendance_Ratio', 'Recent_Payment_to_Visits',
        'Tenure_Quartile', 'Early_Churn_Risk', 'Inactivity_Ratio',
        'Attendance_Dropoff', 'Access_Gap_Months', 'High_Inactivity'
    ]

    logger.info(f"\n✅ Created {len(engineered_features)} engineered features:")
    for feat in engineered_features:
        missing = df[feat].isna().sum()
        logger.info(f"   {feat}: {missing} NaN values")

    return df


def get_feature_descriptions() -> dict:
    """
    Get descriptions of all engineered features.

    Returns:
        Dictionary mapping feature name to description
    """
    return {
        'Engagement_Rate': 'Visits per day of tenure. Higher = more engaged.',
        'Recent_Activity_Ratio': 'Recent visits vs historical average. >1 = increasing activity.',
        'Declining_Engagement': 'Binary: 1 if visits dropped >30% recently.',
        'Monthly_Avg_Visits': 'Average visits per month. Higher = more active.',
        'Payment_to_Attendance_Ratio': 'Payment per visit. Higher = less value (risk).',
        'Recent_Payment_to_Visits': 'Payment efficiency in recent period.',
        'Tenure_Quartile': 'Member tenure ranking (1=newest, 4=oldest).',
        'Early_Churn_Risk': 'Combined risk indicator (0=low risk, 1=high risk).',
        'Inactivity_Ratio': 'Days inactive vs tenure. Higher = more inactive.',
        'Attendance_Dropoff': 'Binary: 1 if recent visits dropped >50%.',
        'Access_Gap_Months': 'Months since last gym visit.',
        'High_Inactivity': 'Binary: 1 if >90 days without visit.'
    }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'TenureDays': [365, 180, 90],
        'TotalAttendanceToDate': [45, 20, 8],
        'Visits_Last30d': [3, 1, 0],
        'Visits_Last90d': [12, 6, 1],
        'DaysSinceLastAccessed': [5, 30, 120],
        'RegularPayment': [49.99, 39.99, 29.99]
    })

    result = engineer_sql_features(sample_data)
    print("\nEngineered Features Sample:")
    print(result[['Engagement_Rate', 'Recent_Activity_Ratio', 'High_Inactivity']])
