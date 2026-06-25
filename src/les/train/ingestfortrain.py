"""
Enhanced ingest script for Les Mills training data.
Retrieves all available features from the database for better model accuracy.
"""

import sys
import pandas as pd
from src.les.exception import CustomException
from src.les.logger import logger
from airflow.models import Variable
from pathlib import Path
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook


def ingest_data_enhanced(snap: pd.DataFrame):
    """
    Enhanced ingest with cleaned features (pause/COVID/weak removed).

    Features (23 total):
    - Demographics: Gender, Age
    - Membership: SubCategory, Category, Channel, ClubName, MembershipTypeDesc
    - Financial: RegularPayment, Amount, Base Amount, PaymentFrequency, Term, TermDays, BillingDay
    - Tenure: TenureDays, DaysSinceOriginalStart
    - Engagement: TotalAttendanceToDate, Visits_Last90d, Visits_Last30d, DaysSinceLastAccessed
    - Transfer: Transferred_IN, Transferred_OUT, EndedWithinCoolingPeriod
    - Target: Churned
    - Snapshots: ews_pct, risk_band (added during merge)
    """
    start_date = pd.to_datetime(Variable.get("train_date")).strftime("%Y-%m-%d")

    try:
        logger.info(f"Start date used in SQL: {start_date}")
        hook = MsSqlHook(mssql_conn_id="mssql")

        query = f"""
DECLARE @TestDate DATE;
SET @TestDate = '{start_date}';

WITH base_memberships AS (
    SELECT
        cm.MembershipID,
        cm.MemberID,
        cm.[Start Date],
        cm.[End Date],
        cm.[Status Desc],
        cm.SubCategory,
        cm.Category,
        cm.Channel,
        cm.ClubName,
        cm.MembershipTypeDesc,
        cm.RegularPayment,
        cm.Gender,
        cm.DOB,
        cm.Discount,
        cm.Amount,
        cm.[Base Amount],
        cm.PaymentFrequency,
        cm.[Term],
        cm.TermDays,
        cm.BillingDay,
        cm.[Lump Sum Flag],
        cm.[Original Start Date],
        cm.DateLastAccessed,
        cm.Transferred_IN,
        cm.Transferred_OUT,
        cm.EndedWithinCoolingPeriod,
        cm.[Member Name]
    FROM fact.LMNZ_ALLMemberships cm
    WHERE cm.[Start Date] < @TestDate
      AND (cm.[End Date] > @TestDate OR cm.[End Date] IS NULL)
      AND cm.SubCategory NOT IN ('Unvaccinated','Prepay', 'Complimentary')
),

churn_target AS (
    SELECT
        MembershipID,
        CASE
            WHEN [End Date] BETWEEN @TestDate AND DATEADD(MONTH, 3, @TestDate) THEN 1
            WHEN [End Date] > DATEADD(MONTH, 3, @TestDate)
             AND [End Date] <= DATEADD(MONTH, 6, @TestDate) THEN 2
            ELSE 3
        END AS Churned
    FROM base_memberships
),

tenure_features AS (
    SELECT
        MembershipID,
        DATEDIFF(DAY, [Start Date], @TestDate) AS TenureDays,
        DATEDIFF(DAY, [Original Start Date], @TestDate) AS DaysSinceOriginalStart,
        DATEDIFF(DAY, @TestDate, [End Date]) AS DaysToContractEnd,
        DATEDIFF(YEAR, DOB, GETDATE()) AS Age
    FROM base_memberships
),

attendance_features AS (
    SELECT
        att.MembershipID,
        SUM(att.WeekVisits) AS TotalAttendanceToDate,
        SUM(CASE WHEN att.WeekBeginningDate >= DATEADD(DAY, -30, @TestDate)
                 THEN att.WeekVisits ELSE 0 END) AS Visits_Last30d,
        SUM(CASE WHEN att.WeekBeginningDate >= DATEADD(DAY, -90, @TestDate)
                 THEN att.WeekVisits ELSE 0 END) AS Visits_Last90d,
        MAX(att.WeekBeginningDate) AS LastVisitDate,
        DATEDIFF(DAY, MAX(att.WeekBeginningDate), @TestDate) AS DaysSinceLastVisit,
        SUM(CASE WHEN att.[OnPauseThisWeek] = 1 THEN 1 ELSE 0 END) AS WeeksPausedToDate
    FROM repo.MemberWeeklyAttendanceCounts att
    WHERE att.WeekBeginningDate < @TestDate
    GROUP BY att.MembershipID
),

final_dataset AS (
    SELECT
        bm.MembershipID,
        bm.Gender,
        tf.Age,
        bm.SubCategory,
        bm.Category,
        bm.Channel,
        bm.ClubName,
        bm.MembershipTypeDesc,
        bm.RegularPayment,
        bm.Amount,
        bm.[Base Amount],
        bm.PaymentFrequency,
        bm.[Term],
        bm.TermDays,
        bm.BillingDay,
        bm.[Lump Sum Flag],
        CASE
            WHEN bm.DateLastAccessed IS NULL THEN 999999
            WHEN bm.DateLastAccessed > @TestDate THEN 999999
            ELSE DATEDIFF(DAY, bm.DateLastAccessed, @TestDate)
        END AS DaysSinceLastAccessed,
        tf.TenureDays,
        tf.DaysSinceOriginalStart,
        af.TotalAttendanceToDate,
        af.Visits_Last30d,
        af.Visits_Last90d,
        bm.Transferred_IN,
        bm.Transferred_OUT,
        bm.EndedWithinCoolingPeriod,
        bm.[End Date] AS end_date,
        ct.Churned
    FROM base_memberships bm
    LEFT JOIN tenure_features tf ON bm.MembershipID = tf.MembershipID
    LEFT JOIN attendance_features af ON bm.MembershipID = af.MembershipID
    LEFT JOIN churn_target ct ON bm.MembershipID = ct.MembershipID
)

SELECT * FROM final_dataset
WHERE TotalAttendanceToDate > 0
ORDER BY MembershipID;
        """

        logger.info("Executing enhanced ingest query...")
        df = hook.get_pandas_df(sql=query)

        logger.info(f"✅ SQL rows retrieved: {len(df)}")
        logger.info(f"📊 Columns: {list(df.columns)}")
        logger.info(f"📊 Data types:\n{df.dtypes}")

        # Show sample data
        logger.info(f"📊 Sample data:\n{df.head()}")

        # Data quality checks
        null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        logger.info(f"📊 Null percentages:\n{null_pct[null_pct > 0]}")

        # Prepare for merge with snapshots
        df["MembershipID"] = df["MembershipID"].map(str)
        snaps = snap[['member_id', 'ews_pct', 'risk_band']].copy()
        df["member_id"] = df["MembershipID"].str.strip().str.lower()
        snaps["member_id"] = snaps["member_id"].astype(str).str.strip().str.lower()

        logger.info(f"SQL rows: {len(df)}")
        logger.info(f"Snap rows: {len(snaps)}")

        merged = df.merge(snaps, how="inner", on="member_id")
        logger.info(f"✅ Merged rows: {len(merged)}")

        # Save artifacts
        DATA_DIR = Path("/usr/local/airflow/include/data")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        sql_path = DATA_DIR / "df_sql_enhanced.parquet"
        merged_path = DATA_DIR / "df_merged_enhanced.parquet"

        df.to_parquet(sql_path, index=False)
        merged.to_parquet(merged_path, index=False)

        logger.info(f"📦 Saved to:\n  - {sql_path}\n  - {merged_path}")

        return {
            "df_sql_path": str(sql_path),
            "df_merged_path": str(merged_path),
            "row_count": len(merged),
            "column_count": len(merged.columns),
            "columns": list(merged.columns)
        }

    except Exception as e:
        logger.error(f"❌ Enhanced ingest failed: {str(e)}")
        raise CustomException(e, sys)


# Alias for backward compatibility - DAGs call ingest_data()
ingest_data = ingest_data_enhanced
