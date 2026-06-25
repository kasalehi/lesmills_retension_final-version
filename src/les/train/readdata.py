from urllib.parse import quote_plus
import sys
import pandas as pd
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
from airflow.models import Variable
from pathlib import Path
def read_data():
    test_date = Variable.get("train_date") 
    # test_date = pd.Timestamp.today().date()
    ## get date for running if its null  
    hook = MsSqlHook(mssql_conn_id="mssql")
    query = f"""
    DECLARE @TestDate DATE = '{test_date}';
    DECLARE @ChurnWindowEnd DATE = DATEADD(WEEK, 52, @TestDate);
    DECLARE @MinVisits12m INT = 12;
    WITH base_members AS (
        SELECT
            fam.MemberID,
            fam.MembershipID,
            CAST(fam.[ActiveOn Date] AS date) AS active_on_date,
            CAST(fam.[End Date]      AS date) AS end_date
        FROM fact.LMNZ_ALLMemberships AS fam
        WHERE fam.[ActiveOn Date] <= @TestDate
          AND (fam.[End Date] >= @TestDate OR fam.[End Date] IS NULL)
          AND fam.SubCategory NOT IN ('Unvaccinated','Prepay', 'Complimentary') and Origin not in ('Transfer (Club to Club)')
          AND fam.Category IN ('Contract')
    ),
    labels AS (
        SELECT
            bm.MembershipID,
            CASE
                WHEN bm.end_date IS NOT NULL
                 AND bm.end_date >  @TestDate
                 AND bm.end_date <  @ChurnWindowEnd
                THEN 1 ELSE 0
            END AS churn_flag,
            CASE
                WHEN bm.end_date IS NULL OR bm.end_date <= @TestDate THEN NULL
                WHEN bm.end_date <= DATEADD(MONTH,  1, @TestDate) THEN  1
                WHEN bm.end_date <= DATEADD(MONTH,  2, @TestDate) THEN  2
                WHEN bm.end_date <= DATEADD(MONTH,  3, @TestDate) THEN  3
                WHEN bm.end_date <= DATEADD(MONTH,  4, @TestDate) THEN  4
                WHEN bm.end_date <= DATEADD(MONTH,  5, @TestDate) THEN  5
                WHEN bm.end_date <= DATEADD(MONTH,  6, @TestDate) THEN  6
                WHEN bm.end_date <= DATEADD(MONTH,  7, @TestDate) THEN  7
                WHEN bm.end_date <= DATEADD(MONTH,  8, @TestDate) THEN  8
                WHEN bm.end_date <= DATEADD(MONTH,  9, @TestDate) THEN  9
                WHEN bm.end_date <= DATEADD(MONTH, 10, @TestDate) THEN 10
                WHEN bm.end_date <= DATEADD(MONTH, 11, @TestDate) THEN 11
                WHEN bm.end_date <= DATEADD(MONTH, 12, @TestDate) THEN 12
                ELSE NULL
            END AS end_month_bucket
        FROM base_members bm
    ),
    mw AS (
        SELECT
            a.MembershipID,
            CAST(a.WeekBeginningDate AS date) AS week,
            COALESCE(a.WeekVisits, 0) AS engagement,
            a.[OnPauseThisWeek] AS pause
        FROM repo.MemberWeeklyAttendanceCounts a
        INNER JOIN base_members bm
            ON a.MembershipID = bm.MembershipID
        WHERE a.WeekBeginningDate >= DATEADD(YEAR, -1, @TestDate)
          AND a.WeekBeginningDate  <  @TestDate
    ),
    eligible AS (
        SELECT
            mw.MembershipID
        FROM mw
        GROUP BY mw.MembershipID
        HAVING SUM(mw.engagement) >= @MinVisits12m
    )
    SELECT
        mw.MembershipID AS member_id,
        mw.week,
        mw.engagement,
        CAST(mw.pause AS int) AS paused,
        lbl.end_month_bucket
    FROM mw
    JOIN eligible e
      ON e.MembershipID = mw.MembershipID
    LEFT JOIN labels lbl
      ON lbl.MembershipID = mw.MembershipID
    ORDER BY member_id, week;
    """
    df = hook.get_pandas_df(sql=query, parse_dates=["week"],
        dtype={
            "member_id": "string",
            "paused": "int16",
        })
    # df.to_csv(Path("/usr/local/airflow/include/data")/"ingest.csv")

    dg = df.drop(columns=["end_month_bucket"])

   
    return dg
    
