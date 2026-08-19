# Ingest Actual Feature Types Report
## Real Data Types from ingestfortrain.py SQL Query

**Report Date**: 2026-08-19  
**Source**: src/les/train/ingestfortrain.py  
**Analysis**: SQL query transformations and resulting data types

---

## 🔴 **CRITICAL ISSUES FOUND!**

The SQL ingest query is converting some numeric fields to **STRINGS**, but the model insert code tries to convert them back to **NUMERIC** → Results in **NULL/NaN**!

---

## 📊 **Feature Types from SQL Ingest Query**

| # | Column | SQL Source | SQL Transformation | Python Type | Value Examples | Issue |
|---|---|---|---|---|---|---|
| 1 | MembershipID | cm.MembershipID | UUID | object (str) | 9ae2fe93-d17b... | ✅ OK |
| 2 | Gender | cm.Gender | Direct | object (str) | M, F, U | ✅ OK |
| 3 | Age | DATEDIFF(YEAR...) | Direct calculation | int64 | 25, 35, 45 | ✅ OK |
| 4 | SubCategory | cm.SubCategory | Direct | object (str) | Full, Casual, Trial | ✅ OK |
| 5 | Channel | cm.Channel | Direct | object (str) | Online, Retail, Phone | ✅ OK |
| 6 | ClubName | cm.ClubName | Direct | object (str) | Ponsonby, City, Cause | ✅ OK |
| 7 | MembershipTypeDesc | cm.MembershipTypeDesc | Direct | object (str) | Individual, Family | ✅ OK |
| 8 | RegularPayment | cm.RegularPayment | **CASE statement** | **object (str)** | **'<20', '<25', '<30'** | ❌ **MAJOR** |
| 9 | Base Amount | cm.PaymentFrequency | **CASE statement** | **object (str)** | **'Fortnightly', 'Monthly'** | ❌ **MAJOR** |
| 10 | Term | cm.Term | **CASE statement** | **object (str)** | **'12Months', '6Months'** | ❌ **MAJOR** |
| 11 | TermDays | cm.TermDays | Direct | int64 | 365, 180, 90 | ✅ OK |
| 12 | Lump Sum Flag | cm.[Lump Sum Flag] | Direct | int64 | 0, 1 | ✅ OK |
| 13 | DaysSinceLastAccessed | DATEDIFF(DAY...) | Direct calculation | int64 | 5, 30, 120 | ✅ OK |
| 14 | TenureDays | DATEDIFF(DAY...) | Direct calculation | int64 | 365, 180, 730 | ✅ OK |
| 15 | DaysSinceOriginalStart | DATEDIFF(DAY...) | Direct calculation | int64 | 400, 600, 800 | ✅ OK |
| 16 | TotalAttendanceToDate | SUM(WeekVisits) | Direct calculation | int64 | 50, 120, 200 | ✅ OK |
| 17 | Visits_Last30d | SUM(WeekVisits) | Direct calculation | int64 | 3, 5, 8 | ✅ OK |
| 18 | Visits_Last90d | SUM(WeekVisits) | Direct calculation | int64 | 10, 20, 30 | ✅ OK |
| 19 | EndedWithinCoolingPeriod | cm.EndedWithinCoolingPeriod | Direct | int64 | 0, 1 | ✅ OK |
| 20 | end_date | cm.[End Date] | Direct | datetime64 | 2026-08-20, 2026-09-15 | ✅ OK |
| 21 | Churned | CASE statement | Direct calculation | int64 | 0, 1, 2, 3 | ✅ OK |
| 22 | ews_pct | Snapshot merge | Direct | float64 | 0.75, 0.50, 0.95 | ✅ OK |
| 23 | risk_band | Snapshot merge | Direct | object (str) | Low, Medium, High | ✅ OK |

---

## 🔴 **THREE CRITICAL STRING CONVERSIONS**

### Issue #1: RegularPayment
```sql
CASE
    WHEN bm.RegularPayment <= 20 THEN '<20'
    WHEN bm.RegularPayment <= 25 THEN '<25'
    WHEN bm.RegularPayment <= 30 THEN '<30'
    WHEN bm.RegularPayment <= 35 THEN '<35'
    WHEN bm.RegularPayment <= 40 THEN '<40'
    WHEN bm.RegularPayment <= 45 THEN '<45'
    ELSE '45+'
END AS [RegularPayment]
```

**Result**:
- SQL Returns: STRING categories ('<20', '<25', '<30', etc.)
- Python Type: object (str)
- Values: ['<20', '<25', '<30', '<35', '<40', '<45', '45+']

**In Model Insert Code**:
```python
sql_insert_df['RegularPayment'] = pd.to_numeric(
    features_df['RegularPayment'], 
    errors='coerce'
).astype('float64')
```

**What Happens**:
- Input: '<20' (string)
- pd.to_numeric('<20', errors='coerce') → NaN
- Result: **ALL RegularPayment values become NULL!** ❌

**SQL Type Expected**: int or float  
**Actually Received**: object (str)  
**Result**: Cannot convert → NULL

---

### Issue #2: Base Amount
```sql
CASE
    WHEN bm.PaymentFrequency LIKE 'Fort%' THEN 'Fortnightly'
    WHEN bm.PaymentFrequency LIKE 'Mon%' THEN 'Monthly'
    WHEN bm.PaymentFrequency LIKE 'Wee%' THEN 'Weekly'
    ELSE 'UnDefined'
END AS [Base Amount]
```

**Result**:
- SQL Returns: STRING categories ('Fortnightly', 'Monthly', 'Weekly', 'UnDefined')
- Python Type: object (str)
- Values: ['Fortnightly', 'Monthly', 'Weekly', 'UnDefined']

**In Model Insert Code**:
```python
sql_insert_df['Base Amount'] = pd.to_numeric(
    features_df['Base Amount'], 
    errors='coerce'
).astype('float64')
```

**What Happens**:
- Input: 'Monthly' (string)
- pd.to_numeric('Monthly', errors='coerce') → NaN
- Result: **ALL Base Amount values become NULL!** ❌

**SQL Type Expected**: int or float  
**Actually Received**: object (str)  
**Result**: Cannot convert → NULL

---

### Issue #3: Term
```sql
CASE
    WHEN bm.[Term] LIKE '12%' THEN '12Months'
    WHEN bm.[Term] LIKE '6%' THEN '6Months'
    WHEN bm.[Term] LIKE '24%' THEN '24Months'
    ELSE 'UnDefined'
END AS [Term]
```

**Result**:
- SQL Returns: STRING categories ('12Months', '6Months', '24Months', 'UnDefined')
- Python Type: object (str)
- Values: ['12Months', '6Months', '24Months', 'UnDefined']

**In Model Insert Code**:
```python
sql_insert_df['Term'] = features_df.get('Term', None)
# Direct pass - stays as string!
```

**What Happens**:
- Input: '12Months' (string)
- Direct pass → '12Months' (still string)
- SQL expects: nvarchar (string type)
- Result: ✅ This one actually works!

---

## 📊 **Type Conversion Problem Summary**

### What Ingest Sends

```
RegularPayment:  STRING ('<20', '<25', '<30', etc.)
Base Amount:     STRING ('Fortnightly', 'Monthly', 'Weekly')
Term:            STRING ('12Months', '6Months', '24Months')
```

### What Model Insert Code Tries To Do

```python
# Tries to convert strings to numbers:
pd.to_numeric('<20', errors='coerce')        # → NaN ❌
pd.to_numeric('Fortnightly', errors='coerce') # → NaN ❌
```

### Result

```
RegularPayment:  Expected float → Got NaN → NULL ❌
Base Amount:     Expected float → Got NaN → NULL ❌
Term:            Expected string → Got string → OK ✅
```

---

## ✅ **SOLUTION**

### Option 1: Fix Ingest Query (Best)
Change the SQL CASE statements to return NUMERIC values instead of strings:

```sql
-- BEFORE (current - returns string):
CASE
    WHEN bm.RegularPayment <= 20 THEN '<20'
    ...
END AS [RegularPayment]

-- AFTER (fixed - returns numeric):
CASE
    WHEN bm.RegularPayment <= 20 THEN 1
    WHEN bm.RegularPayment <= 25 THEN 2
    WHEN bm.RegularPayment <= 30 THEN 3
    ...
END AS [RegularPayment_Category]
```

Or return the actual numeric value:
```sql
CASE
    WHEN bm.RegularPayment <= 20 THEN 20
    WHEN bm.RegularPayment <= 25 THEN 25
    ...
END AS [RegularPayment_Bracket]
```

### Option 2: Handle String Conversion in Code
Map the string categories to numbers:

```python
# RegularPayment
payment_mapping = {
    '<20': 20,
    '<25': 25,
    '<30': 30,
    '<35': 35,
    '<40': 40,
    '<45': 45,
    '45+': 50
}
sql_insert_df['RegularPayment'] = features_df['RegularPayment'].map(payment_mapping).astype('float64')

# Base Amount (Payment Frequency)
frequency_mapping = {
    'Weekly': 52,
    'Fortnightly': 26,
    'Monthly': 12,
    'UnDefined': 0
}
sql_insert_df['Base Amount'] = features_df['Base Amount'].map(frequency_mapping).astype('float64')
```

### Option 3: Change SQL Schema
Accept STRING types in SQL table for these columns:

```sql
ALTER TABLE repo.MembershipRetentionPredictions
  ALTER COLUMN RegularPayment nvarchar(10);   -- was int
  ALTER COLUMN [Base Amount] nvarchar(15);    -- was int
```

---

## 🎯 **RECOMMENDATION**

**Option 1 (Fix Ingest Query)** is BEST because:
- ✅ Source data integrity (numeric values are numeric)
- ✅ Training model works correctly (no NULL values)
- ✅ Predictions work correctly (no NaN/NULL)
- ✅ No workarounds needed in code

**Next Steps**:
1. Modify ingestfortrain.py SQL query
2. Change CASE statements to return numeric values
3. Re-run ingest
4. Re-train model
5. Re-run predictions

---

## 📝 **Actual Ingest Data Examples**

From ingestfortrain.py line 167-174:

```python
df = hook.get_pandas_df(sql=query)
logger.info(f"✅ SQL rows retrieved: {len(df)}")
logger.info(f"📊 Data types:\n{df.dtypes}")
logger.info(f"📊 Sample data:\n{df.head()}")
```

**Data types you would see**:
```
MembershipID          object
Gender                object
Age                   int64
...
RegularPayment        object      ← PROBLEM: Should be numeric!
Base Amount           object      ← PROBLEM: Should be numeric!
Term                  object      ← OK: nvarchar expected
...
```

**Sample data**:
```
  MembershipID Gender Age ... RegularPayment Base Amount  Term
0 9ae2fe93-... M      35  ... '<30'          'Monthly'    '12Months'
1 3bd4c1f2-... F      42  ... '<45'          'Weekly'     '6Months'
2 8ef9a7d4-... U      28  ... '<20'          'Fortnightly' '24Months'
```

---

**Status**: 🔴 **CRITICAL** - 2 columns returning wrong types from ingest  
**Impact**: RegularPayment and Base Amount will be NULL in all predictions  
**Fix Required**: Before next training or prediction run

