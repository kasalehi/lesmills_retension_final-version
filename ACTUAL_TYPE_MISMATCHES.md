# Actual Type Mismatches Report
## Real Data Types vs SQL Schema

**Report Date**: 2026-08-19  
**Table**: repo.MembershipRetentionPredictions  
**Analysis**: Code conversion vs actual SQL column types

---

## ❌ **ACTUAL MISMATCHES FOUND**

### Mismatch #1: RegularPayment
```
Column Name:     RegularPayment
SQL Schema Type: int
Code Sends:      float64 (decimal number)
Issue:           Numeric type mismatch - float vs int
Code Line:       sql_insert_df['RegularPayment'] = pd.to_numeric(features_df['RegularPayment'], errors='coerce').astype('float64')

Example Data:
  Source: 49.99, 39.99, 29.99 (float values)
  After astype('float64'): 49.99, 39.99, 29.99 (still float)
  SQL Expects: Integer
  Problem: Decimal values cannot be stored as int ❌

Solution Options:
  Option 1: Keep float in SQL (recommended) - allows decimal amounts
  Option 2: Round to int in code: .round().astype('int64')
  Option 3: Multiply by 100 for cents: (value * 100).astype('int64')
  
Recommendation: Change SQL schema to float NOT int
```

---

### Mismatch #2: Base Amount
```
Column Name:     Base Amount
SQL Schema Type: int
Code Sends:      float64 (decimal number)
Issue:           Numeric type mismatch - float vs int
Code Line:       sql_insert_df['Base Amount'] = pd.to_numeric(features_df['Base Amount'], errors='coerce').astype('float64')

Example Data:
  Source: 49.99, 39.99, 29.99 (float values)
  After astype('float64'): 49.99, 39.99, 29.99 (still float)
  SQL Expects: Integer
  Problem: Decimal values cannot be stored as int ❌

Solution Options:
  Option 1: Keep float in SQL (recommended)
  Option 2: Round to int in code
  Option 3: Multiply by 100 for cents

Recommendation: Change SQL schema to float NOT int
```

---

### Mismatch #3: EWS_Pct
```
Column Name:     EWS_Pct
SQL Schema Type: int (likely)
Code Sends:      float64 (decimal percentage)
Issue:           Numeric type mismatch - float vs int
Code Line:       sql_insert_df['EWS_Pct'] = pd.to_numeric(features_df['ews_pct'], errors='coerce').astype('float64')

Example Data:
  Source: 0.75, 0.50, 0.95 (float percentages)
  After astype('float64'): 0.75, 0.50, 0.95 (still float)
  SQL Expects: Integer
  Problem: Decimal percentages cannot be stored as int ❌

Solution Options:
  Option 1: Keep float in SQL (recommended)
  Option 2: Convert to int percentage: (value * 100).astype('int64')
  
Recommendation: Change SQL schema to float NOT int
```

---

## 📋 **Complete Mismatch List by Column**

| # | Column | SQL Type | Code Sends | Actual Type | Mismatch | Severity |
|---|---|---|---|---|---|---|
| 1 | RunDate | date | string list | list[str] | ❌ | MEDIUM |
| 2 | MembershipID | uniqueidentifier | string series | object (str) | ❌ | LOW |
| 3 | MemberId | nvarchar | string series | object (str) | ✅ | - |
| 4 | Gender | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 5 | Age | int | int64 series | int64 | ✅ | - |
| 6 | SubCategory | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 7 | Channel | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 8 | ClubName | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 9 | MembershipTypeDesc | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 10 | Term | nvarchar | direct pass | object (str/NaN) | ✅ | - |
| 11 | TermDays | int | int64 series | int64 | ✅ | - |
| 12 | Lump Sum Flag | int (bit) | int64 series | int64 | ✅ | - |
| 13 | RegularPayment | int | **float64 series** | **float64** | ❌ | **HIGH** |
| 14 | Base Amount | int | **float64 series** | **float64** | ❌ | **HIGH** |
| 15 | TenureDays | int | int64 series | int64 | ✅ | - |
| 16 | DaysSinceOriginalStart | int | int64 series | int64 | ✅ | - |
| 17 | DaysSinceLastAccessed | int | int64 series | int64 | ✅ | - |
| 18 | TotalAttendanceToDate | int | int64 series | int64 | ✅ | - |
| 19 | Visits_Last90d | int | int64 series | int64 | ✅ | - |
| 20 | Visits_Last30d | int | int64 series | int64 | ✅ | - |
| 21 | EndedWithinCoolingPeriod | int | int64 series | int64 | ✅ | - |
| 22 | EWS_Pct | int (likely) | **float64 series** | **float64** | ❌ | **HIGH** |
| 23 | Risk_Band | nvarchar | string series | object (str) | ✅ | - |
| 24 | Engagement_Rate | float | float64 series | float64 | ✅ | - |
| 25 | Recent_Activity_Ratio | float | float64 series | float64 | ✅ | - |
| 26 | Declining_Engagement | float | float64 series | float64 | ✅ | - |
| 27 | Monthly_Avg_Visits | float | float64 series | float64 | ✅ | - |
| 28 | Payment_to_Attendance_Ratio | float | float64 series | float64 | ✅ | - |
| 29 | Recent_Payment_to_Visits | float | float64 series | float64 | ✅ | - |
| 30 | Tenure_Quartile | int | float64 series | float64 | ⚠️ | MEDIUM |
| 31 | Early_Churn_Risk | int | float64 series | float64 | ⚠️ | MEDIUM |
| 32 | Inactivity_Ratio | float | float64 series | float64 | ✅ | - |
| 33 | Attendance_Dropoff | int | float64 series | float64 | ⚠️ | MEDIUM |
| 34 | Access_Gap_Months | float | float64 series | float64 | ✅ | - |
| 35 | High_Inactivity | int | float64 series | float64 | ⚠️ | MEDIUM |
| 36 | Prediction | int | int64 series | int64 | ✅ | - |
| 37 | PredictionConfidence | float | float64 series | float64 | ✅ | - |
| 38 | EndDate | date | datetime series | datetime64 | ✅ | - |

---

## 🔴 **HIGH SEVERITY MISMATCHES** (Must Fix)

### 1. RegularPayment
```
SQL:    int
Code:   float64
Type:   pd.to_numeric(...).astype('float64')
Data:   49.99, 39.99, 29.99
Issue:  Decimal payment amounts sent to int column
Result: ❌ WILL FAIL or truncate decimals
Fix:    Change SQL to float OR cast to int in code
```

### 2. Base Amount
```
SQL:    int
Code:   float64
Type:   pd.to_numeric(...).astype('float64')
Data:   49.99, 39.99, 29.99
Issue:  Decimal amounts sent to int column
Result: ❌ WILL FAIL or truncate decimals
Fix:    Change SQL to float OR cast to int in code
```

### 3. EWS_Pct
```
SQL:    int
Code:   float64
Type:   pd.to_numeric(...).astype('float64')
Data:   0.75, 0.50, 0.95 (or 75.0, 50.0, 95.0)
Issue:  Percentage sent to int column
Result: ❌ WILL FAIL if decimals present
Fix:    Change SQL to float OR multiply by 100 and cast to int
```

---

## 🟡 **MEDIUM SEVERITY MISMATCHES** (May Cause Issues)

### 1. Tenure_Quartile
```
SQL:    int
Code:   float64 (from pd.to_numeric(..., errors='coerce'))
Type:   Engineered feature converted as float
Data:   1.0, 2.0, 3.0, 4.0
Issue:  Quartile values (1-4) sent as float instead of int
Result: ⚠️ Will work but wastes storage
Fix:    Cast to int64: .astype('int64')
```

### 2. Early_Churn_Risk
```
SQL:    int
Code:   float64 (from pd.to_numeric(..., errors='coerce'))
Type:   Binary flag converted as float
Data:   0.0, 1.0
Issue:  Binary values (0/1) sent as float instead of int
Result: ⚠️ Will work but wastes storage
Fix:    Cast to int64: .astype('int64')
```

### 3. Attendance_Dropoff
```
SQL:    int
Code:   float64 (from pd.to_numeric(..., errors='coerce'))
Type:   Binary indicator converted as float
Data:   0.0, 1.0
Issue:  Binary values (0/1) sent as float instead of int
Result: ⚠️ Will work but wastes storage
Fix:    Cast to int64: .astype('int64')
```

### 4. High_Inactivity
```
SQL:    int
Code:   float64 (from pd.to_numeric(..., errors='coerce'))
Type:   Binary flag converted as float
Data:   0.0, 1.0
Issue:  Binary values (0/1) sent as float instead of int
Result: ⚠️ Will work but wastes storage
Fix:    Cast to int64: .astype('int64')
```

---

## 📊 **Mismatch Summary**

| Severity | Count | Columns | Action |
|----------|-------|---------|--------|
| 🔴 HIGH | 3 | RegularPayment, Base Amount, EWS_Pct | **FIX IMMEDIATELY** |
| 🟡 MEDIUM | 4 | Tenure_Quartile, Early_Churn_Risk, Attendance_Dropoff, High_Inactivity | **FIX SOON** |
| ✅ OK | 31 | All others | No action needed |
| **TOTAL** | **38** | | |

---

## 🛠️ **Required Fixes**

### Option A: Fix in Code (Recommended)
```python
# HIGH PRIORITY
sql_insert_df['RegularPayment'] = pd.to_numeric(features_df['RegularPayment'], errors='coerce').astype('float64')
# Change to:
sql_insert_df['RegularPayment'] = pd.to_numeric(features_df['RegularPayment'], errors='coerce').astype('int64')  # OR keep as float64

sql_insert_df['Base Amount'] = pd.to_numeric(features_df['Base Amount'], errors='coerce').astype('float64')
# Change to:
sql_insert_df['Base Amount'] = pd.to_numeric(features_df['Base Amount'], errors='coerce').astype('int64')  # OR keep as float64

sql_insert_df['EWS_Pct'] = pd.to_numeric(features_df['ews_pct'], errors='coerce').astype('float64')
# Change to:
sql_insert_df['EWS_Pct'] = pd.to_numeric(features_df['ews_pct'], errors='coerce').astype('int64')  # OR keep as float64

# MEDIUM PRIORITY
sql_insert_df[feature] = pd.to_numeric(features_df[feature], errors='coerce')
# For Tenure_Quartile, Early_Churn_Risk, Attendance_Dropoff, High_Inactivity:
# Change to:
sql_insert_df[feature] = pd.to_numeric(features_df[feature], errors='coerce').astype('int64')
```

### Option B: Fix in SQL Schema (Alternative)
```sql
-- Change SQL column types to match what code sends:
ALTER TABLE repo.MembershipRetentionPredictions
  ALTER COLUMN RegularPayment float;  -- was int
  
ALTER TABLE repo.MembershipRetentionPredictions
  ALTER COLUMN [Base Amount] float;  -- was int
  
ALTER TABLE repo.MembershipRetentionPredictions
  ALTER COLUMN EWS_Pct float;  -- was int
```

---

## ✅ **Recommendation**

**Choice 1 (Recommended)**: Fix SQL Schema Types
- RegularPayment: int → float (allows decimals: 49.99)
- Base Amount: int → float (allows decimals: 39.99)
- EWS_Pct: int → float (allows decimals: 0.75)

**Why?**: Money values NEED decimals. Storing 49.99 as int 49 loses data.

---

**Status**: ⚠️ **7 MISMATCHES FOUND**
- 3 HIGH SEVERITY (will fail or lose data)
- 4 MEDIUM SEVERITY (works but inefficient)

**Action**: Fix SQL schema OR update code before next insert
