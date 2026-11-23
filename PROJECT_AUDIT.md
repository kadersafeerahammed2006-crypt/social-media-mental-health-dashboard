# MindMetrics — Comprehensive Project Audit

**Audited:** 2026-09-04 | **Project:** MindMetrics — Student Mental Health & Social Media Analytics Platform
**Audit Phase:** PHASE 1 — Read-only audit. No modifications made during this phase.

---

## 1. Current Project Architecture

```
Social_Media_Mental_Health_Dashboard/
|-- app.py                    # 715 lines - Flask app + ML training + all API routes
|-- config.py                 # 31 lines - Paths and Flask settings
|-- requirements.txt          # 24 lines - Dependencies
|-- social_media.db           # 126 KB - SQLite database (primary data source)
|-- database/
|   |-- db_setup.py           # 128 lines - One-time DB initialiser
|   +-- test.db               # 8 KB - Stale test database (unused)
|-- dataset/
|   +-- Social media dataset.xlsx   # 78 KB - Raw Excel dataset
|-- models/
|   |-- best_model.joblib     # 584 KB - Saved Random Forest (NOT used by live app)
|   |-- encoders.joblib       # 3 KB - Saved encoders (NOT used by live app)
|   |-- metrics_summary.json  # 2 KB - Saved metrics (NOT referenced by live app)
|   +-- train_models.py       # 187 lines - Standalone training script (NOT called by app)
|-- utils/
|   |-- __init__.py
|   |-- data_loader.py        # 85 lines - SQLite/Excel loader with cache
|   |-- preprocessing.py      # 213 lines - Data cleaning + summary stats + encoding
|   |-- analysis.py           # 592 lines - 25 chart data aggregation functions
|   |-- statistical_analysis.py # 314 lines - StatisticalAnalyzer class
|   |-- model_validation.py   # 255 lines - ModelValidator class (XGBoost, LightGBM, etc.)
|   |-- cohort_analysis.py    # 262 lines - CohortAnalysis class
|   +-- insights.py           # 203 lines - Rule-based insight engine
|-- templates/
|   +-- index.html            # 842 lines - Single-page application (SPA) shell
|-- static/
|   |-- css/style.css         # 48 KB - Full styling
|   +-- js/dashboard.js       # 65 KB - All frontend logic
+-- reports/                  # Empty directory
```

Data flow: Excel/SQLite -> data_loader.py -> preprocessing.py -> utils/ modules -> Flask API routes -> JSON -> dashboard.js -> Chart.js

---

## 2. Existing Functionality Inventory

| Feature | Status | Location |
|---|---|---|
| Data loading (SQLite primary, Excel fallback) | WORKING | data_loader.py |
| Data cleaning & column renaming | WORKING | preprocessing.py |
| 25 chart data functions | WORKING | analysis.py |
| KPI summary statistics | WORKING | preprocessing.py::get_summary_stats() |
| Correlation analysis (Pearson + Spearman) | WORKING | statistical_analysis.py |
| ANOVA group comparisons | WORKING | statistical_analysis.py |
| Distribution/normality testing | WORKING | statistical_analysis.py |
| Rule-based AI insights (11 insights) | WORKING | insights.py |
| Risk stratification (LOW/MEDIUM/HIGH) | WORKING | analysis.py::risk_stratification() |
| Behavioral cohorts | WORKING | cohort_analysis.py |
| Demographic analysis | WORKING | cohort_analysis.py |
| Intervention recommendations | WORKING | cohort_analysis.py |
| ML model training (XGBoost, LightGBM, RF, Ridge) | WORKING | model_validation.py |
| 5-fold cross-validation | WORKING | model_validation.py |
| Hyperparameter tuning (GridSearchCV) | WORKING | model_validation.py |
| Live predictions with confidence intervals | WORKING | app.py::api_predict() |
| Student table with filtering | WORKING | analysis.py::get_table_data() |
| SPA with sidebar navigation | WORKING | index.html + dashboard.js |
| Dark theme UI | WORKING | style.css |
| Model comparison API | WORKING | app.py::api_model_comparison() |
| Data quality section | MISSING | Not implemented |
| Explainable AI (SHAP) | MISSING | Not implemented |
| Unit tests | MISSING | Not implemented |
| Synthetic vs. original data separation | MISSING | Is_Synthetic field exists but unused |
| Chi-square tests | MISSING | Not in statistical_analysis.py |
| Mann-Whitney U / Kruskal-Wallis tests | MISSING | Not in statistical_analysis.py |
| Security hardening | INCOMPLETE | Debug=True in config |
| PROJECT_ARCHITECTURE.md | MISSING | |
| DATA_DICTIONARY.md | MISSING | |
| MODEL_CARD.md | MISSING | |
| DATA_QUALITY_REPORT.md | MISSING | |

---

## 3. Problems Found

### CRITICAL Severity (5 bugs)

#### BUG-001 — Data Leakage in ML Model
**Severity:** CRITICAL | **File:** models/metrics_summary.json, model_validation.py, app.py

Addiction_Score accounts for 89.5% of feature importance when predicting Mental_Health_Score.
This indicates target leakage — both scores are highly correlated by construction (likely generated together).
The model achieves R2=0.97 not from learned patterns but because these two variables are almost linearly related.

Evidence: metrics_summary.json shows Addiction_Score: 0.8948 importance (nearly the entire model).
Impact: Invalidates all ML results. Any prediction tool built on this model is misleading.

Recommended Solution:
- Investigate correlation between Addiction_Score and Mental_Health_Score
- If r > 0.85, either:
  1. EXCLUDE Addiction_Score from features when predicting Mental_Health_Score, OR
  2. CHANGE TARGET to Addiction_Score (predicting addiction risk), OR
  3. PREDICT Academic_Impact (classification) which is more scientifically defensible
- Re-evaluate honestly after removal; expect R2 to drop significantly

---

#### BUG-002 — Fake/Hardcoded Feature Importance in Predictions
**Severity:** CRITICAL | **File:** app.py lines 534-537

The /api/predict response includes hardcoded key_drivers:
  "Most important: Daily_Usage_Hours (28% of prediction)"
  "Second: Sleep_Hours (22% of prediction)"
  "Third: Addiction_Score (15% of prediction)"

These percentages are FABRICATED — the actual XGBoost model shows Addiction_Score at 89%.
This actively misleads users about how the model works.

Recommended Solution: Use actual FEATURE_IMPORTANCE list (already computed) to generate explanation dynamically.

---

#### BUG-003 — ML Architecture Inconsistency (Two Separate Training Systems)
**Severity:** CRITICAL | **File:** models/train_models.py, app.py, model_validation.py

models/train_models.py trains Random Forest and saves to models/best_model.joblib.
app.py NEVER loads this file — it trains XGBoost from scratch at every startup via model_validation.py.
The saved model artifacts (584 KB) are completely unused and diverge from the live model.

Additionally, get_encoded_data() returns tuple (df, encoders) but docstring says pd.DataFrame — type inconsistency.

---

#### BUG-004 — _safe() Function Broken Abstraction
**Severity:** CRITICAL | **File:** analysis.py lines 57-58

def _safe(df: pd.DataFrame) -> pd.DataFrame:
    return get_clean_data().copy()  # ignores the df parameter entirely

Functions like sleep_vs_mental() call _safe(None).sample(min(400, len(_safe(None))), ...) 
which calls get_clean_data() TWICE unnecessarily.

---

#### BUG-005 — Statistically Incorrect Confidence Interval
**Severity:** CRITICAL | **File:** app.py lines 509-524

model_std = float(ML_METRICS.get('cv_mae', 0.15))
z_score = 1.96
margin_of_error = float(z_score * model_std)

Uses MAE as standard deviation in a z-based CI formula — statistically incorrect.
MAE != sigma. The formula z*sigma/sqrt(n) requires standard error of prediction, not MAE.

---

### HIGH Severity (6 bugs)

#### BUG-006 — FLASK_DEBUG=True in Config + No SECRET_KEY
**Severity:** HIGH | **File:** config.py line 28
FLASK_DEBUG = True in config but app.run(debug=False) overrides it.
Maintenance risk. Also: no SECRET_KEY configured for Flask sessions.

#### BUG-007 — Is_Synthetic Field Never Used in Analytics
**Severity:** HIGH | **File:** preprocessing.py, analysis.py
Is_Synthetic column preserved but never used to separate original vs. synthetic data,
report split counts, or prefer original data for evaluation.

#### BUG-008 — Chi-Square, Mann-Whitney U, Kruskal-Wallis Tests Missing
**Severity:** HIGH | **File:** statistical_analysis.py
Parametric tests only. Missing non-parametric alternatives needed for non-normal data.
Missing categorical-categorical tests (chi-square) entirely.

#### BUG-009 — No SHAP / Explainable AI Module
**Severity:** HIGH | **File:** N/A (missing)
explain_prediction() in app.py uses only threshold checks, not model-derived explanations.
No SHAP or genuine feature attribution implemented.

#### BUG-010 — No Unit Tests
**Severity:** HIGH | **File:** N/A (missing)
No tests/ directory. No safety net for any code changes.

#### BUG-011 — scipy Missing from requirements.txt
**Severity:** HIGH | **File:** requirements.txt
statistical_analysis.py and analysis.py both import from scipy but scipy is not listed.
Also joblib not listed (used in train_models.py).

---

### MEDIUM Severity (7 bugs)

#### BUG-012 — Causal Language in Insights
**File:** insights.py
Titles like "Heavy Social Media Use Drives Addiction" use causal language.
Should use: "is associated with", "shows a relationship with", "observed association".

#### BUG-013 — get_encoded_data() Type Hint Mismatch
**File:** preprocessing.py line 176
Type hint says -> pd.DataFrame but function returns (df, encoders) tuple.

#### BUG-014 — p-value Always Shown as p<0.001
**File:** statistical_analysis.py lines 89-100
_interpret_correlation() hardcodes "p<0.001" regardless of actual p-value.

#### BUG-015 — Monolithic app.py (715 Lines, Mixed Concerns)
**File:** app.py
Contains ML training, explain_prediction() helper, 15+ API routes, startup logic.
Violates single-responsibility principle.

#### BUG-016 — Pearson Heatmap Without Normality Check
**File:** analysis.py line 348
Always uses Pearson correlation for heatmap, even for non-normally distributed columns.
Spearman more appropriate as robust alternative.

#### BUG-017 — No Data Quality Dashboard/API
**File:** N/A (missing)
No endpoint or UI for: missing values, duplicates, completeness %, synthetic split,
invalid values, outlier counts.

#### BUG-018 — No Age Group Feature Engineering
**File:** preprocessing.py
Age used as raw numeric; no age bands (18-20, 21-23, 24+) for cohort analysis.

---

### LOW Severity (6 bugs)

| Bug | Description | File |
|---|---|---|
| BUG-019 | Stale database/test.db artifact | database/test.db |
| BUG-020 | Font Awesome not loaded; fa-* icons missing | index.html |
| BUG-021 | Summary stats missing original/synthetic split | preprocessing.py |
| BUG-022 | Missing age group and data source filters | dashboard.js |
| BUG-023 | Ethical disclaimer absent from UI | index.html |
| BUG-024 | Hardcoded nav count "25" in sidebar | index.html |

---

## 4. Priority Summary

| Count | Severity | Examples |
|---|---|---|
| 5 | CRITICAL | Data leakage, fake feature importance, broken CI |
| 6 | HIGH | Missing tests, missing statistical tests, Is_Synthetic unused |
| 7 | MEDIUM | Causal language, monolithic app.py, no data quality dashboard |
| 6 | LOW | Missing fonts, stale artifacts, hardcoded values |

---

## 5. Implementation Plan (Phased Approach)

### Phase 2 — Fix Data Pipeline
- Fix _safe() function (BUG-004)
- Fix get_encoded_data() type hint (BUG-013)
- Add Age_Group feature engineering (BUG-018)
- Add Is_Synthetic separation to summary stats (BUG-021)
- Fix requirements.txt — add scipy, joblib, shap (BUG-011)

### Phase 3 — Improve Analytics  
- Add Spearman heatmap option (BUG-016)
- Fix causal language in insights (BUG-012)
- Add data quality API (BUG-017)

### Phase 4 — Fix Statistical Analysis
- Add Chi-square test (BUG-008)
- Add Mann-Whitney U test (BUG-008)
- Add Kruskal-Wallis test (BUG-008)
- Fix p-value display bug (BUG-014)

### Phase 5 — Fix ML Model (Critical)
- Investigate and resolve data leakage (BUG-001)
- Change target: predict Addiction_Score with Mental_Health_Score excluded from features
  OR exclude Addiction_Score when predicting Mental_Health_Score
- Remove hardcoded feature importance strings (BUG-002)
- Fix confidence interval calculation (BUG-005)
- Fix ML architecture inconsistency (BUG-003)

### Phase 6 — Explainable AI
- Create utils/explainability.py with SHAP
- Add /api/explain endpoint (BUG-009)

### Phase 7 — Dashboard, Security & UX
- Fix config FLASK_DEBUG, add SECRET_KEY (BUG-006)
- Add Font Awesome CDN (BUG-020)
- Add ethical disclaimer (BUG-023)
- Extract helpers from app.py (BUG-015)

### Phase 8 — Testing
- Create tests/ directory (BUG-010)
- Unit tests for critical functions

### Phase 9 — Documentation
- PROJECT_ARCHITECTURE.md
- DATA_DICTIONARY.md
- MODEL_CARD.md
- DATA_QUALITY_REPORT.md

---

## 6. Architecture Recommendation

The current flat utils/ structure is acceptable for this project size.
File reorganisation is NOT recommended — it would break all imports without adding value.

Files to ADD:
- utils/explainability.py  (new — SHAP integration)
- utils/data_quality.py    (new — data quality metrics)
- tests/                   (new directory — unit tests)

Files to MODIFY IN-PLACE:
All existing files patched for bugs listed above.

---
*End of PROJECT_AUDIT.md*
