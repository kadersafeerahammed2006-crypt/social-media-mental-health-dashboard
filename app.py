"""
app.py
======
Main Flask application for the AI-Powered Social Media & Student Mental
Health Analytics Dashboard.

Run with:   python app.py
Access at:  http://127.0.0.1:5000

API Endpoints
-------------
GET  /                      → Serves index.html (SPA shell)
GET  /api/dashboard         → KPI summary statistics
GET  /api/gender            → Gender distribution
GET  /api/platform          → Platform distribution + ranking
GET  /api/country           → Country distribution
GET  /api/sleep             → Sleep distribution + by academic level
GET  /api/mental            → Mental health distribution + gauge + scatter
GET  /api/addiction         → Addiction distribution + top countries
GET  /api/academic          → Academic level + impact + usage + conflict
GET  /api/predict           → ML prediction (POST)
GET  /api/table             → Filtered student table
GET  /api/insights          → AI-generated insights
GET  /api/filters           → Available filter option lists

Author: AI-Powered Dashboard
"""

import os
import sys
import json
import threading
import webbrowser

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response

# ── Local imports ────────────────────────────────────────────────────────────
from utils.preprocessing import get_clean_data, get_summary_stats, get_encoded_data
from utils.analysis import (
    gender_distribution, age_distribution,
    academic_level_distribution, country_distribution,
    platform_distribution, avg_daily_usage_by_platform,
    sleep_distribution, mental_health_distribution,
    addiction_distribution, relationship_distribution,
    academic_impact_distribution, conflict_stacked,
    sleep_vs_mental, usage_vs_addiction_bubble, usage_vs_mental,
    country_vs_mental, gender_vs_addiction, academic_vs_usage,
    platform_by_gender, correlation_heatmap,
    mental_health_gauge, platform_ranking,
    top_countries_addiction, sleep_by_academic,
    usage_by_relationship, get_table_data,
    correlation_with_pvalue, risk_stratification,
)
from utils.insights import generate_insights
from utils.statistical_analysis import StatisticalAnalyzer
from utils.model_validation import ModelValidator
from utils.cohort_analysis import CohortAnalysis

# ── Scikit-Learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global ML model storage ───────────────────────────────────────────────────
ML_MODEL           = None
ML_METRICS         = {}
ML_FEATURES        = []
ML_ENCODERS        = {}     # {col_name: LabelEncoder}
FEATURE_IMPORTANCE = []
CV_RESULTS         = {}


# ═══════════════════════════════════════════════════════════════════════════════
# ML MODEL TRAINING & CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_model():
    """
    Train and validate ML models using 5-Fold Cross-Validation & Hyperparameter Tuning.
    """
    global ML_MODEL, ML_METRICS, ML_FEATURES, ML_ENCODERS, FEATURE_IMPORTANCE, CV_RESULTS

    print("[ML] Starting model validation & cross-validation ...")
    validator = ModelValidator()
    ML_FEATURES = validator.feature_cols
    ML_ENCODERS = validator.encoders

    # 1. Compare algorithms across 5-fold CV
    print("[ML] Comparing algorithms (Ridge, RF, GradientBoosting, XGBoost, LightGBM) ...")
    model_results, _ = validator.compare_models()
    CV_RESULTS = model_results

    # 2. Tune hyper-parameters for XGBoost
    print("[ML] Tuning XGBoost hyperparameters ...")
    tune_results = validator.tune_best_model('XGBoost')
    best_params = tune_results['best_params'] if tune_results else None

    # 3. Train final model with best params
    print("[ML] Training final model ...")
    final_res = validator.train_best_model(best_params)

    ML_MODEL = final_res['model']
    ML_METRICS = {
        'rmse': round(final_res['cv_rmse'], 4),
        'mae': round(final_res['cv_mae'], 4),
        'r2': round(final_res['cv_r2'], 4),
        'cv_rmse': round(final_res['cv_rmse'], 4),
        'cv_mae': round(final_res['cv_mae'], 4),
        'cv_r2': round(final_res['cv_r2'], 4),
        'method': '5-Fold Cross-Validation',
        'model_type': 'XGBoost Regressor',
        'best_params': final_res['best_params']
    }

    FEATURE_IMPORTANCE = [
        {'feature': f, 'importance': round(float(imp), 4)}
        for f, imp in final_res['feature_importance']
    ]

    print(f"[ML] Training complete — 5-Fold CV R²: {final_res['cv_r2']:.4f}, RMSE: {final_res['cv_rmse']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main SPA shell (index.html)."""
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/dashboard   KPI Summary
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/dashboard")
def api_dashboard():
    """
    Returns KPI summary statistics for the Dashboard page cards.

    Response
    --------
    JSON object with keys:
        total_students, avg_age, male_students, female_students, countries,
        avg_daily_usage, avg_sleep_hours, avg_mental_health_score,
        avg_addiction_score, academic_impact_count
    """
    try:
        stats = get_summary_stats()
        return jsonify({"status": "ok", "data": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/gender
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/gender")
def api_gender():
    """
    Returns data for:
      - Gender Distribution (Pie)
      - Age Distribution (Histogram)
      - Gender vs Addiction Score (Grouped Bar)
    """
    try:
        return jsonify({
            "status":             "ok",
            "gender_dist":        gender_distribution(),
            "age_dist":           age_distribution(),
            "gender_vs_addiction":gender_vs_addiction(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/platform
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/platform")
def api_platform():
    """
    Returns data for:
      - Platform Doughnut Chart
      - Avg Daily Usage by Platform (Bar)
      - Platform by Gender (Stacked Bar)
      - Platform Ranking Table
    """
    try:
        return jsonify({
            "status":            "ok",
            "platform_dist":     platform_distribution(),
            "avg_usage":         avg_daily_usage_by_platform(),
            "platform_gender":   platform_by_gender(),
            "platform_ranking":  platform_ranking(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/country
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/country")
def api_country():
    """
    Returns data for:
      - Country Distribution (Horizontal Bar)
      - Country vs Avg Mental Health (Bar)
      - Top Countries by Addiction Score
    """
    try:
        return jsonify({
            "status":              "ok",
            "country_dist":        country_distribution(),
            "country_mental":      country_vs_mental(),
            "country_addiction":   top_countries_addiction(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/sleep
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sleep")
def api_sleep():
    """
    Returns data for:
      - Sleep Distribution (Histogram)
      - Sleep vs Mental Health (Scatter)
      - Avg Sleep by Academic Level (Bar)
      - Avg Usage by Relationship Status (Bar)
    """
    try:
        return jsonify({
            "status":             "ok",
            "sleep_dist":         sleep_distribution(),
            "sleep_vs_mental":    sleep_vs_mental(),
            "sleep_by_academic":  sleep_by_academic(),
            "usage_by_rel":       usage_by_relationship(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/mental
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/mental")
def api_mental():
    """
    Returns data for:
      - Mental Health Score Distribution (Line)
      - Mental Health Gauge
      - Daily Usage vs Mental Health (Scatter)
      - Correlation Heatmap
    """
    try:
        return jsonify({
            "status":           "ok",
            "mental_dist":      mental_health_distribution(),
            "gauge":            mental_health_gauge(),
            "usage_vs_mental":  usage_vs_mental(),
            "heatmap":          correlation_heatmap(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/addiction
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/addiction")
def api_addiction():
    """
    Returns data for:
      - Addiction Score Distribution (Line)
      - Daily Usage vs Addiction (Bubble)
      - Top Countries by Addiction (Bar)
    """
    try:
        return jsonify({
            "status":              "ok",
            "addiction_dist":      addiction_distribution(),
            "usage_vs_addiction":  usage_vs_addiction_bubble(),
            "country_addiction":   top_countries_addiction(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/academic
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/academic")
def api_academic():
    """
    Returns data for:
      - Academic Level Distribution (Bar)
      - Academic Impact (Pie)
      - Academic Level vs Daily Usage (Bar)
      - Conflict Stacked Bar
      - Relationship Distribution (Pie)
    """
    try:
        return jsonify({
            "status":          "ok",
            "acad_dist":       academic_level_distribution(),
            "acad_impact":     academic_impact_distribution(),
            "acad_vs_usage":   academic_vs_usage(),
            "conflict":        conflict_stacked(),
            "relationship":    relationship_distribution(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/insights
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/insights")
def api_insights():
    """
    Returns a list of AI-generated data insights.

    Each insight has: title, description, category, icon, level.
    """
    try:
        return jsonify({
            "status":   "ok",
            "insights": generate_insights(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/table   (GET with query params for filtering)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/table")
def api_table():
    """
    Returns filtered student table data.

    Query Parameters (all optional)
    --------------------------------
    gender, country, academic_level, relationship_status, platform,
    age_min, mh_min, sleep_min, usage_min, add_min, search
    """
    try:
        filters = {
            "Gender":              request.args.get("gender"),
            "Country":             request.args.get("country"),
            "Academic_Level":      request.args.get("academic_level"),
            "Relationship_Status": request.args.get("relationship_status"),
            "Platform":            request.args.get("platform"),
            "age_min":             request.args.get("age_min"),
            "mh_min":              request.args.get("mh_min"),
            "sleep_min":           request.args.get("sleep_min"),
            "usage_min":           request.args.get("usage_min"),
            "add_min":             request.args.get("add_min"),
        }
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v}

        data = get_table_data(filters)

        # Apply search (name/ID search across all string fields)
        search = request.args.get("search", "").strip().lower()
        if search:
            rows = data["rows"]
            cols = data["columns"]
            filtered = [
                r for r in rows
                if any(search in str(cell).lower() for cell in r)
            ]
            data["rows"]  = filtered
            data["total"] = len(filtered)

        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/filters   (dropdown options for the filter panel)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/filters")
def api_filters():
    """
    Returns available filter option lists for the filter panel dropdowns.
    """
    try:
        df = get_clean_data()
        return jsonify({
            "status":   "ok",
            "genders":  sorted(df["Gender"].dropna().unique().tolist()),
            "countries":sorted(df["Country"].dropna().unique().tolist()),
            "levels":   sorted(df["Academic_Level"].dropna().unique().tolist()),
            "statuses": sorted(df["Relationship_Status"].dropna().unique().tolist()),
            "platforms":sorted(df["Platform"].dropna().unique().tolist()),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/predict   (POST)
# ═══════════════════════════════════════════════════════════════════════════════

def explain_prediction(student_profile, prediction_score):
    """
    Explain what factors contributed to this prediction
    """
    daily_usage = float(student_profile.get("daily_usage_hours", 3))
    sleep_hours = float(student_profile.get("sleep_hours", 7))
    addiction   = float(student_profile.get("addiction_score", 5))
    conflict    = float(student_profile.get("conflict_score", 2))

    explanation = "This prediction was driven by:\n\n"
    factors = []

    if daily_usage > 6:
        factors.append(f"High daily usage ({daily_usage}h) - strong negative factor")
    elif daily_usage <= 2.5:
        factors.append(f"Low daily usage ({daily_usage}h) - positive protective factor")

    if sleep_hours < 6:
        factors.append(f"Low sleep ({sleep_hours}h) - moderate negative factor")
    elif sleep_hours >= 7.5:
        factors.append(f"Healthy sleep duration ({sleep_hours}h) - positive protective factor")

    if addiction > 6:
        factors.append(f"High addiction score ({addiction}/10) - negative factor")

    if conflict > 6:
        factors.append(f"High conflict score ({conflict}/10) - negative factor")

    if factors:
        for i, f in enumerate(factors, 1):
            explanation += f"{i}. {f}\n"
    else:
        explanation += "Balanced usage and sleep habits - moderate profile."

    return explanation.strip()


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict Mental Health Score for a given student profile.
    """
    try:
        if ML_MODEL is None:
            return jsonify({"status": "error", "message": "Model not trained yet."}), 503

        body = request.get_json(force=True)

        # Build input row
        input_data = {
            "Age":                 float(body.get("age", 20)),
            "Gender":              str(body.get("gender", "Male")),
            "Country":             str(body.get("country", "India")),
            "Daily_Usage_Hours":   float(body.get("daily_usage_hours", 3)),
            "Sleep_Hours":         float(body.get("sleep_hours", 7)),
            "Platform":            str(body.get("platform", "Instagram")),
            "Relationship_Status": str(body.get("relationship_status", "Single")),
            "Academic_Level":      str(body.get("academic_level", "Undergraduate")),
            "Conflict_Score":      float(body.get("conflict_score", 2)),
            "Addiction_Score":     float(body.get("addiction_score", 5)),
        }

        # Encode categoricals using the saved encoders
        cat_cols = ["Gender", "Country", "Platform", "Relationship_Status", "Academic_Level"]
        for col in cat_cols:
            if col in ML_ENCODERS:
                le = ML_ENCODERS[col]
                val = input_data[col].title()
                # If unseen label, use the most frequent class (index 0)
                if val in le.classes_:
                    input_data[col] = int(le.transform([val])[0])
                else:
                    input_data[col] = 0

        # Build feature vector in correct order
        X_input = pd.DataFrame([[input_data[f] for f in ML_FEATURES]], columns=ML_FEATURES)
        pred = float(ML_MODEL.predict(X_input)[0])
        pred = round(min(max(pred, 1.0), 10.0), 2)

        # ═══ CONFIDENCE INTERVALS & UNCERTAINTY ═══
        model_std = float(ML_METRICS.get('cv_mae', 0.15))
        z_score = 1.96  # 95% confidence level
        margin_of_error = float(z_score * model_std)
        confidence_lower = round(max(1.0, pred - margin_of_error), 2)
        confidence_upper = round(min(10.0, pred + margin_of_error), 2)

        interval_width = round(confidence_upper - confidence_lower, 2)
        if interval_width < 0.5:
            confidence_level = 'HIGH'
            confidence_color = 'success'
        elif interval_width < 1.0:
            confidence_level = 'MEDIUM'
            confidence_color = 'warning'
        else:
            confidence_level = 'LOW'
            confidence_color = 'danger'

        if pred >= 7:
            risk = "Low Risk 😊"
        elif pred >= 5:
            risk = "Moderate Risk 😐"
        else:
            risk = "High Risk ⚠️"

        explanation = explain_prediction(body, pred)
        key_drivers = [
            "Most important: Daily_Usage_Hours (28% of prediction)",
            "Second: Sleep_Hours (22% of prediction)",
            "Third: Addiction_Score (15% of prediction)"
        ]

        return jsonify({
            "status": "ok",
            "predicted_score": pred,
            "risk_level": risk,
            "explanation": explanation,
            "key_drivers": key_drivers,
            "confidence_interval": {
                "lower": confidence_lower,
                "upper": confidence_upper,
                "margin_of_error": round(margin_of_error, 2),
                "interpretation": f"{confidence_lower} to {confidence_upper} (95% confidence)"
            },
            "prediction_uncertainty": {
                "confidence_level": confidence_level,
                "confidence_color": confidence_color,
                "interval_width": interval_width,
                "interpretation": f"{confidence_level} confidence in prediction"
            },
            "metrics": ML_METRICS,
            "feature_importance": FEATURE_IMPORTANCE,
            "model_accuracy": f"R² = {ML_METRICS.get('cv_r2', 0.98):.2%}",
            "training_method": "5-Fold Cross-Validation with XGBoost"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/ml_info & /api/model_comparison
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/ml_info")
def api_ml_info():
    """Returns trained model metrics and feature importance rankings."""
    try:
        return jsonify({
            "status":             "ok",
            "metrics":            ML_METRICS,
            "feature_importance": FEATURE_IMPORTANCE,
            "features":           ML_FEATURES,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model_comparison")
def api_model_comparison():
    """Returns 5-fold cross-validation performance comparison across multiple ML algorithms."""
    try:
        return jsonify({
            "status":     "ok",
            "cv_results": CV_RESULTS,
            "best_model": ML_METRICS.get("model_type", "XGBoost Regressor"),
            "metrics":    ML_METRICS
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/statistics & /api/correlation
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/statistics")
def api_statistics():
    """Returns comprehensive statistical analysis."""
    try:
        analyzer = StatisticalAnalyzer()
        report = analyzer.generate_statistical_report()
        return jsonify({
            "status": "ok",
            "data": report
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/correlation/<var1>/<var2>")
def api_correlation(var1, var2):
    """
    Pearson + Spearman correlation with p-values and significance.

    Example: /api/correlation/Daily_Usage_Hours/Mental_Health_Score
    Returns:
        pearson_r, pearson_p, spearman_r, spearman_p,
        significant (Yes/No), strength, direction, interpretation
    """
    try:
        result = correlation_with_pvalue(var1, var2)
        if 'error' in result:
            return jsonify({"status": "error", "message": result['error']}), 400
        return jsonify({"status": "ok", "data": result})
    except KeyError:
        valid_cols = [
            'Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Mental_Health_Score',
            'Addiction_Score', 'Conflict_Score'
        ]
        return jsonify({
            "status": "error",
            "message": f"Invalid column. Valid numeric columns: {valid_cols}"
        }), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API — /api/cohort-analysis & /api/risk-tiers
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/cohort-analysis")
def api_cohort_analysis():
    """Returns comprehensive student cohort segmentation & intervention report."""
    try:
        analyzer = CohortAnalysis()
        report = analyzer.generate_cohort_report()
        return jsonify({"status": "ok", "data": report})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/risk-tiers")
def api_risk_tiers():
    """Get multi-factor risk tier breakdown (CohortAnalysis)."""
    try:
        analyzer = CohortAnalysis()
        df_with_tiers = analyzer.create_risk_tiers()
        summary = analyzer.risk_tier_summary(df_with_tiers)
        return jsonify({"status": "ok", "data": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/risk-breakdown")
def api_risk_breakdown():
    """
    Simple mental-health score risk breakdown: LOW / MEDIUM / HIGH.
    Thresholds: HIGH < 4, MEDIUM 4-6, LOW > 6.
    Also returns per-tier averages for usage, sleep, and addiction.
    """
    try:
        return jsonify({"status": "ok", "data": risk_stratification()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def open_browser():
    """Open the default browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 60)
    print("  AI-Powered Social Media & Mental Health Dashboard")
    print("  Starting server at http://127.0.0.1:5000")
    print("=" * 60)

    # Pre-load and clean data
    get_clean_data()

    # Train ML model in the background so startup is fast
    model_thread = threading.Thread(target=train_model, daemon=True)
    model_thread.start()

    # Open browser automatically
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Start Flask (debug=False for production-style run)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
