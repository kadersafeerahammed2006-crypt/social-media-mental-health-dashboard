"""
config.py
=========
Configuration settings for the Social Media & Mental Health Dashboard.
Defines paths for the SQLite database, Excel dataset, and machine learning models.
"""

import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "Social media dataset.xlsx")

# Database Paths
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(BASE_DIR, "social_media.db")

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoders.joblib")

# Flask Settings
FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = "127.0.0.1"
