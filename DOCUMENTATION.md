# Project Documentation

## 1. Project Overview

This project is an AI-powered analytics dashboard designed to explore the relationship between social media usage and student mental health. It combines data analysis, machine learning, and a web-based dashboard to present insights in an interactive and visual way.

The system loads a dataset from Excel, cleans and preprocesses it, generates charts and summary statistics, trains a machine learning model to predict mental health scores, and serves everything through a Flask web application.

## 2. Objectives

The main goals of the project are to:

- Analyze how social media habits affect mental health
- Visualize trends through charts and KPI cards
- Predict mental health scores using a machine learning model
- Provide rule-based insights based on the dataset
- Deliver the results through a modern dashboard interface

## 3. Features

### Dashboard Features
- Interactive charts for demographics, sleep, usage, addiction, and mental health
- KPI summary cards for key statistics
- Filterable student data table
- AI-style insights generated from the dataset
- Prediction interface for mental health score estimation
- Responsive web interface with dark/light mode support

### Technical Features
- Flask backend with REST-style API endpoints
- Data preprocessing pipeline for cleaning and standardizing input data
- Random Forest regressor for prediction tasks
- Excel-based dataset loading using pandas and openpyxl

## 4. Technology Stack

### Backend
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- OpenPyXL

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap
- Chart.js

## 5. Project Structure

```text
Social_Media_Mental_Health_Dashboard/
├── app.py                     # Main Flask application and route definitions
├── README.md                  # Project overview and quick start guide
├── DOCUMENTATION.md           # Full project documentation
├── requirements.txt           # Python dependencies
├── dataset/                   # Input dataset files
├── reports/                   # Generated output reports
├── static/                    # CSS, JS, images for frontend
├── templates/                 # HTML templates
└── utils/                     # Data loading, preprocessing, analysis, and insight modules
```

## 6. Module Description

### app.py
Main application file. It:
- creates the Flask app
- defines API routes for dashboard data
- trains the machine learning model
- serves the frontend page

### utils/data_loader.py
Loads the Excel dataset from the dataset folder. It also caches the data to avoid repeatedly reading the file.

### utils/preprocessing.py
Cleans and standardizes the dataset by:
- removing duplicates
- fixing column names
- filling missing values
- converting appropriate data types
- clipping values to realistic ranges

### utils/analysis.py
Builds the data structures used by the charts and dashboard widgets. These functions return JSON-friendly data for the frontend.

### utils/insights.py
Generates human-readable insights from the dataset using rule-based logic. These insights appear in the dashboard to explain the data patterns.

## 7. Installation and Setup

### Prerequisites
Make sure Python is installed on your system.

### Step 1: Clone or open the project
Navigate to the project directory:

```bash
cd Social_Media_Mental_Health_Dashboard
```

### Step 2: Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset
Place the Excel dataset file inside the dataset folder:

```text
dataset/
└── Social media dataset.xlsx
```

### Step 5: Run the application

```bash
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000
```

## 8. Data Flow

The project follows this flow:

1. The Excel dataset is loaded by the data loader.
2. The preprocessing module cleans and standardizes the data.
3. Analysis functions compute summary statistics and chart data.
4. Flask routes expose this data as JSON to the frontend.
5. The frontend renders the data in charts, cards, and tables.
6. The machine learning model predicts mental health scores based on selected features.

## 9. API Endpoints

The Flask app exposes the following endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | / | Serves the main dashboard page |
| GET | /api/dashboard | Returns KPI summary statistics |
| GET | /api/gender | Returns gender and age-based data |
| GET | /api/platform | Returns platform usage and ranking data |
| GET | /api/country | Returns country-based analysis |
| GET | /api/sleep | Returns sleep-related analytics |
| GET | /api/mental | Returns mental health related analysis |
| GET | /api/addiction | Returns addiction analysis |
| GET | /api/academic | Returns academic performance analytics |
| POST | /api/predict | Predicts mental health score |
| GET | /api/insights | Returns generated insights |
| GET | /api/table | Returns filtered student data |
| GET | /api/filters | Returns available filter options |
| GET | /api/ml_info | Returns model metrics and feature importance |

## 10. Machine Learning Model

The project uses a Random Forest Regressor to predict the Mental Health Score.

### Target Variable
- Mental_Health_Score

### Input Features
- Age
- Gender
- Country
- Daily_Usage_Hours
- Sleep_Hours
- Platform
- Relationship_Status
- Academic_Level
- Conflict_Score
- Addiction_Score

### Training Setup
- Train/Test split: 80% / 20%
- Random state: 42
- Number of trees: 100

### Evaluation Metrics
- RMSE
- MAE
- R² Score

## 11. Frontend Overview

The frontend is a single-page dashboard built with HTML, CSS, and JavaScript. It uses Chart.js for visualizations and Bootstrap for layout. The main page loads data from the Flask APIs and renders the charts dynamically.

## 12. Customization Guide

You can extend the project in several ways:

- Add more charts by creating new functions in utils/analysis.py
- Add new insights in utils/insights.py
- Change the UI layout in templates/index.html or static/css/style.css
- Add new machine learning features by updating the feature list in app.py

## 13. Troubleshooting

### Dataset not found
If you see an error saying the dataset cannot be found, make sure the Excel file exists in the dataset folder and has the expected name.

### Module import errors
If imports fail, reinstall the required packages:

```bash
pip install -r requirements.txt
```

### Port already in use
If port 5000 is already occupied, stop the process using that port or change the Flask port in app.py.

## 14. License and Notes

This project is intended for academic and educational purposes as a final-year computer science project. It can be used as a base for further research, improvements, or portfolio demonstrations.
