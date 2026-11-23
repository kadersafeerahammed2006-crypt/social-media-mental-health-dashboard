# 🧠 AI-Powered Social Media & Student Mental Health Analytics Dashboard

> **Final Year B.Sc. Computer Science Project** — An interactive, production-ready analytics dashboard combining data science, machine learning, and modern web development.

---

## 📋 Project Overview

This dashboard analyses the relationship between **social media usage patterns** and **student mental health** using a real-world dataset. It provides:

- **25 Interactive Charts** powered by Chart.js
- **10 KPI Metric Cards** with animated counters
- **Random Forest ML Model** predicting mental health scores
- **AI-Generated Insights** from rule-based analysis
- **Advanced Filtering** with 10 filter dimensions
- **Dark / Light Mode** toggle
- **Glassmorphism UI** inspired by Power BI / Tableau

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, Flask 3.1 |
| Data Processing | Pandas 2.2, NumPy 2.2 |
| Machine Learning | Scikit-Learn 1.6 (Random Forest) |
| Excel I/O | OpenPyXL 3.1 |
| Frontend | HTML5, CSS3, Bootstrap 5.3 |
| Charts | Chart.js 4.4 |
| Animations | AOS Library, Custom CSS |
| Icons | Font Awesome 6.5 |
| Fonts | Google Fonts (Inter) |

---

## 📁 Project Structure

```
Social_Media_Mental_Health_Dashboard/
│
├── app.py                     # Flask application & API endpoints
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── dataset/
│   └── Social media dataset.xlsx   # Input data
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Excel file loader with caching
│   ├── preprocessing.py       # Data cleaning & standardisation
│   ├── analysis.py            # 25 chart aggregation functions
│   └── insights.py            # AI insight generation engine
│
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism + Dark/Light styles
│   ├── js/
│   │   └── dashboard.js       # SPA logic, charts, ML form
│   └── images/                # (optional assets)
│
├── templates/
│   └── index.html             # Single-Page Application shell
│
└── reports/                   # (generated report exports)
```

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have **Python 3.13** installed:

```bash
python --version
# Python 3.13.x
```

### 2. Setup

```bash
# Navigate to project directory
cd Social_Media_Mental_Health_Dashboard

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset

Place `Social media dataset.xlsx` inside the `dataset/` folder:

```
dataset/
└── Social media dataset.xlsx
```

### 4. Run

```bash
python app.py
```

The application will:
1. Load and clean the dataset
2. Train the Random Forest model in the background
3. Open your browser automatically at `http://127.0.0.1:5000`

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | KPI cards, mini charts, gauge, quick actions |
| 📈 **Analytics** | All 25 interactive charts organised by category |
| 🤖 **Predictions** | ML form, model metrics, feature importance |
| 👥 **Students** | Filterable data table with pagination & export |
| 💡 **Insights** | AI-generated data insights |
| ℹ️ **About** | Project documentation & setup guide |

---

## 📉 Analytics Charts (25 Total)

| # | Chart | Type |
|---|-------|------|
| 1 | Gender Distribution | Pie |
| 2 | Age Distribution | Histogram |
| 3 | Academic Level Distribution | Bar |
| 4 | Country Distribution (Top 15) | Horizontal Bar |
| 5 | Most Used Platform | Doughnut |
| 6 | Avg Daily Usage by Platform | Bar |
| 7 | Sleep Hours Distribution | Histogram |
| 8 | Mental Health Score Distribution | Line |
| 9 | Addiction Score Distribution | Line |
| 10 | Relationship Status | Pie |
| 11 | Academic Performance Impact | Pie |
| 12 | Conflicts Over Social Media | Stacked Bar |
| 13 | Sleep vs Mental Health | Scatter |
| 14 | Daily Usage vs Addiction Score | Bubble |
| 15 | Daily Usage vs Mental Health | Scatter |
| 16 | Country vs Avg Mental Health | Bar |
| 17 | Gender vs Addiction Score | Grouped Bar |
| 18 | Academic Level vs Daily Usage | Bar |
| 19 | Platform Usage by Gender | Stacked Bar |
| 20 | Correlation Heatmap | Matrix |
| 21 | Mental Health Risk Gauge | Gauge |
| 22 | Platform Popularity Ranking | Ranking List |
| 23 | Top Countries by Addiction Score | Bar |
| 24 | Avg Sleep by Academic Level | Bar |
| 25 | Avg Usage by Relationship Status | Bar |

---

## 🤖 Machine Learning

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Target | Mental Health Score (1–10) |
| Features | 10 (Age, Gender, Country, Usage, Sleep, Platform, Relationship, Academic Level, Conflict, Addiction) |
| Train/Test Split | 80% / 20% |
| n_estimators | 100 |
| random_state | 42 |
| Metrics | RMSE, MAE, R² Score |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve SPA index.html |
| GET | `/api/dashboard` | KPI summary statistics |
| GET | `/api/gender` | Gender & age data |
| GET | `/api/platform` | Platform analysis |
| GET | `/api/country` | Country data |
| GET | `/api/sleep` | Sleep analysis |
| GET | `/api/mental` | Mental health data + heatmap |
| GET | `/api/addiction` | Addiction analysis |
| GET | `/api/academic` | Academic data |
| POST | `/api/predict` | ML prediction |
| GET | `/api/insights` | AI insights |
| GET | `/api/table` | Filtered student records |
| GET | `/api/filters` | Filter dropdown options |
| GET | `/api/ml_info` | Model metrics & feature importance |

---

## 🎨 Design Features

- **Glassmorphism** — frosted glass cards with backdrop blur
- **Gradient Background** — animated radial gradient blobs
- **Dark Mode** (default) + **Light Mode** — persisted in localStorage
- **Responsive** — works on desktop, tablet, and mobile
- **AOS Animations** — elements animate in on scroll
- **Hover Effects** — cards lift and glow on hover
- **Loading Screen** — brain emoji spinner with progress bar
- **Toast Notifications** — non-intrusive feedback messages
- **Counter Animations** — KPI values count up from 0

---

## 📄 Documentation

A full project guide is available in [DOCUMENTATION.md](DOCUMENTATION.md).

## 📄 License

This project is created for academic purposes as a Final Year B.Sc. Computer Science project.

---

## 👨‍💻 Author

**Student Analyst**
B.Sc. Computer Science — Final Year Project 2025–2026

---

*Built with ❤️ using Python, Flask, Pandas, Scikit-Learn, and Chart.js*
