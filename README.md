# 🌊 FloodGuard AI

**AI-Powered Flood Risk Prediction and Early Warning System**

FloodGuard AI is a machine learning-based flood risk analysis and decision support system developed for **Samsun, Türkiye**.

The system combines regional flood-related data with real-time weather information to estimate flood risk levels, calculate risk scores, visualize high-risk areas, and provide intervention recommendations.

---

## 🚀 Features

* 🤖 Machine learning-based flood risk classification
* 📊 Numerical flood risk score prediction
* 🌦️ Real-time weather data integration
* ⏱️ 3-hour future risk estimation
* 🗺️ Interactive flood risk map for Samsun
* 📈 Risk distribution and regional analysis charts
* 🧠 Feature importance analysis for model explainability
* 🚨 Identification of critical regions
* 📋 Prioritized intervention list
* 👥 Citizen report impact simulation
* 💡 Automatic explanations and intervention recommendations
* 🖥️ Multi-tab graphical user interface

---

## 🧠 Machine Learning Models

FloodGuard AI uses two machine learning models:

### Random Forest Classifier

Used to classify each region into one of four flood risk levels:

* 🟢 Low
* 🟡 Medium
* 🟠 High
* 🔴 Critical

### Random Forest Regressor

Used to estimate a numerical flood risk score between **0 and 100**.

The system evaluates model performance using:

* Accuracy
* F1-Score
* Mean Absolute Error (MAE)

---

## 📊 Input Features

The machine learning models analyze several regional factors:

| Feature                  | Description                            |
| ------------------------ | -------------------------------------- |
| Rainfall Amount          | Regional rainfall level                |
| Stream Level             | Current stream/water level             |
| Terrain Slope            | Geographic slope of the region         |
| Infrastructure Condition | Infrastructure resilience level        |
| Previous Floods          | Historical flood frequency             |
| Population Density       | Population concentration of the region |

---

## 🌦️ Real-Time Weather Integration

FloodGuard AI retrieves current weather information using the **Open-Meteo API**.

The system uses:

* Current temperature
* Relative humidity
* Current rainfall
* Hourly rainfall forecast

Real-time rainfall information is combined with regional data to support the flood risk analysis.

---

## ⏱️ Future Risk Prediction

In addition to current flood risk, the system estimates the possible risk level for the following **3 hours** using forecast rainfall data.

The application displays:

* Predicted risk level
* Model confidence
* Risk score
* Future risk estimation

---

## 🗺️ Interactive Risk Map

FloodGuard AI includes an interactive map designed for Samsun.

The map provides:

* Regional flood risk visualization
* Heat map layer
* Critical risk markers
* Risk-level filtering
* Multiple map views
* Region-specific information panels

---

## 📈 Explainable AI

The application includes a feature importance visualization that shows how strongly each input variable influences the Random Forest model.

This helps make the model's predictions easier to interpret.

---

## 🚨 Decision Support

FloodGuard AI does more than generate predictions.

Based on the detected risk level, the system also provides suggested actions such as:

* Monitoring high-risk areas
* Inspecting streams and drainage infrastructure
* Preparing emergency response teams
* Prioritizing critical regions
* Supporting evacuation decisions in critical scenarios

> **Note:** FloodGuard AI is an educational and experimental project. It is not intended to replace official disaster warning or emergency management systems.

---

## 🛠️ Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Tkinter
* Requests
* HTML / CSS / JavaScript
* Leaflet
* Open-Meteo API

---

## 📁 Project Structure

```text
FloodGuard-AI/
│
├── main.py
├── sel_verileri.csv
├── samsun_sel_risk_haritasi-5.html
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install the required Python libraries:

```bash
pip install pandas matplotlib requests scikit-learn
```

---

## ▶️ Running the Application

Make sure the Python file, dataset and HTML map are located in the same directory.

Then run:

```bash
python main.py
```

The FloodGuard AI desktop interface will open automatically.

---

## 🔄 System Workflow

```text
Regional Dataset
       ↓
Data Processing
       ↓
Machine Learning Models
       ↓
Real-Time Weather Data
       ↓
Flood Risk Analysis
       ↓
Risk Classification + Risk Score
       ↓
Visualization & Decision Support
```

---

## 🎯 Project Purpose

The main purpose of FloodGuard AI is to explore how **machine learning, real-time environmental data and visualization techniques** can be combined to support flood risk analysis.

The project was also developed as a practical study in data processing, machine learning, API integration and graphical interface development.

---

## 🔮 Future Improvements

Possible future improvements include:

* Larger and verified historical flood datasets
* Additional meteorological variables
* Automated model retraining
* Database integration
* Web-based dashboard
* More advanced time-series forecasting
* Real-time sensor integration
* Improved geographic risk modeling

---

## 👩‍💻 Developer

Developed by **Yaren Alımlı**

Artificial Intelligence and Data Engineering Student
