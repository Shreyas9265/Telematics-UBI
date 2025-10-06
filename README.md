# 🚗 Telematics UBI — Driver Dashboard & API

This project is a **local-first implementation** of a telematics-based **Usage-Based Insurance (UBI)** system that leverages real or simulated driving behavior data to predict driver risk and compute personalized insurance premiums.

---

## 🧭 Overview

Traditional insurance pricing relies on generalized factors like age or vehicle type, which fail to reflect true driving behavior.  
This project introduces a **data-driven approach** using telematics to enable fairer and behavior-based insurance models.

### 🎯 Key Objectives
- Ingest and process telematics event data (speed, acceleration, braking, etc.)
- Engineer behavior-based driver features
- Train ML models to predict **incident probability** and derive **risk scores**
- Compute **usage-based premiums** dynamically
- Provide a real-time **interactive dashboard** built with Streamlit and FastAPI

---

## 📦 Repository Structure

```
Telematics-UBI/
│
├── src/
│   ├── api.py              # FastAPI backend (risk scoring + pricing)
│   ├── dashboard.py        # Streamlit dashboard
│   ├── model.py            # Training / inference logic
│   └── utils/              # Helper functions (if any)
│
├── data/
│   ├── sample/features.csv # Aggregated driver-day features
│   ├── sample/events.json  # Synthetic telematics events
│   └── ingested_events.csv # Processed dataset
│
├── models/
│   └── risk_model.joblib   # Trained ML model
│
├── docs/
│   └── dashboard_screenshot.png  # Example visualization
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Shreyas9265/Telematics-UBI.git
cd Telematics-UBI
```

### 2️⃣ Create and Activate a Virtual Environment
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn joblib requests pyyaml
```

---

## ▶️ Run the Application

### Start the FastAPI Backend
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### Start the Streamlit Dashboard
In a new terminal (keep the API running):
```bash
streamlit run src/dashboard.py
```

### Access in Browser
- **Dashboard:** [http://127.0.0.1:8502](http://127.0.0.1:8502)  
- **API Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧠 API Endpoints

| Method | Endpoint | Description |
|:--------|:----------|:-------------|
| **POST** | `/ingest` | Ingest a single telematics event |
| **GET** | `/driver/{driver_id}/score` | Retrieve driver risk score and incident probability |
| **POST** | `/price` | Compute dynamic premium based on behavior |
| **GET** | `/health` | API health status |

**Example Request**
```bash
curl -X POST "http://127.0.0.1:8000/price"      -H "Content-Type: application/json"      -d '{"driver_id": "D_000", "base_premium": 120}'
```

**Expected Response**
```json
{
  "driver_id": "D_000",
  "risk_score": 54,
  "incident_probability": 0.1439,
  "estimated_premium": 123.36
}
```

---

## 📊 Streamlit Dashboard

### Tabs Overview
- **Overview** → Risk score, probability, and premium comparison  
- **Events** → View raw telematics data  
- **Features** → Display engineered driver metrics  
- **Behavior** → Trend analysis of driving KPIs  
- **Suggestions** → Personalized improvement insights  

### Example Output
```
Driver ID: D_000
Risk Score: 54
Incident Probability: 14.39%
Estimated Premium: $123.36
```

Access Dashboard: [http://127.0.0.1:8502](http://127.0.0.1:8502)

---

## 📈 Model Information

- Models are stored in `/models` (e.g., `risk_model.joblib`)
- Trained using **scikit-learn** (Logistic Regression / Gradient Boosting)
- Produces a **calibrated incident probability (0–1)** mapped to a **risk score (0–100)**

### Retraining Example
```bash
python src/train.py --input data/ingested.csv --output models/risk_model.joblib
```

### Model Techniques
- Logistic Regression or Gradient Boosting
- Class imbalance handling using weighted loss
- Probability calibration using Platt or Isotonic scaling

---

## 🧾 Data Details

- Uses **synthetic** or **anonymized** telematics datasets.
- Key columns:
  ```
  driver_id, avg_speed, speed_std, hard_brakes_per_100km, 
  hard_accels_per_100km, phone_use_min_per_100km, 
  night_km_share, rush_km_share, urban_km_share
  ```
- Replace sample CSV/JSON files with real anonymized data as needed.

---

## 🧪 Evaluation Steps

1. **Health Check**
   - Visit `http://127.0.0.1:8000/health`
   - Expected: `{"status":"ok"}`

2. **Data Validation**
   - Open dashboard → check if driver data loads properly

3. **Risk Scoring**
   - Select a driver and click **Generate/Score**
   - Verify the computed score and probability

4. **Premium Calculation**
   - Compare base premium vs adjusted UBI premium
   - Validate results reflect driver behavior

---

## 🌐 External Services

- None required  
- Entire system runs **locally**
- Dependencies:
  - **FastAPI** — Backend API
  - **Streamlit** — Dashboard UI
  - **scikit-learn / joblib** — Model handling
  - **pandas / numpy** — Data processing

---

## 📸 Example Dashboard Screenshot
<img width="1816" height="954" alt="image" src="https://github.com/user-attachments/assets/9b0de009-9ddc-4916-89b6-6cd2b034ea68" />

<img width="1872" height="958" alt="image" src="https://github.com/user-attachments/assets/82860422-a4ed-46a3-a879-b8b1725e2ea9" />

<img width="1856" height="882" alt="image" src="https://github.com/user-attachments/assets/8b03c0be-dc9b-40a8-b1db-7ac86d12c97c" />

<img width="1828" height="934" alt="image" src="https://github.com/user-attachments/assets/3f8e8bd5-e5da-49c6-aa73-eaec1fff8cdd" />

<img width="1894" height="520" alt="image" src="https://github.com/user-attachments/assets/1bbb0785-8268-43ec-aa82-304e636930f2" />

<img width="1868" height="906" alt="image" src="https://github.com/user-attachments/assets/9d51c40f-d3c3-49e7-98c6-359c46794ed5" />



---

## 👨‍💻 Author

**Name:** Shreyas Peddireddy  
**Email:** peddireddy.shreyas@gmail.com  
**University:** University at Buffalo


---

## 🏁 Summary

This project illustrates how **AI and telematics** can enable personalized, fair, and transparent auto insurance pricing.  
By integrating machine learning, real-time data processing, and interactive visualization, it demonstrates the power of data-driven decision-making in the insurance industry.

> 🚘 A practical proof of concept that bridges telematics data and predictive modeling to achieve dynamic, behavior-based insurance pricing.

---

**End of README**
