# Adaptive Demand–Aware Reinforcement Forecasting (ADRF)

## Project Overview
This project implements an IIT-level demand forecasting and inventory decision system using a novel algorithm called **Adaptive Demand–Aware Reinforcement Forecasting (ADRF)**.

Unlike traditional forecasting models, ADRF integrates demand prediction with business outcome feedback to dynamically correct future forecasts.

---

## Key Features
- LSTM-based demand forecasting
- Reinforcement-style adaptive correction
- Inventory simulation with stockout and overstock penalties
- Ablation study and baseline comparison
- Business-aware performance evaluation

---

## Algorithm: ADRF (Proposed)
ADRF introduces a correction factor that adapts forecasts based on inventory outcomes such as stockouts and excess stock. This creates a closed-loop learning system that minimizes cumulative business loss.

---

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
python main.py
