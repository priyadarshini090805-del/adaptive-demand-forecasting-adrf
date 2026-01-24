# Adaptive Demand–Aware Reinforcement Forecasting (ADRF)

## Project Overview
This project implements an advanced demand forecasting and inventory decision system using a novel algorithm called **Adaptive Demand–Aware Reinforcement Forecasting (ADRF)**.

Unlike traditional forecasting models, ADRF integrates demand prediction with business outcome feedback to dynamically adjust future forecasts based on inventory performance.

---

## Key Features
- LSTM-based demand forecasting
- Adaptive correction using reinforcement-style feedback
- Inventory simulation with stockout and overstock penalties
- Baseline comparison and ablation study
- Business-aware performance evaluation

---

## Algorithm Description
ADRF introduces a correction factor that adapts demand forecasts based on observed inventory outcomes such as stockouts or excess stock. This creates a closed-loop learning system that improves decision quality over time.

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


Save the file.

---

## STEP 3 — Commit the README change

Open terminal in the project folder and run:

```bash
git status


