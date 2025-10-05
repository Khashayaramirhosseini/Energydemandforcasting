# Energy demand forcasting
Machine learning project forecasting electricity and heat demand for 8 residential houses using Random Forest, XGBoost, and CNN-GRU models. Features engineered from temperature, time, and lag data to improve smart energy management accuracy within the Smart Energy Twin framework.
🌍 Smart Energy Twin — Machine Learning–Driven Energy Forecasting for Residential Buildings

This repository contains the code, datasets, and documentation for the Smart Energy Twin project — a research initiative under the Master Engineering Systems program at HAN University of Applied Sciences.
The project focuses on forecasting electricity consumption and heat demand for eight residential households using advanced machine learning and neural network techniques.

🔍 Project Overview

The goal is to develop reliable, interpretable, and efficient models to predict short-term (hourly/daily) energy usage, improving sustainability and enabling smarter energy management in residential systems.

⚙️ Key Features

Electricity Forecasting:

Random Forest Regressor

XGBoost Regressor (with lag & rolling features)

CNN–GRU hybrid model for sequential learning

Heat Demand Forecasting:

XGBoost models for hourly and daily predictions

Neural network ensembles for pattern recognition

Feature engineering with lag, rolling averages, and temperature interactions

Performance Highlights:

Electricity model (XGBoost): RMSE ≈ 1769 W, R² ≈ 0.86

Heat demand (Daily model): RMSE ≈ 0.39 kW, R² ≈ 0.87

📊 Data

Sensor data collected from 8 residential households in Wolfheze, Netherlands.

Features include temperature, time-of-day cycles, and energy meter readings (minutely to daily).

🧠 Tech Stack

Languages: Python 3.12

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Matplotlib, Seaborn

APIs: Open-Meteo for weather data integration

🧩 Methodology

Data cleaning, normalization, and anomaly correction

Cyclical and statistical feature engineering

Model training with cross-validation and GridSearchCV

Evaluation using RMSE, MAE, and R² metrics

Visualization and feature-importance analysis

🧾 Results Summary

Ensemble and boosting models outperform baselines.

Lag and rolling features significantly improve forecasting accuracy.

Neural models capture temporal dependencies but require more data to generalize.

👥 Team

Melika Mirmohammad

Pavel Petrovski

Khashayar Amir Hosseini

Mohammad Eghbali Ghahyazi

Supervised by Aishwarya Aswal (HAN University)
Client: Trung Nguyen

📄 License

MIT License (or specify your chosen license)

🏁 Citation

If you use this work, please cite:

Amir Hosseini K., Mirmohammad M., Petrovski P., Eghbali G., Smart Energy Twin: Machine Learning Forecasting of Residential Heat and Power Demand, HAN University of Applied Sciences, 2025.
