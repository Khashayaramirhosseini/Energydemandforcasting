#!/usr/bin/env bash
set -e
python data_preprocessing.py
python feature_engineering.py
python model_tuning.py
python train_and_save.py
python evaluate_and_plot.py
