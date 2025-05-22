# Sleep Quality Forecasting with Wearable Data

## Overview
This repository provides a framework for forecasting next-day sleep quality (Good, Moderate, Poor) using a multivariate time-series data from consumer-grade wearables (e.g., Fitbit(https://www.kaggle.com/datasets/arashnic/fitbit)). It processes merged streams of heart rate, step count, calories burned, METs, and derived stress levels to build machine learning and deep learning models for sleep quality prediction.

## Features
- **Data Aggregation & Labeling**  
  Merges hourly Fitbit CSVs (steps, calories, heart rate, METs, sleep) and computes `StressLevel = HR − Q1(HR)`. Labels next‐day sleep quality via rule‐based thresholds on total sleep time and restless minutes.

- **Deep Learning Models**  
  - **Bidirectional LSTM (BiLSTM):** Captures long-term temporal dependencies across 16-hour windows.  
  - **CNN–BiLSTM Hybrid:** Adds Conv1D layers to learn local temporal patterns and denoise before BiLSTM.
  - 
- **Baseline Classifiers**  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Multi-Layer Perceptron (MLP)  
  - Random Forest  

- **Evaluation & Visualization**  
  Training/validation loss and accuracy curves, precision/recall plots, confusion matrices, and comparative performance tables.

## Requirements
- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- torch  
- torchvision  
- matplotlib  
- seaborn  
- tqdm  


## Workflow
1. **Data Preprocessing**  
   - Merge raw Fitbit CSVs (steps, calories, heart rate, sleep, METs)  
   - Compute stress level and normalize all features  
   - Generate sliding windows of length 16 and label next-day sleep quality  

2. **Model Training**  
   - Train BiLSTM, CNN–BiLSTM, and GRU models in PyTorch  
   - Save best model checkpoints based on validation loss  

3. **Baseline Training**  
   - Fit logistic regression, SVM, MLP, and random forest on processed data  

4. **Evaluation & Visualization**  
   - Plot training curves, confusion matrices, precision/recall  
   - Compile performance summary table  

## Outputs
Processed datasets including labeled sleep quality classes based on wearable data.  
Trained model checkpoints for BiLSTM, BiLSTM–CNN, GRU, and baseline models.  
Evaluation metrics including accuracy, precision, and recall for each model.  
Visualizations of training curves and confusion matrices for performance comparison.


## Usage
Ensure all required dependencies are installed, and provide the preprocessed Fitbit wearable data as specified in the scripts. Follow the outlined steps to generate labeled datasets, train deep learning models, evaluate baseline classifiers, and visualize results. This notebook is designed for health researchers and developers interested in leveraging wearable data for sleep quality prediction using machine learning.

