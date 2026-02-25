# Machine Learning-Based Intrusion Detection System (IDS)

## Overview

This project implements a Machine Learning-based Intrusion Detection System (IDS) designed to classify network traffic as either **Normal** or **Malicious (Attack)**. The system is trained using labeled historical network traffic data and is capable of analyzing new incoming traffic to detect potential cyber threats.

Traditional signature-based detection systems struggle to detect unknown or zero-day attacks. This project demonstrates an anomaly-based detection approach using supervised machine learning techniques to improve adaptability and detection performance.

The project simulates a simplified Security Operations Center (SOC) workflow by generating alerts and logging detected malicious traffic.

## Problem Statement

Traditional intrusion detection systems rely on predefined signatures of known attacks. While effective against recognized threats, they fail when encountering new or evolving attack patterns.

This project addresses that limitation by applying supervised machine learning techniques to learn network behavior patterns and classify traffic based on features extracted from network connections.

The goal is to build a structured, reproducible, and professionally organized ML pipeline for intrusion detection.

## Dataset

This project uses the **NSL-KDD dataset**, an improved version of the KDD Cup 1999 dataset.

Dataset characteristics:

- 41 network traffic features
- Labeled records (Normal and multiple attack types)
- Includes attack categories such as:
  - Denial of Service (DoS)
  - Probe
  - Remote to Local (R2L)
  - User to Root (U2R)

Feature types include:

- Basic TCP/IP connection features (duration, protocol type, service, flag)
- Content-based features (failed logins, root access attempts)
- Traffic-based features (connection counts, error rates)
- Host-based features (destination host statistics)

The dataset enables supervised training of classification models for intrusion detection.

## Machine Learning Approach

The project follows a structured ML pipeline:

1. Data Loading  
   - Import dataset  
   - Assign feature names  

2. Data Preprocessing  
   - Encode categorical features  
   - Convert labels to binary (Normal vs Attack)  
   - Feature scaling (if required)  

3. Model Training  
   - Apply supervised classification algorithms  
   - Train using labeled traffic data  

4. Model Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion matrix  

5. Prediction  
   - Provide fresh network traffic data  
   - Classify as Normal or Attack  

6. Logging  
   - Generate simulated SOC alerts  
   - Store detection logs in the logs directory  

7. Model Saving  
   - Save trained model using joblib  
   - Enable reuse without retraining  


## Project Structure

ml-intrusion-detection-system/
├── data/
│   ├── raw/                         # Original dataset files (NSL-KDD)
│   └── processed/                   # Cleaned and preprocessed data
│
├── notebooks/                       # Research and experimentation
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/                             # Core production source code
│   ├── data_loader.py               # Dataset loading functions
│   ├── preprocess.py                # Data cleaning & encoding
│   ├── train_model.py               # Model training pipeline
│   ├── predict.py                   # Prediction logic
│   ├── logging_system.py            # SOC-style alert logging
│   └── config.py                    # Configuration & constants
│
├── models/                          # Saved trained models
│   └── trained_model.pkl
│
├── logs/                            # Generated intrusion alert logs
│   └── ids_alert_logs.csv
│
├── tests/                           # Unit tests (optional but recommended)
│
├── main.py                          # Application entry point
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Ignored files & folders
├── LICENSE                          # MIT License
└── README.md                        # Project documentation

## Technologies Used

- Python 3  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  
- Git & GitHub  
- VS Code  