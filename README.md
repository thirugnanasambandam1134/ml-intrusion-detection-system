# Machine Learning-Based Intrusion Detection System (IDS)

## Overview

This project implements a **Machine Learning-based Intrusion Detection System (IDS)** designed to classify network traffic as either **Normal** or **Malicious (Attack)**. The system is trained using labeled historical network traffic data and is capable of analyzing new incoming traffic to detect potential cyber threats.

Traditional signature-based detection systems rely on predefined attack patterns and are limited in detecting zero-day or evolving threats. This project adopts an anomaly-based detection approach using supervised machine learning techniques to improve adaptability, scalability, and detection accuracy.

The system simulates a simplified Security Operations Center (SOC) workflow by generating alerts and logging detected malicious traffic for monitoring and analysis.

---

## Problem Statement

Traditional intrusion detection systems rely on predefined signatures of known attacks. While effective against recognized threats, they struggle to identify new, sophisticated, or previously unseen attack patterns.

With increasing network complexity and traffic volume, static rule-based systems are insufficient. Intelligent, data-driven approaches are required to model network behavior and detect anomalies dynamically.

This project addresses this limitation by applying supervised machine learning techniques to learn network behavior patterns and classify traffic based on features extracted from network connections.

The goal is to design and implement a structured, reproducible, and professionally organized machine learning pipeline for intrusion detection.

---

## Dataset

This project uses the **NSL-KDD dataset**, an improved and refined version of the KDD Cup 1999 dataset, designed to address redundancy and imbalance issues in the original dataset.

### Dataset Characteristics

- 41 network traffic features  
- Labeled records (Normal and multiple attack types)  
- Multiple attack categories including:
  - Denial of Service (DoS)  
  - Probe  
  - Remote to Local (R2L)  
  - User to Root (U2R)  

### Feature Categories

- Basic TCP/IP connection features (duration, protocol type, service, flag)  
- Content-based features (failed login attempts, root access attempts)  
- Traffic-based features (connection count, error rates)  
- Host-based features (destination host statistics)  

The dataset enables supervised training of classification models for intrusion detection and evaluation using standardized metrics.

---

## Machine Learning Approach

The project follows a structured and modular machine learning pipeline:

### 1. Data Loading
- Import dataset  
- Assign feature names  
- Validate data integrity  

### 2. Data Preprocessing
- Handle categorical feature encoding  
- Convert multi-class labels to binary classification (Normal vs Attack)  
- Feature scaling and normalization (if required)  

### 3. Model Training
- Apply supervised classification algorithms  
- Train models using labeled network traffic data  

### 4. Model Evaluation
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### 5. Prediction
- Provide new network traffic input  
- Classify as Normal or Attack  

### 6. Logging & Alert Simulation
- Generate SOC-style alerts for detected attacks  
- Store detection logs in the `logs/` directory  

### 7. Model Persistence
- Save trained model using `joblib`  
- Enable reuse without retraining  

---

## Project Structure

```
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
```

---

## Technologies Used

- Python 3  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  
- Git & GitHub  
- Visual Studio Code  

---

## Installation

```bash
git clone https://github.com/thirugnanasambandam1134/ml-intrusion-detection-system.git
cd ml-intrusion-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

To train and run the intrusion detection system:

```bash
python main.py
```

---

## Future Improvements

- Real-time traffic monitoring integration  
- Deep Learning models (LSTM / ANN)  
- REST API deployment using Flask or FastAPI  
- SIEM integration  
- Dashboard visualization for alert monitoring  

---

## License

This project is licensed under the MIT License.
