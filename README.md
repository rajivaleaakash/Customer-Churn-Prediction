# Customer Churn Prediction

This repository provides an end-to-end pipeline for predicting customer churn in the telecom industry using machine learning techniques. The project covers data ingestion, feature engineering, model training, evaluation, and reporting.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Modeling & Evaluation](#modeling--evaluation)
- [Data Ingestion](#data-ingestion)
- [Feature Engineering](#feature-engineering)
- [Reporting](#reporting)
- [License](#license)

---

## Project Overview

Customer churn is a critical business problem for telecom companies. This repository provides a reproducible workflow to predict customer churn using advanced machine learning models and techniques such as feature selection and handling imbalanced data.

The main components include:

- **Data ingestion** from CSV files into a database.
- **Feature engineering** and preprocessing.
- **Model training** with various algorithms (e.g. Random Forest, XGBoost, LightGBM).
- **Feature selection** and importance analysis.
- **Model evaluation** and reporting.

---

## Features

- Automated data ingestion from local CSV files.
- Feature engineering (e.g. creation of high-risk payment features, customer profiles).
- Handling of imbalanced data using SMOTE.
- Model training and selection, including saving the best model.
- Comprehensive reporting of training process and results.
- Modular codebase for easy customization and extension.

---

## Directory Structure

```
Customer-Churn-Prediction/
│
├── data/                  # Raw data CSV files
├── models/                # Saved models
├── features.py            # Feature engineering, selection, importance
├── ingestion_db.py        # Data ingestion into database
├── train.py               # Model training pipeline
├── requirement.txt        # Python dependencies
├── README.md              # Project documentation
└── ...                    # Supporting scripts and files
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rajivaleaakash/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install dependencies:**

   It is recommended to use a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirement.txt
   ```

---

## Usage

### 1. Data Ingestion

Place your raw CSV data files in the `data/` directory.

Run the ingestion script to load data into the database:

```bash
python ingestion_db.py
```

### 2. Feature Engineering & Preprocessing

Feature engineering is performed automatically during training, but you can run feature extraction via:

```bash
python features.py
```

### 3. Model Training

Train multiple models and generate a report:

```bash
python train.py
```

This script will:
- Load and preprocess data
- Perform feature selection
- Handle class imbalance using SMOTE (configurable)
- Train models (Random Forest, XGBoost, LightGBM, etc.)
- Select and save the best performing model
- Save all trained models to the `models/` directory
- Generate a training report

---

## Requirements

Main dependencies (see `requirement.txt` for full list):

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- matplotlib, seaborn
- sqlalchemy

---

## Modeling & Evaluation

- **Model Selection:** Multiple classifiers are trained; best model is selected based on ROC AUC.
- **Feature Selection:** Top features are selected and their importances are reported.
- **Imbalanced Data:** SMOTE is used to address class imbalance.
- **Reporting:** Training summary and details are logged and saved.

---

## Data Ingestion

- CSV files in `data/` folder are parsed and loaded into a database.
- Data is cleaned, validated, and ingested with robust error handling and logging.

---

## Feature Engineering

- Custom features such as charges per tenure, high monthly charges, high-risk payment methods, single/family customer profiles.
- Automated preprocessing and feature selection for improved model accuracy.

---

## Reporting

- All steps in the pipeline are logged.
- Model performance metrics and feature importance are reported.
- Output files (models, reports) are saved for further analysis.

---

## License

This project is released under the MIT License.

---

## Author

- [rajivaleaakash](https://github.com/rajivaleaakash)
