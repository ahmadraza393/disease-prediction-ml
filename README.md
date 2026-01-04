# Machine Learning–Based Disease Prediction System

This project demonstrates a basic machine learning workflow for disease prediction using healthcare data.

## Overview
The system includes data preprocessing, feature selection, training of multiple supervised learning models, and performance evaluation.

## Dataset
The project works with publicly available healthcare datasets (e.g., from Kaggle).  
Expected file location:

data/raw/healthcare_dataset.csv

Recommended dataset link:  
[Heart Disease UCI Dataset](https://www.kaggle.com/datasets/uciml/heart-disease-uci)

## Models
- Logistic Regression
- Random Forest
- Support Vector Machine

## Technologies
- Python
- Scikit-learn
- Pandas
- NumPy

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Place the dataset in:

data/raw/healthcare_dataset.csv

3. Run the project:

python main.py

## Folder Structure

data/
    raw/
    processed/
models/
notebooks/
results/
src/
    data_preprocessing.py
    feature_selection.py
    model_training.py
    evaluation.py
main.py
requirements.txt
README.md
