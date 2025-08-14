"""
Breast Cancer Wisconsin ML Project - Main Execution File
======================================================

This is the main execution file that runs the complete machine learning pipeline
for breast cancer diagnosis prediction using the Wisconsin dataset.

Author: Sello Kgole
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import project modules
from data_loader import DataLoader
from eda import EDAAnalyzer
from models import ModelTrainer
from evaluation import ModelEvaluator
from utils import setup_plotting, print_header

def main():
    """Main function to execute the complete ML pipeline"""
    
    print_header("BREAST CANCER WISCONSIN ML PROJECT")
    
    # Setup plotting style
    setup_plotting()
    
    # 1. Data Loading and Preprocessing
    print_header("1. DATA LOADING AND PREPROCESSING")
    
    data_loader = DataLoader()
    df = data_loader.load_data('data/data.csv')  # Update path as needed
    df = data_loader.clean_data(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['diagnosis'].value_counts()}")
    
    # 2. Exploratory Data Analysis
    print_header("2. EXPLORATORY DATA ANALYSIS")
    
    eda = EDAAnalyzer(df)
    eda.plot_target_distribution()
    eda.analyze_features()
    eda.correlation_analysis()
    
    # 3. Data Preparation
    print_header("3. DATA PREPARATION")
    
    X, y, feature_names = data_loader.prepare_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 4. Model Training
    print_header("4. MODEL TRAINING AND COMPARISON")
    
    trainer = ModelTrainer()
    models, results = trainer.train_multiple_models(
        X_train, X_train_scaled, y_train, 
        X_test, X_test_scaled, y_test
    )
    
    # 5. Model Evaluation
    print_header("5. MODEL EVALUATION")
    
    evaluator = ModelEvaluator()
    evaluator.compare_models(results, y_test)
    
    # 6. Hyperparameter Tuning
    print_header("6. HYPERPARAMETER TUNING")
    
    best_model_name = trainer.get_best_model(results)
    print(f"Best model: {best_model_name}")
    
    tuned_model, tuned_results = trainer.tune_hyperparameters(
        best_model_name, X_train, X_train_scaled, y_train,
        X_test, X_test_scaled, y_test
    )
    
    # 7. Feature Importance Analysis
    print_header("7. FEATURE IMPORTANCE ANALYSIS")
    
    evaluator.analyze_feature_importance(
        tuned_model, best_model_name, feature_names
    )
    
    # 8. Final Results Summary
    print_header("8. FINAL RESULTS SUMMARY")
    
    evaluator.final_summary(results, tuned_results, best_model_name, df.shape)
    
    print_header("PROJECT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()