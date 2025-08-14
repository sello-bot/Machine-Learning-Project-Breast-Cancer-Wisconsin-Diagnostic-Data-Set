"""
Data Loader Module
==================

This module handles data loading, cleaning, and preprocessing for the 
Breast Cancer Wisconsin dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def load_data(self, filepath):
        """
        Load the breast cancer dataset from CSV file
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            print("Please ensure the data.csv file is in the correct location.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def clean_data(self, df):
        """
        Clean the dataset by removing unnecessary columns and handling missing values
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if df is None:
            return None
            
        # Remove unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
            print(f"Removed unnamed columns: {unnamed_cols}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")
        
        # Display basic info
        print(f"Cleaned dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def prepare_features_target(self, df):
        """
        Prepare features and target variables
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            tuple: (X, y, feature_names) where X is features, y is encoded target, 
                   and feature_names is list of feature column names
        """
        if df is None:
            return None, None, None
        
        # Encode target variable
        df['diagnosis_encoded'] = self.label_encoder.fit_transform(df['diagnosis'])
        
        # Select numeric features (exclude id and diagnosis columns)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['id', 'diagnosis_encoded']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['diagnosis_encoded']
        
        print(f"Features prepared: {len(feature_cols)} features")
        print(f"Target encoding: 0={self.label_encoder.inverse_transform([0])[0]}, "
              f"1={self.label_encoder.inverse_transform([1])[0]}")
        
        return X, y, feature_cols
    
    def get_feature_groups(self, feature_names):
        """
        Group features by type (mean, se, worst)
        
        Args:
            feature_names (list): List of feature column names
            
        Returns:
            dict: Dictionary with feature groups
        """
        groups = {
            'mean': [],
            'se': [],
            'worst': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            if 'mean' in feature_lower or any(base in feature_lower for base in 
                ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                 'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']):
                groups['mean'].append(feature)
            elif 'se' in feature_lower or 'error' in feature_lower:
                groups['se'].append(feature)
            elif 'worst' in feature_lower:
                groups['worst'].append(feature)
        
        # If no clear groups, divide features into thirds
        if not any(groups.values()):
            n_features = len(feature_names)
            third = n_features // 3
            groups['mean'] = feature_names[:third]
            groups['se'] = feature_names[third:2*third]
            groups['worst'] = feature_names[2*third:]
        
        return groups