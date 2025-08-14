"""
Machine Learning Models Module
==============================

This module handles model training, hyperparameter tuning, and model selection
for the Breast Cancer Wisconsin dataset classification task.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

class ModelTrainer:
    """Handles model training and hyperparameter tuning"""
    
    def __init__(self):
        """Initialize the model trainer with default models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Hyperparameter grids for each model
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    
    def train_single_model(self, model_name, model, X_train, y_train, X_test, y_test, use_scaled=False):
        """
        Train a single model and return predictions and probabilities
        
        Args:
            model_name (str): Name of the model
            model: The sklearn model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            use_scaled (bool): Whether to use scaled features
            
        Returns:
            dict: Dictionary containing model results
        """
        print(f"\nTraining {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_multiple_models(self, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test):
        """
        Train multiple models and compare their performance
        
        Args:
            X_train: Unscaled training features
            X_train_scaled: Scaled training features
            y_train: Training labels
            X_test: Unscaled test features
            X_test_scaled: Scaled test features
            y_test: Test labels
            
        Returns:
            tuple: (models_dict, results_dict)
        """
        models_results = {}
        trained_models = {}
        
        for name, model in self.models.items():
            # Use scaled data for SVM and KNN, unscaled for tree-based models
            if name in ['SVM', 'K-Nearest Neighbors']:
                results = self.train_single_model(
                    name, model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
                )
            else:
                results = self.train_single_model(
                    name, model, X_train, y_train, X_test, y_test, use_scaled=False
                )
            
            models_results[name] = results
            trained_models[name] = model
            
            # Print classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, results['predictions'], 
                                      target_names=['Benign', 'Malignant']))
            print("-" * 60)
        
        return trained_models, models_results
    
    def get_best_model(self, results):
        """
        Get the best performing model based on accuracy
        
        Args:
            results (dict): Results from train_multiple_models
            
        Returns:
            str: Name of the best model
        """
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"\nBest performing model: {best_model}")
        print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
        return best_model
    
    def tune_hyperparameters(self, model_name, X_train, X_train_scaled, y_train, 
                           X_test, X_test_scaled, y_test):
        """
        Perform hyperparameter tuning for the specified model
        
        Args:
            model_name (str): Name of the model to tune
            X_train: Unscaled training features
            X_train_scaled: Scaled training features
            y_train: Training labels
            X_test: Unscaled test features
            X_test_scaled: Scaled test features
            y_test: Test labels
            
        Returns:
            tuple: (best_model, results_dict)
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        print("This may take a few minutes...")
        
        # Get the base model and parameter grid
        base_model = self.models[model_name]
        param_grid = self.param_grids[model_name]
        
        # Create a fresh model instance
        if model_name == 'Logistic Regression':
            fresh_model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'Random Forest':
            fresh_model = RandomForestClassifier(random_state=42)
        elif model_name == 'SVM':
            fresh_model = SVC(random_state=42, probability=True)
        else:  # KNN
            fresh_model = KNeighborsClassifier()
        
        # Perform grid search
        grid_search = GridSearchCV(
            fresh_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        # Use appropriate data (scaled or unscaled)
        if model_name in ['SVM', 'K-Nearest Neighbors']:
            grid_search.fit(X_train_scaled, y_train)
            final_predictions = grid_search.predict(X_test_scaled)
            final_probabilities = grid_search.predict_proba(X_test_scaled)[:, 1]
        else:
            grid_search.fit(X_train, y_train)
            final_predictions = grid_search.predict(X_test)
            final_probabilities = grid_search.predict_proba(X_test)[:, 1]
        
        # Calculate final metrics
        final_accuracy = accuracy_score(y_test, final_predictions)
        final_auc = roc_auc_score(y_test, final_probabilities)
        
        results = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'accuracy': final_accuracy,
            'auc': final_auc,
            'predictions': final_predictions,
            'probabilities': final_probabilities
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Final test accuracy: {final_accuracy:.4f}")
        print(f"Final test AUC: {final_auc:.4f}")
        
        print(f"\nFinal Classification Report:")
        print(classification_report(y_test, final_predictions, 
                                  target_names=['Benign', 'Malignant']))
        
        return grid_search.best_estimator_, results
    
    def predict_new_sample(self, model, scaler, new_sample, model_name):
        """
        Make predictions on new samples
        
        Args:
            model: Trained model
            scaler: Fitted scaler (None if not needed)
            new_sample: New sample to predict
            model_name: Name of the model
            
        Returns:
            tuple: (prediction, probability)
        """
        if model_name in ['SVM', 'K-Nearest Neighbors'] and scaler is not None:
            new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
            prediction = model.predict(new_sample_scaled)[0]
            probability = model.predict_proba(new_sample_scaled)[0]
        else:
            prediction = model.predict(new_sample.reshape(1, -1))[0]
            probability = model.predict_proba(new_sample.reshape(1, -1))[0]
        
        return prediction, probability