"""
Exploratory Data Analysis Module
================================

This module handles all exploratory data analysis and visualization
for the Breast Cancer Wisconsin dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

class EDAAnalyzer:
    """Handles exploratory data analysis and visualization"""
    
    def __init__(self, df):
        """
        Initialize EDA analyzer with dataset
        
        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        self.df = df
        self.data_loader = DataLoader()
    
    def plot_target_distribution(self):
        """Plot the distribution of the target variable (diagnosis)"""
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        counts = self.df['diagnosis'].value_counts()
        bars = plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'], edgecolor='black')
        plt.title('Distribution of Diagnosis', fontsize=14, fontweight='bold')
        plt.xlabel('Diagnosis', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        labels = ['Benign (B)', 'Malignant (M)']
        colors = ['skyblue', 'salmon']
        wedges, texts, autotexts = plt.pie(counts.values, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        plt.title('Diagnosis Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Dataset contains {counts['B']} benign and {counts['M']} malignant cases")
        print(f"Class balance: {counts['B']/(counts['B']+counts['M'])*100:.1f}% benign, "
              f"{counts['M']/(counts['B']+counts['M'])*100:.1f}% malignant")
    
    def analyze_features(self):
        """Analyze and visualize feature distributions by diagnosis"""
        # Get numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id'] if 'id' in numeric_features else []
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # Group features if possible
        feature_groups = self.data_loader.get_feature_groups(feature_cols)
        
        # Plot mean features (or first 10 features)
        plot_features = feature_groups['mean'] if feature_groups['mean'] else feature_cols[:10]
        
        if len(plot_features) > 0:
            n_features = len(plot_features)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(20, n_rows * 4))
            
            for i, feature in enumerate(plot_features):
                plt.subplot(n_rows, n_cols, i + 1)
                
                # Create box plot
                self.df.boxplot(column=feature, by='diagnosis', ax=plt.gca())
                plt.title(f'{feature} by Diagnosis')
                plt.xlabel('Diagnosis')
                plt.ylabel(feature)
                
                # Remove the automatic title from pandas
                plt.suptitle('')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Analyzed {len(plot_features)} key features")
    
    def correlation_analysis(self):
        """Perform and visualize correlation analysis"""
        # Get numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id'] if 'id' in numeric_features else []
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            print("No numeric features found for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.df[feature_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(15, 12))
        
        # Use mask for better visualization if many features
        if len(feature_cols) > 20:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        else:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
                print(f"  {feat1} <-> {feat2}: r = {corr:.3f}")
        else:
            print("No highly correlated feature pairs found (|r| > 0.8)")
    
    def feature_statistics(self):
        """Generate descriptive statistics for features by diagnosis"""
        # Get numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id'] if 'id' in numeric_features else []
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            print("No numeric features found for statistical analysis")
            return
        
        print("Feature Statistics by Diagnosis:")
        print("=" * 50)
        
        for diagnosis in self.df['diagnosis'].unique():
            print(f"\n{diagnosis} Cases:")
            subset = self.df[self.df['diagnosis'] == diagnosis][feature_cols]
            print(subset.describe().round(3))
    
    def plot_feature_distributions(self, features_to_plot=None):
        """
        Plot distribution of specified features
        
        Args:
            features_to_plot (list): List of features to plot. If None, plots first 6 numeric features
        """
        # Get numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id'] if 'id' in numeric_features else []
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        if features_to_plot is None:
            features_to_plot = feature_cols[:6]  # First 6 features
        
        if len(features_to_plot) == 0:
            print("No features to plot")
            return
        
        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 4))
        
        for i, feature in enumerate(features_to_plot):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Plot histograms for each diagnosis
            for diagnosis in self.df['diagnosis'].unique():
                subset = self.df[self.df['diagnosis'] == diagnosis][feature]
                plt.hist(subset, alpha=0.7, label=f'{diagnosis}', bins=20)
            
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()