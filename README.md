# 🔬 Breast Cancer Wisconsin Diagnostic Dataset Classification 🎯
Predict whether the cancer is benign or malignant

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 📋 Project Overview

This machine learning project focuses on **breast cancer classification** using the famous Wisconsin Diagnostic Breast Cancer dataset. The goal is to predict whether a breast mass is **malignant (M)** or **benign (B)** based on features computed from digitized images of fine needle aspirate (FNA) of breast masses.

## 🎯 Objective

Develop and compare various machine learning models to accurately classify breast cancer diagnoses, helping in early detection and medical decision-making.

### Key Features

- **Complete ML Pipeline**: From data loading to model deployment
- **Multiple Algorithms**: Logistic Regression, Random Forest, SVM, and K-Nearest Neighbors
- **Comprehensive Analysis**: EDA, correlation analysis, and feature importance
- **Hyperparameter Tuning**: GridSearchCV optimization for best performance
- **Rich Visualizations**: ROC curves, confusion matrices, and feature plots
- **Modular Design**: Clean, maintainable code structure

## 📊 Dataset Information

The Breast Cancer Wisconsin (Diagnostic) Dataset contains features computed from digitized images of breast mass cell nuclei. 

### Features (30 total)
Ten real-valued features are computed for each cell nucleus:

1. **radius** - mean of distances from center to points on the perimeter
2. **texture** - standard deviation of gray-scale values
3. **perimeter** - perimeter of the nucleus
4. **area** - area of the nucleus
5. **smoothness** - local variation in radius lengths
6. **compactness** - perimeter² / area - 1.0
7. **concavity** - severity of concave portions of the contour
8. **concave points** - number of concave portions of the contour
9. **symmetry** - symmetry of the nucleus
10. **fractal dimension** - "coastline approximation" - 1

For each feature, three values are computed:
- **Mean** - average value
- **Standard Error** - standard error of the mean
- **Worst** - mean of the three largest values

### Target Variable
- **Diagnosis**: M = Malignant, B = Benign
### 📁 Project Structure
```
breast-cancer-classification/
│
├── 📄 README.md
├── 📋 requirements.txt
├── 📓 main.ipynb
├── 🐍 src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
├── 📊 data/
│   └── breast-cancer-wisconsin-data.csv
├── 📈 models/
├── 📋 results/
└── 🖼️ plots/

```

 

## 🔧 Usage

### Running Individual Components

```python
# Data loading and cleaning
from data_loader import DataLoader
loader = DataLoader()
df = loader.load_data('data/data.csv')
df_clean = loader.clean_data(df)

# Exploratory data analysis
from eda import EDAAnalyzer
eda = EDAAnalyzer(df_clean)
eda.plot_target_distribution()
eda.analyze_features()

# Model training
from models import ModelTrainer
trainer = ModelTrainer()
models, results = trainer.train_multiple_models(X_train, X_train_scaled, y_train, 
                                               X_test, X_test_scaled, y_test)

# Model evaluation
from evaluation import ModelEvaluator
evaluator = ModelEvaluator()
evaluator.compare_models(results, y_test)
```

### Customizing the Pipeline

#### Adding New Models
```python
# In models.py, add to the ModelTrainer class
self.models['Your Model'] = YourModelClass(parameters)
self.param_grids['Your Model'] = {
    'param1': [value1, value2],
    'param2': [value3, value4]
}
```

#### Modifying Visualizations
```python
# In eda.py, customize plots
def your_custom_plot(self):
    plt.figure(figsize=(12, 8))
    # Your plotting code here
    plt.show()
```

## 📈 Model Performance

The project compares four machine learning algorithms:

| Model | Typical Accuracy | AUC Score | Best Use Case |
|-------|-----------------|-----------|---------------|
| **Logistic Regression** | 95-97% | 0.98+ | Interpretable baseline |
| **Random Forest** | 96-98% | 0.99+ | Feature importance analysis |
| **SVM** | 96-98% | 0.98+ | High-dimensional data |
| **K-Nearest Neighbors** | 94-96% | 0.97+ | Non-linear relationships |

### Key Results
- All models achieve >95% accuracy on this dataset
- Random Forest typically provides the best performance
- Feature scaling is crucial for SVM and KNN
- The dataset is well-suited for machine learning classification

## 📊 Generated Outputs

The project generates several types of outputs:

### Visualizations
- Target variable distribution (bar chart and pie chart)
- Feature distribution by diagnosis (box plots)
- Correlation heatmap
- ROC curves comparison
- Confusion matrices
- Feature importance plots

### Analysis Reports
- Model performance comparison
- Classification reports with precision/recall
- Feature importance rankings
- Hyperparameter tuning results
- Final comprehensive summary

### Files Created
- Model performance CSV files
- Trained model objects (pickle files)
- High-resolution plot images
- Detailed logs and reports

## 🛠️ Configuration Options

### Data Loading
```python
# In data_loader.py
def load_data(self, filepath, **kwargs):
    # Customize data loading parameters
    df = pd.read_csv(filepath, **kwargs)
    return df
```

### Model Parameters
```python
# In models.py - Modify hyperparameter grids
self.param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],     # Adjust range
        'max_depth': [5, 10, 15, None],     # Add more options
        # Add more parameters
    }
}
```

### Visualization Settings
```python
# In utils.py - Modify plotting defaults
plt.rcParams['figure.figsize'] = (12, 8)   # Change default size
plt.rcParams['font.size'] = 14              # Change font size
```

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found Error**
   ```
   Error: File not found at data/data.csv
   ```
   **Solution**: Ensure your data file is in the correct location and named properly.

2. **Missing Packages**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce dataset size or use data sampling for large datasets.

4. **Plotting Issues**
   ```
   Backend error or plots not showing
   ```
   **Solution**: Try different matplotlib backends:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For server environments
   matplotlib.use('TkAgg')  # For desktop environments
   ```

### Performance Optimization

- **Large Datasets**: Use data sampling for initial analysis
- **Slow Training**: Reduce hyperparameter grid size
- **Memory Usage**: Process data in chunks for very large datasets

## 📚 Educational Value

### Machine Learning Concepts
- **Classification algorithms** - Compare different approaches
- **Feature engineering** - Handle real-world medical data
- **Model evaluation** - Comprehensive metrics and validation
- **Hyperparameter tuning** - Optimization techniques

### Data Science Skills
- **Exploratory Data Analysis** - Pattern discovery in medical data
- **Data visualization** - Effective communication of results
- **Statistical analysis** - Understanding feature relationships
- **Model interpretation** - Feature importance and coefficients

### Programming Best Practices
- **Modular design** - Separation of concerns
- **Error handling** - Robust code development
- **Documentation** - Clear code and user guides
- **Version control** - Professional development workflow


## 📊 Sample Results

Expected output when running the complete pipeline:

```
================================================================================
                        BREAST Cancer Wisconsin

================================================================================
                           1. DATA LOADING AND PREPROCESSING                           
================================================================================
Data loaded successfully: (569, 33)
Removed unnamed columns: ['Unnamed: 32']
No missing values found.
Cleaned dataset shape: (569, 32)

================================================================================
                          2. EXPLORATORY DATA ANALYSIS                          
================================================================================
Dataset contains 357 benign and 212 malignant cases
Class balance: 62.7% benign, 37.3% malignant

================================================================================
                              4. MODEL TRAINING AND COMPARISON                              
================================================================================

Training Logistic Regression...
  Accuracy: 0.9649
  AUC Score: 0.9934
  CV Score: 0.9647 (+/- 0.0214)

Training Random Forest...
  Accuracy: 0.9649
  AUC Score: 0.9960
  CV Score: 0.9626 (+/- 0.0221)

Training SVM...
  Accuracy: 0.9649
  AUC Score: 0.9881
  CV Score: 0.9691 (+/- 0.0203)

Training K-Nearest Neighbors...
  Accuracy: 0.9298
  AUC Score: 0.9881
  CV Score: 0.9231 (+/- 0.0348)

================================================================================
                             5. MODEL EVALUATION                             
================================================================================

Model Performance Summary:
================================================================================
Model                Accuracy   AUC        CV Score  
--------------------------------------------------------------------------------
Logistic Regression  0.9649     0.9934     0.9647    
Random Forest        0.9649     0.9960     0.9626    
SVM                  0.9649     0.9881     0.9691    
K-Nearest Neighbors  0.9298     0.9881     0.9231    
================================================================================

Best performing model: Random Forest
Best accuracy: 0.9649
```
## 🤖 Machine Learning Models

### Models Implemented:
- 🌳 **Random Forest**
- 🎯 **Support Vector Machine (SVM)**
- 🧠 **Logistic Regression**
- 🔄 **K-Nearest Neighbors (KNN)**
- 🎪 **Gradient Boosting**
- 🧬 **Neural Network**

### 📊 Performance Metrics:
- ✅ **Accuracy**
- 🎯 **Precision** 
- 📈 **Recall**
- ⚖️ **F1-Score**
- 📋 **Confusion Matrix**
- 📈 **ROC-AUC**

## 📈 Results

Expected model performance:
- 🎯 **Accuracy**: > 95%
- 🔍 **Precision**: > 94%
- 📊 **Recall**: > 96%
- ⚖️ **F1-Score**: > 95%

## 🎨 Visualizations

The project includes:
- 📊 **Data distribution plots**
- 🔥 **Correlation heatmaps** 
- 📈 **Feature importance charts**
- 🎯 **ROC curves**
- 📋 **Confusion matrices**
- 📈 **Learning curves**

## 📚 References

1. **Original Research**: K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34.

2. **Dataset Source**: UCI Machine Learning Repository
   - URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sello Kgole** - *Initial work* - [sello-bot](https://github.com/sello-bot)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- University of Wisconsin for the original research
- Scikit-learn community for excellent ML tools
- Kaggle for hosting the dataset

## 📬 Contact

- GitHub: [@sello-bot](https://github.com/yourusername)
- Email: skgole6@gmail.com
- LinkedIn: [sello-kgole](https://www.linkedin.com/in/sello-kgole-ba450a295/)

---

⭐ **Star this repository if you found it helpful!** ⭐

🔬 **Happy Learning and Coding!** 🚀





