# Machine-Learning-Project-Breast-Cancer-Wisconsin-Diagnostic-Data-Set
Predict whether the cancer is benign or malignant

# 🔬 Breast Cancer Wisconsin Diagnostic Dataset Classification 🎯

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 📋 Project Overview

This machine learning project focuses on **breast cancer classification** using the famous Wisconsin Diagnostic Breast Cancer dataset. The goal is to predict whether a breast mass is **malignant (M)** or **benign (B)** based on features computed from digitized images of fine needle aspirate (FNA) of breast masses.

## 🎯 Objective

Develop and compare various machine learning models to accurately classify breast cancer diagnoses, helping in early detection and medical decision-making.

## 📊 Dataset Information

### 📈 Dataset Overview
- **Source**: UCI Machine Learning Repository
- **Instances**: 569 samples
- **Features**: 30 real-valued features + ID + Diagnosis
- **Target Classes**: 
  - 🔴 **Malignant (M)**: Cancerous
  - 🟢 **Benign (B)**: Non-cancerous

### 🔬 Feature Description

Features are computed from digitized FNA images describing cell nuclei characteristics:

#### Core Measurements (10 features):
1. **Radius** 📏 - Mean distances from center to perimeter points
2. **Texture** 🌐 - Standard deviation of gray-scale values  
3. **Perimeter** ⭕ - Cell nucleus perimeter
4. **Area** 📐 - Cell nucleus area
5. **Smoothness** 〰️ - Local variation in radius lengths
6. **Compactness** 📦 - perimeter² / area - 1.0
7. **Concavity** 🌙 - Severity of concave portions
8. **Concave Points** 📍 - Number of concave portions
9. **Symmetry** ⚖️ - Cell symmetry measure
10. **Fractal Dimension** 🔢 - "Coastline approximation" - 1

#### Statistical Variations:
For each core measurement, three statistical values are computed:
- **Mean** values (features 3-12)
- **Standard Error** (features 13-22) 
- **Worst/Largest** values (features 23-32)

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
pip install -r requirements.txt
```

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

### 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset** (if using local data)
   - Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
   - Or use Kaggle: `/kaggle/input/breast-cancer-wisconsin-data`

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

## 🛠️ Usage

### For Jupyter Notebook/Kaggle:
```python
# Run the main notebook
jupyter notebook main.ipynb
```

### For Python Scripts:
```python
# Run the complete pipeline
python src/main.py
```

### For VS Code:
1. Open the project folder in VS Code
2. Install Python extension
3. Run `main.ipynb` or individual Python files

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

3. **Additional Access**: UW CS ftp server: `ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/WDBC/`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- University of Wisconsin for the original research
- Scikit-learn community for excellent ML tools
- Kaggle for hosting the dataset

## 📬 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repository if you found it helpful!** ⭐

🔬 **Happy Learning and Coding!** 🚀
