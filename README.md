# Parkinson's Disease Prediction 

A machine learning project for predicting Parkinson's Disease using voice measurements and advanced audio features. This project implements multiple classification algorithms to achieve high accuracy in early disease detection.

## Overview

Parkinson's Disease is a progressive neurodegenerative disorder that affects movement and speech. Early detection is crucial for effective treatment and management. This project uses machine learning techniques to predict Parkinson's Disease based on voice measurements and biomedical features.

### Key Features:
- **High Accuracy**: Achieved 86% overall accuracy on validation data
- **Multiple Models**: Implementation of Logistic Regression, SVM, and XGBoost
- **Class Imbalance Handling**: Uses RandomOverSampler for balanced training
- **Feature Engineering**: Comprehensive feature selection using Chi-squared test
- **755 Features**: Extensive voice measurements including TQWT (Tunable Q-Factor Wavelet Transform) features

## Dataset

The dataset contains **755 features** from voice recordings of patients, including:

- **Basic Voice Measurements**:
  - PPE (Pitch Period Entropy)
  - DFA (Detrended Fluctuation Analysis)
  - RPDE (Recurrence Period Density Entropy)

- **TQWT Features**: Advanced wavelet transform features (kurtosis values across 36 decomposition levels)
- **Demographic Data**: Gender and patient ID
- **Target Variable**: Binary classification (0 = Healthy, 1 = Parkinson's Disease)

### Dataset Statistics:
- Total samples: 756
- Features: 755
- Classes: Binary (Healthy vs. Parkinson's)
- Class distribution handled using oversampling

## Features

### Audio Signal Processing Features:
1. **Pitch Period Entropy (PPE)**: Measures voice irregularity
2. **Detrended Fluctuation Analysis (DFA)**: Analyzes signal self-similarity
3. **Recurrence Period Density Entropy (RPDE)**: Quantifies voice periodicity
4. **TQWT Kurtosis Values**: Statistical measures from wavelet decomposition (dec_1 through dec_36)

### Machine Learning Pipeline:
- Data preprocessing and normalization
- Feature selection (SelectKBest with Chi-squared test)
- Class imbalance correction (RandomOverSampler)
- Train-test split (80-20)
- Model training and evaluation

## 💻 Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn xgboost imbalanced-learn
pip install tqdm
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/parkinson-disease-prediction.git
cd parkinson-disease-prediction
```

## 🚀 Usage

### 1. Load and Explore Data
```python
import pandas as pd

# Load dataset
df = pd.read_csv('parkinson_disease.csv')
print(df.head())
print(f"Dataset shape: {df.shape}")
```

### 2. Run the Complete Pipeline
```python
# The notebook includes:
# - Data loading and exploration
# - Preprocessing and feature scaling
# - Feature selection
# - Model training (Logistic Regression, SVM, XGBoost)
# - Model evaluation
```

### 3. Make Predictions
```python
# Using the trained model
predictions = models[0].predict(X_val)
```

## 📈 Model Performance

### Validation Results (Logistic Regression):

| Metric | Healthy (Class 0) | Parkinson's (Class 1) |
|--------|-------------------|------------------------|
| **Precision** | 0.77 | 0.89 |
| **Recall** | 0.71 | 0.92 |
| **F1-Score** | 0.74 | 0.91 |
| **Support** | 14 samples | 37 samples |

### Overall Performance:
- **Accuracy**: 86%
- **Macro Average F1-Score**: 0.82
- **Weighted Average F1-Score**: 0.86

### Model Strengths:
  High recall (92%) for detecting Parkinson's Disease
  Strong precision (89%) minimizing false positives
  Balanced performance across both classes
  Effective handling of class imbalance

## Technologies Used

### Core Libraries:
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization

### Machine Learning:
- **Scikit-learn**: 
  - Model implementations (Logistic Regression, SVM)
  - Preprocessing tools (MinMaxScaler, LabelEncoder)
  - Feature selection (SelectKBest, Chi-squared)
  - Model evaluation metrics
- **XGBoost**: Gradient boosting classifier
- **Imbalanced-learn**: RandomOverSampler for class balancing

### Utilities:
- **tqdm**: Progress tracking
- **warnings**: Error suppression for cleaner output


## 🔬 Methodology

### 1. **Data Preprocessing**
   - Load dataset with 755 features
   - Handle missing values (if any)
   - Encode categorical variables (gender)
   - Remove irrelevant features (ID column)

### 2. **Feature Scaling**
   - Apply MinMaxScaler for normalization
   - Scale features to [0, 1] range

### 3. **Feature Selection**
   - Use SelectKBest with Chi-squared test
   - Select most informative features
   - Reduce dimensionality while maintaining predictive power

### 4. **Class Imbalance Correction**
   - Apply RandomOverSampler
   - Balance training data distribution
   - Prevent model bias toward majority class

### 5. **Model Training**
   - Train multiple classifiers:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - XGBoost Classifier
   - Use train-test split (80-20)

### 6. **Model Evaluation**
   - Generate classification reports
   - Calculate precision, recall, F1-score
   - Analyze confusion matrices
   - Compare model performances

## Results

The Logistic Regression model demonstrates excellent performance:

- **High Sensitivity**: 92% recall for Parkinson's detection ensures minimal false negatives
- **Strong Specificity**: 71% recall for healthy classification
- **Balanced Precision**: 89% precision for disease detection reduces false alarms
- **Overall Accuracy**: 86% correct predictions across all samples

### Clinical Significance:
The high recall rate (92%) for Parkinson's Disease detection is particularly important in medical diagnosis, as it minimizes the risk of missing true positive cases, enabling early intervention and treatment.
