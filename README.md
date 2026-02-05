Parkinson’s Disease Prediction Using Machine Learning
Overview

This project investigates the application of supervised machine learning techniques for predicting Parkinson’s disease using structured biomedical features. The dataset contains multiple observations per patient, requiring aggregation and careful preprocessing to ensure patient-level consistency. The study focuses on principled data preparation, feature reduction, class imbalance handling, and comparative model evaluation rather than aggressive performance optimisation.

The implementation is designed to be reproducible and methodologically transparent, providing a solid baseline for further research-oriented extensions in healthcare analytics.

Project Highlights

Problem Type: Binary classification (Parkinson’s disease prediction)

Models Evaluated: Logistic Regression, XGBoost, Support Vector Classifier

Feature Reduction: Correlation filtering + Chi-square feature selection

Class Imbalance Handling: Random oversampling (training data only)

Evaluation Metrics: ROC-AUC, confusion matrix, precision, recall, F1-score

Primary Outcome: Stable generalisation performance with interpretable modelling

Dataset

Source: Parkinson’s disease biomedical dataset

Observations: Multiple records per patient

Target Variable: class

0 – Healthy

1 – Parkinson’s disease

Feature Characteristics

Entirely numerical biomedical features

High dimensionality relative to sample size

No missing values present in the dataset

Patient-Level Aggregation

Since each patient appears multiple times in the raw data, records are aggregated by computing the mean of numerical features per patient. This ensures a single representative feature vector per individual and avoids bias due to repeated measurements.

Project Structure
parkinsons-disease-prediction-ml/
│
├── src/
│   └── train.py              # End-to-end training and evaluation pipeline
├── notebooks/
│   └── parkinson_ml.ipynb    # Exploratory analysis (optional)
├── data/
│   └── raw/                  # Dataset location (not included)
├── requirements.txt
├── LICENSE
└── README.md

Methodology
1. Exploratory Data Analysis

Initial inspection includes:

Dataset shape and structure

Data type verification

Descriptive statistical analysis

Missing value assessment

The analysis confirms that the dataset is complete and suitable for direct modelling without imputation.

2. Feature Reduction
Correlation-Based Filtering

Pearson correlation analysis is used to identify highly correlated feature pairs. Features with a correlation coefficient greater than 0.70 are treated as redundant, and one feature from each correlated pair is removed. This step reduces multicollinearity and improves numerical stability.

Statistical Feature Selection

Despite correlation filtering, the feature space remains large relative to the number of observations. To further reduce dimensionality, chi-square feature selection is applied after Min-Max scaling. The top 30 features with the strongest statistical association with the target variable are retained.

3. Data Splitting and Class Imbalance Handling

Train–validation split: 80% training, 20% validation

Class imbalance: Addressed using random oversampling applied exclusively to the training set

Restricting oversampling to the training data prevents information leakage and preserves the validity of validation results.

4. Model Development

Three classifiers are trained and compared:

Logistic Regression (with class weighting)

XGBoost Classifier

Support Vector Classifier (RBF kernel)

These models represent linear, ensemble-based, and kernel-based approaches, enabling a balanced comparison across modelling paradigms.

Model Evaluation
Metrics Used

ROC-AUC (probability-based)

Confusion matrix

Precision, recall, and F1-score

ROC-AUC is computed using predicted probabilities rather than class labels, providing a more informative performance measure in the presence of class imbalance.

Results Summary

Logistic Regression demonstrates stable validation performance with a smaller gap between training and validation scores.

More complex models achieve higher training performance but exhibit increased variance on the validation set.

False negatives remain present across models, highlighting the importance of cautious interpretation in clinical contexts.

Key observation: Simpler, interpretable models generalise more reliably under limited data conditions.

Key Insights

Patient-level aggregation is critical when working with repeated biomedical measurements.

Feature redundancy can significantly affect model stability if not addressed.

Class imbalance handling improves sensitivity to the minority class.

Evaluation metrics must be chosen carefully in healthcare applications, where false negatives are particularly costly.

Limitations

Limited dataset size restricts generalisability.

Feature selection is filter-based and does not capture feature interactions.

Models are evaluated on a single train–validation split rather than cross-validation.

Future Work

Potential extensions include:

Wrapper or embedded feature selection methods

Cross-validation and uncertainty estimation

Integration of longitudinal or clinical metadata

Probabilistic or Bayesian modelling approaches

Conclusion

This project presents a structured and reproducible machine learning pipeline for Parkinson’s disease prediction using numerical biomedical features. By addressing repeated observations, feature redundancy, and class imbalance, the study demonstrates a principled application of supervised learning methods in a healthcare setting. The work serves as a strong methodological baseline for further research-driven investigation.
