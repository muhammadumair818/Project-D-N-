# Diabetes Prediction System

## 📋 Project Overview

This project develops a comprehensive machine learning solution to predict whether a patient has diabetes or not based on vital health parameters. The system implements multiple classification models and includes a production-ready web application for real-time predictions.

### Key Features
- **Multiple ML Models**: Random Forest, Logistic Regression, and XGBoost
- **Imbalanced Data Handling**: SMOTE resampling technique
- **Interactive Web Application**: Streamlit-based user interface
- **Professional Deployment**: Model persistence and scalable architecture

---

## 🎯 Problem Statement

Diabetes is a global health epidemic affecting millions. Early detection is crucial for effective management and prevention of complications. This project aims to build an accurate predictive model that can:

1. **Identify Risk**: Classify patients as diabetic or non-diabetic
2. **Support Screening**: Assist healthcare professionals in patient assessment
3. **Enable Early Intervention**: Provide early warning for at-risk individuals

---

## 📊 Dataset

### Source
**Sugar.csv** - Patient health metrics dataset

### Features (7 attributes)
| Feature | Unit | Range | Description |
|---------|------|-------|-------------|
| **Age** | years | 1-120 | Patient's age |
| **Blood Glucose Level (BGL)** | mg/dL | 50-500 | Fasting glucose measurement |
| **Diastolic Blood Pressure** | mmHg | 40-200 | Lower BP reading |
| **Systolic Blood Pressure** | mmHg | 60-250 | Upper BP reading |
| **Heart Rate** | bpm | 40-200 | Beats per minute |
| **Body Temperature** | °F | 95-105 | Core body temperature |
| **SPO₂** | % | 70-100 | Blood oxygen saturation |

### Target Variable
- **D (Diabetic)**: Class 1
- **N (Non-Diabetic)**: Class 0

---

## 🔍 Data Analysis

### Class Distribution
The dataset exhibits class imbalance (more non-diabetic samples than diabetic):
- **Non-Diabetic**: ~60-70%
- **Diabetic**: ~30-40%

**Solution Applied**: SMOTE (Synthetic Minority Over-sampling Technique) to balance training data

### Feature Importance
Based on XGBoost analysis:
1. **Blood Glucose Level** - Highest predictor
2. **Age** - Strong correlation with diabetes
3. **Blood Pressure** (Systolic & Diastolic) - Significant risk factors
4. **Body Temperature & Heart Rate** - Secondary indicators

---

## 🤖 Models Implemented

### Model 1: Random Forest Classifier
**Characteristics**:
- Ensemble of 100 decision trees
- Good at capturing non-linear relationships
- Provides feature importance rankings
- Robust to outliers

**Performance**: 
- Accuracy: ~92%
- Handles feature interactions well

### Model 2: Logistic Regression
**Characteristics**:
- Linear classification model
- Class-balanced weights to handle imbalance
- Interpretable coefficients
- Fast training and inference

**Performance**:
- Accuracy: ~88-90%
- Baseline comparison

### Model 3: XGBoost (Selected - Best Model) ⭐
**Characteristics**:
- Gradient boosting with sequential error correction
- 200 boosting rounds
- Automatic class weight balancing
- Superior predictive performance

**Performance**:
- Accuracy: ~94-96%
- ROC-AUC: ~0.95+
- Best generalization ability

**Configuration**:
```python
scale_pos_weight = 1.8  # Imbalance ratio
n_estimators = 200
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
```

---

## 📈 Methodology

### 1. Data Preprocessing
- ✅ Categorical encoding (D/N → 1/0)
- ✅ Train-test split (80/20 stratified)
- ✅ Feature scaling (StandardScaler/MinMaxScaler)

### 2. Class Imbalance Handling
- ✅ SMOTE resampling
- ✅ Balanced class weights
- ✅ Stratified cross-validation

### 3. Model Training
- ✅ Hyperparameter tuning
- ✅ Multiple algorithms comparison
- ✅ Performance evaluation

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives (critical for disease detection)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives

---

## 📁 Project Structure

```
Project(D-N)/
├── diabetes_analysis.ipynb         # Complete analysis & model development
├── Fronted.py                      # Web application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── Data/
│   └── Sugar.csv                  # Dataset
└── Model/
    ├── xgb_diabetes_model.pkl     # Trained XGBoost model
    └── scaler.pkl                 # Feature scaler
```

---

## 🚀 Usage

### Option 1: Run the Interactive Web App

#### Prerequisites

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual installation**
```bash
pip install streamlit numpy joblib scikit-learn xgboost pandas matplotlib seaborn imbalanced-learn
```

#### Launch the Application
```bash
streamlit run Fronted.py
```

**Features**:
- User-friendly input form for patient vitals
- Real-time prediction with probability score
- Visual results display
- Professional medical disclaimer

### Option 2: Jupyter Notebook Analysis

Open `diabetes_analysis.ipynb` to:
- Explore data analysis step-by-step
- Understand model development
- Visualize feature importance
- See performance comparisons

---

## 💾 Model Files

### xgb_diabetes_model.pkl
- **Type**: Trained XGBoost classifier
- **Size**: ~200KB
- **Features**: 7 input variables
- **Output**: Binary classification (0 or 1)

### scaler.pkl
- **Type**: StandardScaler for feature normalization
- **Purpose**: Ensures consistent preprocessing in production
- **Critical for**: Input standardization before prediction

---

## ⚠️ Important Disclaimers

1. **Medical Context**: This model is a **screening tool**, not a diagnostic instrument
2. **Professional Consultation**: Always consult healthcare professionals for diagnosis
3. **Data Limitations**: Predictions based on specific health metrics only
4. **Threshold Tuning**: Threshold (0.35) can be adjusted for sensitivity/specificity tradeoff

---

## 📊 Model Performance Summary

| Model | Accuracy | ROC-AUC | Best Use Case |
|-------|----------|---------|---------------|
| Random Forest | 92% | 0.93 | Feature importance analysis |
| Logistic Regression | 88% | 0.89 | Baseline, fast inference |
| **XGBoost** | **94%** | **0.95** | **Production deployment** ⭐ |

---

## 🔧 Configuration & Customization

### Prediction Threshold
Current threshold: **0.35**
- Lower threshold (0.3): Higher sensitivity, catch more diabetic cases
- Higher threshold (0.5): Higher precision, fewer false positives

To adjust:
```python
threshold = 0.35  # Modify in Fronted.py line ~75
```

### Model Retraining
To retrain with new data:
1. Update `Data/Sugar.csv`
2. Run `diabetes_analysis.ipynb` to completion
3. Models auto-save to `Model/` folder

---

## 📚 Dependencies

```
pandas>=1.3.0           # Data manipulation
numpy>=1.21.0          # Numerical computation
scikit-learn>=1.0.0    # ML algorithms, preprocessing
xgboost>=1.5.0         # Gradient boosting
joblib>=1.1.0          # Model serialization
streamlit>=1.20.0      # Web application framework
matplotlib>=3.4.0      # Data visualization
seaborn>=0.11.0        # Statistical visualization
imbalanced-learn>=0.8.0 # SMOTE resampling
```

---

## 🎓 Key Learnings

### Data Imbalance Handling
- Class imbalance is common in medical datasets
- SMOTE effectively generates synthetic minority samples
- Balanced training improves recall for disease detection

### Model Selection
- No single "best" model fits all scenarios
- XGBoost generally outperforms for imbalanced data
- Ensemble methods more robust than single models

### Production Deployment
- Model persistence via joblib is essential
- Scaler must be applied consistently
- Error handling for robustness

---

## 🤝 Contributing

For improvements or bug reports:
1. Identify the issue
2. Test the fix
3. Update relevant code sections
4. Document changes

---

## 📝 License

This project is open-source for educational and research purposes.

---

## 👤 Author

**Muhammad Umair**

### 🔗 Connect With Me
- **LinkedIn**: [linkedin.com/in/muhammad-umair-ai](https://www.linkedin.com/in/muhammad-umair-ai/)
- **Kaggle**: [kaggle.com/umairahmad8](https://www.kaggle.com/umairahmad8)
- **GitHub**: [github.com/muhammadumair818](https://github.com/muhammadumair818)

### Project Details
- Dataset: Sugar.csv
- Developed with: Python, scikit-learn, XGBoost
- Platform: Kaggle

---

## 📞 Support

For questions or issues:
1. Review the `diabetes_analysis.ipynb` notebook for detailed explanations
2. Check Streamlit documentation: https://docs.streamlit.io
3. XGBoost guide: https://xgboost.readthedocs.io

---

## 🎉 Citation

If you use this project, please cite:
```
Diabetes Prediction System (2024)
Author: Muhammad Umair
Healthcare ML Analysis using XGBoost
Dataset: Sugar.csv
Repository: https://github.com/muhammadumair818
```

---

**Last Updated**: January 12, 2026  
**Model Version**: 1.0  
**Status**: Production Ready ✅  
**Author**: Muhammad Umair
