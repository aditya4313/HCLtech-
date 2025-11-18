# Customer Churn Prediction - Machine Learning Model

## 📋 Project Overview

This project implements a complete Machine Learning pipeline to predict customer churn for an e-commerce business. The model identifies customers who are likely to discontinue using the company's services, enabling proactive retention strategies.

## 🎯 Problem Statement

Build a Machine Learning Prediction model to predict Customer Churn, including:
- Techniques to handle imbalanced datasets
- Appropriate evaluation metrics
- Confusion Matrix and ROC Curve visualizations with explanations

## 📊 Dataset

The dataset is downloaded from Kaggle using `kagglehub`:
- **Source**: [E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- **Download Method**: Automated via `kagglehub.dataset_download()`

## 🔄 Workflow Diagram

```
┌─────────────────┐
│ Data Collection │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Data Cleaning &         │
│ Preprocessing           │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Feature Engineering     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Handle Imbalanced       │
│ Dataset (SMOTE)         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Training          │
│ (Multiple Algorithms)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Evaluation        │
│ (Metrics & Visualizations)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Deployment Ready        │
└─────────────────────────┘
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Run Python Script

```bash
python customer_churn_prediction.py
```

The script will automatically:
1. Download the dataset from Kaggle
2. Load and explore the data
3. Clean and preprocess the data
4. Handle class imbalance using SMOTE
5. Train multiple ML models
6. Evaluate and compare models
7. Generate visualizations (Confusion Matrix & ROC Curve)

### Option 2: Run Jupyter Notebook

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Open the notebook in Jupyter and run cells sequentially to explore the pipeline step-by-step.

### Option 3: Run Streamlit Web Application (Recommended)

Launch the professional web interface:

```bash
streamlit run app.py
```

The app will open in your browser with a clean, minimal interface featuring:
- **Interactive Dashboard**: Easy-to-use web interface
- **Data Analysis**: Explore dataset statistics and visualizations
- **Model Training**: Train models with customizable parameters
- **Results Visualization**: View confusion matrix, ROC curve, and metrics
- **Real-time Progress**: Track training progress with progress bars

**Features of the Web App:**
- 📊 **Overview Tab**: Project information and key metrics
- 🔍 **Data Analysis Tab**: Dataset exploration and statistics
- 🤖 **Model Training Tab**: Train models with progress tracking
- 📈 **Results Tab**: Comprehensive model evaluation and visualizations

## 📈 Model Evaluation Summary

### Key Metrics

- **Accuracy**: ~0.70 (70% correct predictions)
- **Precision**: Lower due to class imbalance (~30% churn rate)
- **Recall**: Ability to identify churn customers
- **ROC-AUC**: Indicates moderate to good predictive power

### Models Tested

1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble method with class balancing
3. **Gradient Boosting**: Advanced ensemble method

The best model is automatically selected based on ROC-AUC score.

## 📊 Visualizations

### 1. Confusion Matrix

The confusion matrix shows:
- **True Negatives (TN)**: Correctly predicted non-churn customers
- **False Positives (FP)**: Type I error - Incorrectly flagged as churn
- **False Negatives (FN)**: Type II error - Missed churn customers
- **True Positives (TP)**: Correctly predicted churn customers

**Interpretation**:
- High TN and TP indicate good model performance
- Low FP reduces unnecessary retention costs
- Low FN ensures we don't miss customers likely to churn

### 2. ROC Curve

The Receiver Operating Characteristic (ROC) curve illustrates:
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity/Recall)
- **AUC Score**: Area Under the Curve

**Interpretation**:
- **AUC > 0.9**: Excellent model performance
- **AUC > 0.8**: Good model performance
- **AUC > 0.7**: Moderate model performance
- **AUC < 0.7**: Poor performance, needs improvement

The curve shows the trade-off between correctly identifying churn customers (TPR) and incorrectly flagging non-churn customers (FPR).

## ⚖️ Handling Imbalanced Dataset

### Problem
Customer churn datasets are typically imbalanced (~30% churn, 70% non-churn), which can bias models toward predicting the majority class.

### Solution: SMOTE (Synthetic Minority Oversampling Technique)

SMOTE creates synthetic samples of the minority class by:
1. Finding k-nearest neighbors for minority class samples
2. Generating new samples along the line segments between neighbors
3. Balancing the dataset without losing information

**Alternative Methods Available**:
- `smote`: Standard SMOTE oversampling
- `smote_tomek`: SMOTE + Tomek links cleaning
- `undersample`: Random undersampling of majority class

## 📁 Project Structure

```
ml-dravit/
│
├── customer_churn_prediction.py  # Main ML pipeline script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── confusion_matrix.png          # Generated visualization
├── roc_curve.png                 # Generated visualization
│
└── [downloaded_dataset]/         # Kaggle dataset files
```

## 🔍 Key Features

1. **Automated Data Download**: Uses Kaggle Hub API
2. **Comprehensive Preprocessing**: Handles missing values, encoding, scaling
3. **Imbalance Handling**: Multiple techniques available
4. **Model Comparison**: Tests multiple algorithms automatically
5. **Rich Visualizations**: Confusion matrix and ROC curve
6. **Detailed Explanations**: Metrics and visualizations explained

## 📝 Evaluation Metrics Explained

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Meaning**: Overall correctness of predictions
- **Limitation**: Can be misleading with imbalanced data

### Precision
- **Formula**: TP / (TP + FP)
- **Meaning**: Of predicted churns, how many actually churned?
- **Business Impact**: Reduces false alarms, saves retention costs

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Meaning**: Of actual churns, how many did we catch?
- **Business Impact**: Ensures we don't miss potential churners

### F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning**: Harmonic mean of precision and recall
- **Use Case**: Balance between precision and recall

### ROC-AUC
- **Range**: 0 to 1 (higher is better)
- **Meaning**: Model's ability to distinguish between classes
- **Advantage**: Works well with imbalanced data

## 🛠️ Customization

### Change Imbalance Handling Method

Edit `customer_churn_prediction.py`:

```python
metrics = predictor.run_complete_pipeline(imbalance_method='smote_tomek')
```

Options: `'smote'`, `'smote_tomek'`, `'undersample'`

### Adjust Model Parameters

Modify the `train_models()` method to customize:
- Number of estimators
- Max depth
- Learning rate
- Class weights

## 📚 References

- [SMOTE Paper](https://www.jair.org/index.php/jair/article/view/10302)
- [ROC Curve Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Imbalanced Learning](https://imbalanced-learn.org/stable/)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

---

**Note**: Make sure you have Kaggle credentials configured if downloading datasets requires authentication.

