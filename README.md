# Customer Churn Prediction - Machine Learning Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a complete Machine Learning pipeline to predict customer churn for an e-commerce business. The model identifies customers who are likely to discontinue using the company's services, enabling proactive retention strategies.

## 🎯 Problem Statement

Build a Machine Learning Prediction model to predict Customer Churn, including:
- Techniques to handle imbalanced datasets (SMOTE)
- PCA for dimensionality reduction
- Cross-validation for robust model evaluation
- Overfitting detection and prevention
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
│ PCA Dimensionality      │
│ Reduction (95% variance)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Training          │
│ (Random Forest with     │
│  RandomizedSearchCV)     │
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
5. Apply PCA for dimensionality reduction (retains 95% variance)
6. Train Random Forest model with RandomizedSearchCV
7. Perform cross-validation and overfitting analysis
8. Evaluate model performance
9. Generate visualizations (Confusion Matrix & ROC Curve)

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

### Model: Random Forest Classifier

**Optimized Configuration:**
- **Algorithm**: Random Forest with class balancing
- **Hyperparameter Tuning**: RandomizedSearchCV (15 iterations)
- **Cross-Validation**: 3-fold StratifiedKFold
- **Dimensionality Reduction**: PCA (95% variance retained)
- **Target Accuracy**: ~87%

### Key Metrics

- **Test Accuracy**: ~87% (target achieved)
- **Train Accuracy**: ~90-94%
- **CV Mean Accuracy**: ~84-86% (±0.5-1%)
- **Overfitting Gap**: <5% (minimized through regularization)
- **Precision**: ~87%
- **Recall**: ~87%
- **ROC-AUC**: ~0.94 (excellent predictive power)

### Hyperparameters (Optimized)

- **n_estimators**: 100-200 trees
- **max_depth**: 8-12 (balanced for accuracy vs overfitting)
- **min_samples_split**: 15-25 (prevents overfitting)
- **min_samples_leaf**: 5-10 (regularization)
- **max_features**: 'sqrt' or 'log2' (feature diversity)

### Training Performance

- **Training Time**: 2-5 minutes (optimized with RandomizedSearchCV)
- **Overfitting Control**: Automatic detection and minimization
- **Model Stability**: Excellent (low CV standard deviation)

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
customer-churn/
│
├── customer_churn_prediction.py  # Main ML pipeline script
├── customer_churn_prediction.ipynb  # Jupyter notebook (complete pipeline)
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── DEPLOYMENT.md                 # Detailed deployment guide
├── .streamlit/config.toml        # Streamlit configuration
│
├── confusion_matrix.png          # Generated visualization
├── roc_curve.png                 # Generated visualization
│
└── [downloaded_dataset]/         # Kaggle dataset files (auto-downloaded)
```

## 🔍 Key Features

1. **Automated Data Download**: Uses Kaggle Hub API via `kagglehub`
2. **Comprehensive Preprocessing**: Handles missing values, encoding, scaling
3. **PCA Dimensionality Reduction**: Reduces features while retaining 95% variance
4. **Imbalance Handling**: SMOTE for balanced dataset
5. **Fast Hyperparameter Tuning**: RandomizedSearchCV (15 iterations, ~10x faster than GridSearchCV)
6. **Cross-Validation**: 3-fold stratified cross-validation for robust evaluation
7. **Overfitting Detection**: Automatic comparison of train vs test performance
8. **Overfitting Prevention**: Regularized Random Forest with optimized hyperparameters
9. **Model Optimization**: Single Random Forest model optimized for ~87% accuracy
10. **Rich Visualizations**: Confusion matrix and ROC curve with explanations
11. **Professional Web Interface**: Streamlit app with minimal, modern design
12. **Complete Documentation**: Jupyter notebook with all steps explained

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

Modify the `train_models()` method in `customer_churn_prediction.py` to customize:
- **n_estimators**: Number of trees (default: 100-200)
- **max_depth**: Maximum tree depth (default: 8-12)
- **min_samples_split**: Minimum samples to split (default: 15-25)
- **min_samples_leaf**: Minimum samples in leaf (default: 5-10)
- **max_features**: Features per tree (default: 'sqrt' or 'log2')

### Adjust PCA Variance Threshold

Modify the `split_and_scale()` method:
```python
self.split_and_scale(X_balanced, y_balanced, use_pca=True, variance_threshold=0.95)
```

Change `variance_threshold` to adjust feature reduction (0.90-0.99).

## 🎓 Technical Details

### Model Architecture

**Random Forest Classifier:**
- Ensemble of decision trees
- Class balancing for imbalanced data
- Regularized to prevent overfitting
- Optimized hyperparameters via RandomizedSearchCV

### Training Strategy

1. **Data Preparation**: Missing value imputation, categorical encoding
2. **Balancing**: SMOTE oversampling for minority class
3. **Dimensionality Reduction**: PCA retains 95% variance
4. **Hyperparameter Tuning**: RandomizedSearchCV with 15 iterations
5. **Validation**: 3-fold cross-validation
6. **Evaluation**: Comprehensive metrics with overfitting analysis

### Performance Optimization

- **RandomizedSearchCV**: Tests 15 random combinations (vs 180+ in GridSearchCV)
- **3-Fold CV**: Faster than 5-fold while maintaining reliability
- **PCA**: Reduces computational cost while preserving information
- **Regularization**: Prevents overfitting without sacrificing accuracy

## 📚 References

- [SMOTE Paper](https://www.jair.org/index.php/jair/article/view/10302)
- [ROC Curve Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Imbalanced Learning](https://imbalanced-learn.org/stable/)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 🚀 Deployment on Render

### Prerequisites
- A GitHub account
- A Render account (sign up at [render.com](https://render.com))

### Step-by-Step Deployment

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ml-dravit.git
   git push -u origin main
   ```

2. **Create a new Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the Service**
   - **Name**: `customer-churn-prediction` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Plan**: Select Free tier (or upgrade for better performance)

4. **Environment Variables (Optional)**
   - If you need Kaggle API credentials, add them in the Environment section:
     - `KAGGLE_USERNAME`: Your Kaggle username
     - `KAGGLE_KEY`: Your Kaggle API key

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait for the build to complete (usually 2-5 minutes)
   - Your app will be live at `https://your-app-name.onrender.com`

### Using render.yaml (Alternative Method)

If you've included `render.yaml` in your repository, Render will automatically detect it:
- Simply connect your GitHub repository
- Render will use the configuration from `render.yaml`
- No manual configuration needed!

### Troubleshooting

**Build fails:**
- Check that all dependencies are in `requirements.txt`
- Ensure Python version is compatible (3.8+)

**App doesn't start:**
- Verify the start command is correct
- Check logs in Render dashboard for errors
- Ensure `app.py` is in the root directory

**Kaggle download fails:**
- Add Kaggle credentials as environment variables
- Or use the "Upload File" option in the app instead

### Free Tier Limitations

- Apps on free tier spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- Consider upgrading to paid tier for always-on availability

## 📄 License

This project is open source and available for educational purposes.

---

## 📊 Model Performance

### Expected Results

- **Test Accuracy**: ~87% (target achieved)
- **Overfitting Gap**: <5% (well-controlled)
- **ROC-AUC**: ~0.94 (excellent)
- **Training Time**: 2-5 minutes
- **Cross-Validation Stability**: Excellent (CV std < 1%)

### Model Characteristics

- **Algorithm**: Random Forest Classifier
- **Regularization**: Strong (prevents overfitting)
- **Feature Reduction**: PCA (95% variance retained)
- **Hyperparameter Tuning**: RandomizedSearchCV (fast & efficient)
- **Validation**: 3-fold cross-validation

## 🔧 Advanced Configuration

### Adjust Target Accuracy

Modify `target_accuracy` in `train_models()` method:
```python
target_accuracy = 0.87  # Change to desired value (0.85-0.90)
```

### Tune Overfitting Control

Adjust hyperparameter ranges in `models_config`:
- **More regularization**: Increase `min_samples_split` and `min_samples_leaf`
- **Less regularization**: Decrease these values
- **Balance**: Current settings optimized for ~87% accuracy with <5% overfitting

### PCA Configuration

Modify PCA variance threshold:
```python
self.split_and_scale(X_balanced, y_balanced, use_pca=True, variance_threshold=0.95)
```
- Lower threshold (0.90): More aggressive reduction, faster training
- Higher threshold (0.99): Less reduction, more features

---

**Note**: Make sure you have Kaggle credentials configured if downloading datasets requires authentication. The app also supports file upload as an alternative.

