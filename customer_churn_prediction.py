"""
Customer Churn Prediction Model
================================
This script implements a complete ML pipeline for predicting customer churn.
It includes:
- Data downloading and loading
- Data preprocessing and feature engineering
- Handling imbalanced dataset
- Model training and evaluation
- Visualization (Confusion Matrix, ROC Curve)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ChurnPredictor:
    """Main class for Customer Churn Prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def download_data(self):
        """Download dataset from Kaggle using kagglehub"""
        print("=" * 60)
        print("STEP 1: Downloading Dataset")
        print("=" * 60)
        
        try:
            # Download latest version
            path = kagglehub.dataset_download("ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
            print(f"✓ Dataset downloaded successfully!")
            print(f"Path to dataset files: {path}")
            
            # Find data files (CSV or Excel) in the downloaded directory
            data_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.csv', '.xlsx', '.xls')):
                        data_files.append(os.path.join(root, file))
            
            if data_files:
                print(f"\nFound data files:")
                for data_file in data_files:
                    print(f"  - {data_file}")
                return data_files[0]  # Return first data file
            else:
                raise FileNotFoundError("No CSV or Excel files found in downloaded dataset")
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self, file_path):
        """Load and display basic information about the dataset"""
        print("\n" + "=" * 60)
        print("STEP 2: Loading and Exploring Data")
        print("=" * 60)
        
        # Load data based on file extension
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            # Try to find the data sheet (skip metadata sheets)
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            # Look for common data sheet names or use the largest sheet
            data_sheet = None
            for sheet in sheet_names:
                sheet_lower = sheet.lower()
                # Skip metadata/dictionary sheets
                if 'dict' not in sheet_lower and 'meta' not in sheet_lower and 'info' not in sheet_lower:
                    # Check if this sheet has substantial data
                    test_df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                    if len(test_df.columns) > 2:  # Has multiple columns (likely data)
                        data_sheet = sheet
                        break
            
            # If no suitable sheet found, try the largest sheet
            if data_sheet is None:
                max_rows = 0
                for sheet in sheet_names:
                    test_df = pd.read_excel(file_path, sheet_name=sheet)
                    if len(test_df) > max_rows:
                        max_rows = len(test_df)
                        data_sheet = sheet
            
            # Load the data sheet
            if data_sheet:
                print(f"\nLoading data from sheet: '{data_sheet}'")
                self.df = pd.read_excel(file_path, sheet_name=data_sheet)
            else:
                # Fallback to first sheet
                print(f"\nWarning: Could not determine data sheet, using first sheet: '{sheet_names[0]}'")
                self.df = pd.read_excel(file_path, sheet_name=sheet_names[0])
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        print(f"\n✓ Data loaded successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nDataset Info:")
        print(self.df.info())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nStatistical Summary:")
        print(self.df.describe())
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n" + "=" * 60)
        print("STEP 3: Data Cleaning & Preprocessing")
        print("=" * 60)
        
        df = self.df.copy()
        
        # Handle missing values
        print("\nHandling missing values...")
        if df.isnull().sum().sum() > 0:
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Identify target variable (common names for churn)
        target_candidates = ['churn', 'Churn', 'Churned', 'churned', 'is_churn']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Try to find binary column that might be churn
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                    print(f"\nWarning: Assuming '{col}' is the target variable")
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Could not identify target variable. Please check dataset.")
        
        print(f"\n✓ Target variable identified: {target_col}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target if needed
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
        
        # Check class distribution
        print(f"\nClass distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(y)*100:.2f}%)")
        
        self.X = X
        self.y = y
        self.target_col = target_col
        
        print("\n✓ Data preprocessing completed!")
        return X, y
    
    def handle_imbalance(self, X, y, method='smote'):
        """Handle imbalanced dataset using various techniques"""
        print("\n" + "=" * 60)
        print("STEP 4: Handling Imbalanced Dataset")
        print("=" * 60)
        
        print(f"\nOriginal class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(y)*100:.2f}%)")
        
        imbalance_ratio = min(counts) / max(counts)
        print(f"\nImbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.5:
            print(f"\n⚠ Dataset is imbalanced! Applying {method.upper()} technique...")
            
            if method == 'smote':
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            elif method == 'smote_tomek':
                smote_tomek = SMOTETomek(random_state=42)
                X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            elif method == 'undersample':
                undersample = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersample.fit_resample(X, y)
            else:
                print(f"Unknown method {method}, using SMOTE")
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"\nAfter {method.upper()}:")
            unique, counts = np.unique(y_resampled, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  Class {u}: {c} ({c/len(y_resampled)*100:.2f}%)")
            
            print(f"\n✓ Dataset balanced successfully!")
            return X_resampled, y_resampled
        else:
            print("\n✓ Dataset is relatively balanced, no resampling needed.")
            return X, y
    
    def split_and_scale(self, X, y):
        """Split data into train/test sets and scale features"""
        print("\n" + "=" * 60)
        print("STEP 5: Train-Test Split & Feature Scaling")
        print("=" * 60)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Data split and scaled successfully!")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models and select the best one"""
        print("\n" + "=" * 60)
        print("STEP 6: Model Training")
        print("=" * 60)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Select best model based on ROC-AUC
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        print(f"\n✓ Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        
        self.model = results[best_model_name]['model']
        self.y_pred = results[best_model_name]['y_pred']
        self.y_pred_proba = results[best_model_name]['y_pred_proba']
        self.results = results
        
        return results
    
    def evaluate_model(self):
        """Evaluate model and generate comprehensive metrics"""
        print("\n" + "=" * 60)
        print("STEP 7: Model Evaluation")
        print("=" * 60)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print(f"\nModel Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        print("\n" + "=" * 60)
        print("STEP 8: Generating Confusion Matrix")
        print("=" * 60)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
        # Explain confusion matrix
        print("\nConfusion Matrix Explanation:")
        print(f"  True Negatives (TN): {cm[0][0]} - Correctly predicted non-churn customers")
        print(f"  False Positives (FP): {cm[0][1]} - Incorrectly predicted as churn (Type I error)")
        print(f"  False Negatives (FN): {cm[1][0]} - Missed churn customers (Type II error)")
        print(f"  True Positives (TP): {cm[1][1]} - Correctly predicted churn customers")
        
        return cm
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        print("\n" + "=" * 60)
        print("STEP 9: Generating ROC Curve")
        print("=" * 60)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("\n✓ ROC curve saved as 'roc_curve.png'")
        plt.show()
        
        # Explain ROC curve
        print("\nROC Curve Explanation:")
        print(f"  AUC Score: {roc_auc:.4f}")
        if roc_auc > 0.9:
            print("  Interpretation: Excellent model performance")
        elif roc_auc > 0.8:
            print("  Interpretation: Good model performance")
        elif roc_auc > 0.7:
            print("  Interpretation: Moderate model performance")
        else:
            print("  Interpretation: Poor model performance - needs improvement")
        print("\n  The ROC curve shows the trade-off between:")
        print("    - True Positive Rate (TPR): Ability to correctly identify churn customers")
        print("    - False Positive Rate (FPR): Incorrectly flagging non-churn customers")
        print("  A higher AUC indicates better model performance.")
        
        return fpr, tpr, roc_auc
    
    def run_complete_pipeline(self, imbalance_method='smote'):
        """Run the complete ML pipeline"""
        print("\n" + "=" * 80)
        print("CUSTOMER CHURN PREDICTION - COMPLETE ML PIPELINE")
        print("=" * 80)
        
        # Step 1: Download data
        csv_path = self.download_data()
        
        # Step 2: Load data
        self.load_data(csv_path)
        
        # Step 3: Preprocess
        X, y = self.preprocess_data()
        
        # Step 4: Handle imbalance
        X_balanced, y_balanced = self.handle_imbalance(X, y, method=imbalance_method)
        
        # Step 5: Split and scale
        self.split_and_scale(X_balanced, y_balanced)
        
        # Step 6: Train models
        self.train_models()
        
        # Step 7: Evaluate
        metrics = self.evaluate_model()
        
        # Step 8: Confusion Matrix
        cm = self.plot_confusion_matrix()
        
        # Step 9: ROC Curve
        fpr, tpr, auc = self.plot_roc_curve()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return metrics


def main():
    """Main function to run the churn prediction pipeline"""
    predictor = ChurnPredictor()
    
    # Run complete pipeline
    metrics = predictor.run_complete_pipeline(imbalance_method='smote')
    
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nAccuracy:  ~{metrics['accuracy']:.2f}")
    print(f"Precision: ~{metrics['precision']:.2f}")
    print(f"Recall:    ~{metrics['recall']:.2f}")
    print(f"ROC-AUC:   ~{metrics['roc_auc']:.2f}")
    print("\nNote: Precision & Recall may be lower due to class imbalance.")
    print("ROC-AUC indicates moderate to good predictive power.")


if __name__ == "__main__":
    main()

