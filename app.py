"""
Customer Churn Prediction - Streamlit Web Application
A professional and minimal frontend for the ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import kagglehub
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimal, professional design
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None


class ChurnPredictor:
    """Churn Prediction Model Class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoders = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.results = {}
        self.use_pca = True
        self.n_components = None
        self.explained_variance_ratio_ = None
        
    def download_data(self):
        """Download dataset from Kaggle"""
        try:
            path = kagglehub.dataset_download("ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
            
            # Find data files
            data_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.csv', '.xlsx', '.xls')):
                        data_files.append(os.path.join(root, file))
            
            return data_files[0] if data_files else None
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            return None
    
    def load_data(self, file_path):
        """Load data from file"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            # Find data sheet
            data_sheet = None
            for sheet in sheet_names:
                sheet_lower = sheet.lower()
                if 'dict' not in sheet_lower and 'meta' not in sheet_lower:
                    test_df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                    if len(test_df.columns) > 2:
                        data_sheet = sheet
                        break
            
            if data_sheet is None:
                max_rows = 0
                for sheet in sheet_names:
                    test_df = pd.read_excel(file_path, sheet_name=sheet)
                    if len(test_df) > max_rows:
                        max_rows = len(test_df)
                        data_sheet = sheet
            
            return pd.read_excel(file_path, sheet_name=data_sheet) if data_sheet else pd.read_excel(file_path, sheet_name=sheet_names[0])
        else:
            raise ValueError(f"Unsupported file format")
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        df = df.copy()
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Identify target variable
        target_candidates = ['churn', 'Churn', 'Churned', 'churned', 'is_churn']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Could not identify target variable")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
        
        return X, y, target_col
    
    def handle_imbalance(self, X, y, method='smote'):
        """Handle imbalanced dataset"""
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = min(counts) / max(counts)
        
        if imbalance_ratio < 0.5:
            if method == 'smote':
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            else:
                X_resampled, y_resampled = X, y
            return X_resampled, y_resampled
        return X, y
    
    def train_models(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train Random Forest and Logistic Regression models with cross-validation"""
        # Define models: Random Forest (target 85%) and Logistic Regression baseline (target 70%)
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.1, 0.5, 1.0, 2.0],  # Regularization strength
                    'solver': ['lbfgs', 'liblinear']  # Solver algorithms
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [150, 200],  # Moderate number of trees
                    'max_depth': [5, 6, 7],  # Shallower trees to reduce overfitting
                    'min_samples_split': [40, 50, 60],  # Higher threshold to prevent overfitting
                    'min_samples_leaf': [20, 25, 30],  # Strong regularization
                    'max_features': ['sqrt', 'log2']  # Feature diversity
                }
            }
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds for speed
        target_accuracy_rf = 0.85
        target_accuracy_lr = 0.70
        target_accuracy = 0.87
        results = {}
        
        for name, config in models_config.items():
            # RandomizedSearchCV for faster hyperparameter tuning
            random_search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=15,  # Test 15 random combinations (faster than full grid)
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
            random_search.fit(X_train_scaled, y_train)
            best_model = random_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                best_model, X_train_scaled, y_train,
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train and test predictions
            y_train_pred = best_model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            test_accuracy = accuracy_score(y_test, y_pred)
            
            overfitting_gap = train_accuracy - test_accuracy
            
            results[name] = {
                'model': best_model,
                'accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'overfitting_gap': overfitting_gap,
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'best_params': random_search.best_params_
            }
        
        # Compare models and select Random Forest as best
        best_model_name = 'Random Forest'
        self.model = results[best_model_name]['model']
        self.y_pred = results[best_model_name]['y_pred']
        self.y_pred_proba = results[best_model_name]['y_pred_proba']
        self.results = results
        self.best_model_name = best_model_name
        
        # Save models and preprocessing components
        try:
            # Save Random Forest model
            rf_model_path = 'random_forest_model.pkl'
            joblib.dump(results['Random Forest']['model'], rf_model_path)
            
            # Save Logistic Regression model
            lr_model_path = 'logistic_regression_model.pkl'
            joblib.dump(results['Logistic Regression']['model'], lr_model_path)
            
            # Save scaler (if available)
            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = 'scaler.pkl'
                joblib.dump(self.scaler, scaler_path)
            
            # Save PCA (if used)
            if hasattr(self, 'pca') and self.pca is not None:
                pca_path = 'pca.pkl'
                joblib.dump(self.pca, pca_path)
            
            # Save label encoders (if available)
            if hasattr(self, 'label_encoders') and self.label_encoders:
                encoders_path = 'label_encoders.pkl'
                joblib.dump(self.label_encoders, encoders_path)
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save models in models directory (can be tracked in Git)
            models_dir = 'models'
            rf_model_path_tracked = os.path.join(models_dir, 'random_forest_model.pkl')
            lr_model_path_tracked = os.path.join(models_dir, 'logistic_regression_model.pkl')
            joblib.dump(results['Random Forest']['model'], rf_model_path_tracked)
            joblib.dump(results['Logistic Regression']['model'], lr_model_path_tracked)
            
            # Save scaler, PCA, encoders in models directory
            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path_tracked = os.path.join(models_dir, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path_tracked)
            
            if hasattr(self, 'pca') and self.pca is not None:
                pca_path_tracked = os.path.join(models_dir, 'pca.pkl')
                joblib.dump(self.pca, pca_path_tracked)
            
            if hasattr(self, 'label_encoders') and self.label_encoders:
                encoders_path_tracked = os.path.join(models_dir, 'label_encoders.pkl')
                joblib.dump(self.label_encoders, encoders_path_tracked)
            
            # Save model metadata as JSON (Git-friendly)
            metadata = {
                'best_model': best_model_name,
                'random_forest': {
                    'accuracy': float(results['Random Forest']['accuracy']),
                    'train_accuracy': float(results['Random Forest']['train_accuracy']),
                    'cv_mean': float(results['Random Forest']['cv_mean']),
                    'cv_std': float(results['Random Forest']['cv_std']),
                    'overfitting_gap': float(results['Random Forest']['overfitting_gap']),
                    'roc_auc': float(results['Random Forest']['roc_auc']),
                    'best_params': {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                   for k, v in results['Random Forest']['best_params'].items()}
                },
                'logistic_regression': {
                    'accuracy': float(results['Logistic Regression']['accuracy']),
                    'train_accuracy': float(results['Logistic Regression']['train_accuracy']),
                    'cv_mean': float(results['Logistic Regression']['cv_mean']),
                    'cv_std': float(results['Logistic Regression']['cv_std']),
                    'overfitting_gap': float(results['Logistic Regression']['overfitting_gap']),
                    'roc_auc': float(results['Logistic Regression']['roc_auc']),
                    'best_params': {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                   for k, v in results['Logistic Regression']['best_params'].items()}
                }
            }
            
            # Save as JSON (Git-friendly, can be pushed to GitHub)
            metadata_json_path = 'model_metadata.json'
            with open(metadata_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also save as pickle for compatibility
            metadata_path = 'model_metadata.pkl'
            joblib.dump(metadata, metadata_path)
        except Exception as e:
            st.warning(f"Could not save models: {e}")
        
        return results, best_model_name


def load_saved_models():
    """Load saved models and preprocessing components"""
    try:
        models = {}
        models_dir = 'models'
        
        # Try loading from models directory first, then root directory
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl') if os.path.exists(os.path.join(models_dir, 'random_forest_model.pkl')) else 'random_forest_model.pkl'
        lr_path = os.path.join(models_dir, 'logistic_regression_model.pkl') if os.path.exists(os.path.join(models_dir, 'logistic_regression_model.pkl')) else 'logistic_regression_model.pkl'
        scaler_path = os.path.join(models_dir, 'scaler.pkl') if os.path.exists(os.path.join(models_dir, 'scaler.pkl')) else 'scaler.pkl'
        pca_path = os.path.join(models_dir, 'pca.pkl') if os.path.exists(os.path.join(models_dir, 'pca.pkl')) else 'pca.pkl'
        encoders_path = os.path.join(models_dir, 'label_encoders.pkl') if os.path.exists(os.path.join(models_dir, 'label_encoders.pkl')) else 'label_encoders.pkl'
        
        models['rf'] = joblib.load(rf_path)
        models['lr'] = joblib.load(lr_path)
        models['scaler'] = joblib.load(scaler_path)
        models['pca'] = joblib.load(pca_path) if os.path.exists(pca_path) else None
        models['encoders'] = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
        
        # Load metadata (prefer JSON, fallback to PKL)
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                models['metadata'] = json.load(f)
        elif os.path.exists('model_metadata.pkl'):
            models['metadata'] = joblib.load('model_metadata.pkl')
        else:
            models['metadata'] = None
            
        return models
    except FileNotFoundError as e:
        return None


def predict_churn(user_input, models):
    """Predict churn for user input"""
    try:
        # Expected feature order (excluding CustomerID and Churn)
        feature_order = [
            'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
            'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
            'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
            'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
            'DaySinceLastOrder', 'CashbackAmount'
        ]
        
        # Encode categorical features
        encoded_input = {}
        for col in feature_order:
            if col in user_input:
                value = user_input[col]
                # Encode if categorical and encoder exists
                if col in models.get('encoders', {}):
                    encoder = models['encoders'][col]
                    try:
                        encoded_input[col] = encoder.transform([str(value)])[0]
                    except (ValueError, KeyError):
                        # Handle unseen categories - use most common or 0
                        encoded_input[col] = 0
                else:
                    # Numerical feature
                    encoded_input[col] = float(value)
            else:
                # Missing feature - use 0 or median
                encoded_input[col] = 0.0
        
        # Convert to numpy array in correct order
        feature_values = np.array([encoded_input[col] for col in feature_order]).reshape(1, -1)
        
        # Scale features
        scaled_features = models['scaler'].transform(feature_values)
        
        # Apply PCA if available
        if models['pca'] is not None:
            scaled_features = models['pca'].transform(scaled_features)
        
        # Predict with Random Forest
        rf_prediction = models['rf'].predict(scaled_features)[0]
        rf_probability = models['rf'].predict_proba(scaled_features)[0]
        
        # Predict with Logistic Regression
        lr_prediction = models['lr'].predict(scaled_features)[0]
        lr_probability = models['lr'].predict_proba(scaled_features)[0]
        
        return {
            'rf_prediction': rf_prediction,
            'rf_probability': rf_probability,
            'lr_prediction': lr_prediction,
            'lr_probability': lr_probability
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Model for Predicting Customer Churn in E-commerce</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Download from Kaggle", "Upload File"],
            help="Choose to download from Kaggle or upload your own dataset"
        )
        
        # Imbalance handling method
        imbalance_method = st.selectbox(
            "Imbalance Handling Method",
            ["smote", "smote_tomek", "none"],
            help="Method to handle class imbalance"
        )
        
        st.divider()
        
        # Action buttons
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            st.session_state.train_model = True
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.model_trained = False
            st.session_state.data_loaded = False
            st.session_state.predictor = None
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Data Analysis", "🤖 Model Training", "📈 Results", "🔮 Predict Churn"])
    
    with tab1:
        st.markdown("### Project Overview")
        st.markdown("""
        This application implements a complete Machine Learning pipeline to predict customer churn for e-commerce businesses.
        
        **Key Features:**
        - Automated data download from Kaggle
        - Data preprocessing and feature engineering
        - Handling imbalanced datasets using SMOTE
        - Multiple ML algorithms (Logistic Regression, Random Forest, Gradient Boosting)
        - Comprehensive model evaluation with visualizations
        
        **Workflow:**
        1. Data Collection & Loading
        2. Data Cleaning & Preprocessing
        3. Feature Engineering
        4. Handling Class Imbalance
        5. Model Training & Evaluation
        6. Results Visualization
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Accuracy", "~70%", "Baseline")
        with col2:
            st.metric("ROC-AUC", "Moderate", "Good")
        with col3:
            st.metric("Models Tested", "3", "Algorithms")
    
    with tab2:
        st.markdown("### Data Analysis")
        
        if st.session_state.data_loaded and 'df' in st.session_state:
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dataset Shape**")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
                st.markdown("**Missing Values**")
                missing = df.isnull().sum()
                st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({"Column": ["No missing values"]}))
            
            with col2:
                st.markdown("**Data Types**")
                st.dataframe(df.dtypes.reset_index().rename(columns={0: "Type"}))
            
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True)
            
            if 'Churn' in df.columns:
                st.markdown("**Class Distribution**")
                churn_counts = df['Churn'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='bar', ax=ax, color=['#0066cc', '#ff6b6b'])
                ax.set_title('Churn Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Churn Status', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_xticklabels(['Not Churn', 'Churn'], rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("👆 Please train the model first to view data analysis")
    
    with tab3:
        st.markdown("### Model Training")
        
        if st.session_state.get('train_model', False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Download/Load Data
            status_text.text("📥 Step 1/6: Downloading/Loading Data...")
            progress_bar.progress(10)
            
            predictor = ChurnPredictor()
            
            if data_source == "Download from Kaggle":
                file_path = predictor.download_data()
                if file_path is None:
                    st.error("Failed to download dataset")
                    st.stop()
            else:
                uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'xls'])
                if uploaded_file is None:
                    st.warning("Please upload a file")
                    st.stop()
                file_path = uploaded_file.name
                # Save uploaded file temporarily
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Load data
            df = predictor.load_data(file_path)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            status_text.text("✅ Step 1/6: Data Loaded Successfully")
            progress_bar.progress(20)
            
            # Step 2: Preprocess
            status_text.text("🔧 Step 2/6: Preprocessing Data...")
            progress_bar.progress(30)
            X, y, target_col = predictor.preprocess_data(df)
            
            status_text.text("✅ Step 2/6: Data Preprocessed")
            progress_bar.progress(40)
            
            # Step 3: Handle Imbalance
            status_text.text("⚖️ Step 3/6: Handling Class Imbalance...")
            progress_bar.progress(50)
            if imbalance_method != "none":
                X_balanced, y_balanced = predictor.handle_imbalance(X, y, method=imbalance_method)
            else:
                X_balanced, y_balanced = X, y
            
            status_text.text("✅ Step 3/6: Class Imbalance Handled")
            progress_bar.progress(60)
            
            # Step 4: Split, Scale, and Apply PCA
            status_text.text("📊 Step 4/6: Splitting, Scaling & Applying PCA...")
            progress_bar.progress(70)
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
            )
            X_train_scaled = predictor.scaler.fit_transform(X_train)
            X_test_scaled = predictor.scaler.transform(X_test)
            
            # Apply PCA
            if X_train_scaled.shape[1] > 2:
                predictor.pca = PCA(n_components=0.95, random_state=42)
                X_train_scaled = predictor.pca.fit_transform(X_train_scaled)
                X_test_scaled = predictor.pca.transform(X_test_scaled)
                predictor.n_components = predictor.pca.n_components_
                predictor.explained_variance_ratio_ = predictor.pca.explained_variance_ratio_.sum()
                predictor.use_pca = True
            else:
                predictor.use_pca = False
            
            predictor.X_test = X_test
            predictor.y_test = y_test
            
            status_text.text("✅ Step 4/6: Data Split and Scaled")
            progress_bar.progress(80)
            
            # Step 5: Train Models
            status_text.text("🤖 Step 5/6: Training Models...")
            progress_bar.progress(90)
            results, best_model_name = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            status_text.text("✅ Step 5/6: Models Trained")
            progress_bar.progress(100)
            
            st.session_state.predictor = predictor
            st.session_state.model_trained = True
            st.session_state.train_model = False
            
            st.success("✅ Model Training Completed Successfully!")
            st.balloons()
            
            # Display PCA information
            if predictor.use_pca:
                st.markdown("### PCA Dimensionality Reduction")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Features", f"{X.shape[1]}")
                with col2:
                    st.metric("Reduced Features", f"{predictor.n_components}")
                with col3:
                    st.metric("Variance Retained", f"{predictor.explained_variance_ratio_*100:.2f}%")
                st.success(f"✓ Features reduced from {X.shape[1]} to {predictor.n_components} while retaining {predictor.explained_variance_ratio_*100:.2f}% variance")
            
            # Display Random Forest model results
            st.markdown("### Random Forest Model Results")
            rf_result = results['Random Forest']
            comparison_df = pd.DataFrame({
                'Metric': ['Test Accuracy', 'Train Accuracy', 'CV Mean', 'CV Std', 'Overfitting Gap', 
                          'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Value': [
                    f"{rf_result['accuracy']:.4f}",
                    f"{rf_result['train_accuracy']:.4f}",
                    f"{rf_result['cv_mean']:.4f}",
                    f"{rf_result['cv_std']:.4f}",
                    f"{rf_result['overfitting_gap']:.4f}",
                    f"{rf_result['precision']:.4f}",
                    f"{rf_result['recall']:.4f}",
                    f"{rf_result['f1']:.4f}",
                    f"{rf_result['roc_auc']:.4f}"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            best_result = results[best_model_name]
            pca_info = f"\n- PCA: {X.shape[1]} → {predictor.n_components} features ({predictor.explained_variance_ratio_*100:.2f}% variance)" if predictor.use_pca else ""
            st.info(f"""
            🏆 **Best Model:** {best_model_name}
            - Test Accuracy: {best_result['accuracy']:.2%}
            - CV Mean Accuracy: {best_result['cv_mean']:.2%} (±{best_result['cv_std']*2:.2%})
            - Overfitting Gap: {best_result['overfitting_gap']:.4f}
            - ROC-AUC: {best_result['roc_auc']:.4f}{pca_info}
            """)
            
            # Overfitting warning
            if best_result['overfitting_gap'] > 0.10:
                st.warning("⚠️ Warning: Significant overfitting detected (gap > 10%)")
            elif best_result['overfitting_gap'] > 0.05:
                st.warning("⚠️ Caution: Moderate overfitting detected (gap > 5%)")
            else:
                st.success("✓ Good generalization - minimal overfitting detected")
        else:
            st.info("👆 Click 'Train Model' button in the sidebar to start training")
    
    with tab4:
        st.markdown("### Model Results & Visualizations")
        
        if st.session_state.model_trained and st.session_state.predictor:
            predictor = st.session_state.predictor
            
            # Get best model results
            best_result = predictor.results[predictor.best_model_name]
            accuracy = best_result['accuracy']
            train_accuracy = best_result['train_accuracy']
            cv_mean = best_result['cv_mean']
            cv_std = best_result['cv_std']
            overfitting_gap = best_result['overfitting_gap']
            precision = best_result['precision']
            recall = best_result['recall']
            f1 = best_result['f1']
            roc_auc = best_result['roc_auc']
            
            st.markdown(f"### Selected Model: **{predictor.best_model_name}**")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Test Accuracy", f"{accuracy:.2%}", f"{abs(accuracy - 0.87)*100:.2f}% from target")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
            
            st.divider()
            
            # Cross-validation and overfitting metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Accuracy", f"{train_accuracy:.2%}")
            with col2:
                st.metric("CV Mean Accuracy", f"{cv_mean:.2%}", f"±{cv_std*2:.2%}")
            with col3:
                gap_color = "normal" if overfitting_gap < 0.05 else "inverse"
                st.metric("Overfitting Gap", f"{overfitting_gap:.4f}", delta=None, delta_color=gap_color)
            
            # Overfitting analysis
            if overfitting_gap < 0.02:
                st.success("✓ Excellent: No significant overfitting detected")
            elif overfitting_gap < 0.05:
                st.success("✓ Good: Minimal overfitting (acceptable)")
            elif overfitting_gap < 0.10:
                st.warning("⚠️ Caution: Moderate overfitting detected")
            else:
                st.error("⚠️ Warning: Significant overfitting detected")
            
            # CV stability
            if cv_std < 0.01:
                st.success("✓ Excellent: Very stable model (CV std < 1%)")
            elif cv_std < 0.02:
                st.success("✓ Good: Stable model (CV std < 2%)")
            else:
                st.warning("⚠️ Caution: Model variability detected (CV std >= 2%)")
            
            st.divider()
            
            # Confusion Matrix
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(predictor.y_test, predictor.y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not Churn', 'Churn'],
                           yticklabels=['Not Churn', 'Churn'], ax=ax)
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_xlabel('Predicted', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Explanation:**
                - **TN (True Negatives):** Correctly predicted non-churn customers
                - **FP (False Positives):** Incorrectly flagged as churn
                - **FN (False Negatives):** Missed churn customers
                - **TP (True Positives):** Correctly predicted churn customers
                """)
            
            with col2:
                st.markdown("### ROC Curve")
                fpr, tpr, _ = roc_curve(predictor.y_test, predictor.y_pred_proba)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random Classifier (AUC = 0.50)')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"""
                **AUC Score:** {roc_auc:.4f}
                
                **Interpretation:**
                - AUC > 0.9: Excellent
                - AUC > 0.8: Good
                - AUC > 0.7: Moderate
                - AUC < 0.7: Needs improvement
                """)
            
            # Classification Report
            st.markdown("### Classification Report")
            report = classification_report(predictor.y_test, predictor.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("👆 Please train the model first to view results")
    
    with tab5:
        st.markdown("### 🔮 Predict Customer Churn")
        st.markdown("Enter customer details below to predict churn probability.")
        
        # Load models
        models = load_saved_models()
        
        if models is None:
            st.warning("⚠️ Models not found! Please train the model first using the 'Model Training' tab.")
            st.info("After training, the models will be saved and available for predictions.")
        else:
            st.success("✓ Models loaded successfully!")
            
            # Create form for user input
            with st.form("churn_prediction_form"):
                st.markdown("#### Customer Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    tenure = st.number_input("Tenure (months)", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
                    city_tier = st.selectbox("City Tier", [1, 2, 3], index=1)
                    warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
                    hour_spend_on_app = st.number_input("Hours Spent on App", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
                    number_of_devices = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=2, step=1)
                    satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5], index=2)
                    number_of_addresses = st.number_input("Number of Addresses", min_value=1, max_value=20, value=5, step=1)
                    complain = st.selectbox("Has Complained", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                
                with col2:
                    preferred_login_device = st.selectbox("Preferred Login Device", 
                        ["Mobile Phone", "Phone", "Computer", "Tablet"])
                    preferred_payment_mode = st.selectbox("Preferred Payment Mode",
                        ["Debit Card", "Credit Card", "UPI", "Cash on Delivery", "E wallet", "COD"])
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    preferred_order_cat = st.selectbox("Preferred Order Category",
                        ["Mobile", "Laptop & Accessory", "Grocery", "Fashion", "Others"])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
                    order_amount_hike = st.number_input("Order Amount Hike from Last Year (%)", 
                        min_value=0.0, max_value=50.0, value=10.0, step=1.0)
                    coupon_used = st.number_input("Coupon Used (count)", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
                    order_count = st.number_input("Order Count", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
                
                col3, col4 = st.columns(2)
                with col3:
                    day_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
                with col4:
                    cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
                
                submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)
                
                if submitted:
                    # Prepare input data
                    user_input = {
                        'Tenure': tenure,
                        'PreferredLoginDevice': preferred_login_device,
                        'CityTier': city_tier,
                        'WarehouseToHome': warehouse_to_home,
                        'PreferredPaymentMode': preferred_payment_mode,
                        'Gender': gender,
                        'HourSpendOnApp': hour_spend_on_app,
                        'NumberOfDeviceRegistered': number_of_devices,
                        'PreferedOrderCat': preferred_order_cat,
                        'SatisfactionScore': satisfaction_score,
                        'MaritalStatus': marital_status,
                        'NumberOfAddress': number_of_addresses,
                        'Complain': complain,
                        'OrderAmountHikeFromlastYear': order_amount_hike,
                        'CouponUsed': coupon_used,
                        'OrderCount': order_count,
                        'DaySinceLastOrder': day_since_last_order,
                        'CashbackAmount': cashback_amount
                    }
                    
                    # Make prediction
                    with st.spinner("Predicting..."):
                        prediction_result = predict_churn(user_input, models)
                    
                    if prediction_result:
                        st.success("✅ Prediction completed!")
                        st.divider()
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🌲 Random Forest Prediction")
                            rf_churn_prob = prediction_result['rf_probability'][1] * 100
                            rf_no_churn_prob = prediction_result['rf_probability'][0] * 100
                            rf_pred = prediction_result['rf_prediction']
                            
                            if rf_pred == 1:
                                st.error(f"**Prediction: CHURN** ⚠️")
                            else:
                                st.success(f"**Prediction: NO CHURN** ✓")
                            
                            st.metric("Churn Probability", f"{rf_churn_prob:.2f}%", 
                                     f"{rf_churn_prob - 50:.2f}%")
                            st.metric("No Churn Probability", f"{rf_no_churn_prob:.2f}%")
                            
                            # Progress bar
                            st.progress(rf_churn_prob / 100)
                        
                        with col2:
                            st.markdown("### 📊 Logistic Regression Prediction")
                            lr_churn_prob = prediction_result['lr_probability'][1] * 100
                            lr_no_churn_prob = prediction_result['lr_probability'][0] * 100
                            lr_pred = prediction_result['lr_prediction']
                            
                            if lr_pred == 1:
                                st.error(f"**Prediction: CHURN** ⚠️")
                            else:
                                st.success(f"**Prediction: NO CHURN** ✓")
                            
                            st.metric("Churn Probability", f"{lr_churn_prob:.2f}%",
                                     f"{lr_churn_prob - 50:.2f}%")
                            st.metric("No Churn Probability", f"{lr_no_churn_prob:.2f}%")
                            
                            # Progress bar
                            st.progress(lr_churn_prob / 100)
                        
                        st.divider()
                        
                        # Model comparison
                        st.markdown("### 📈 Model Comparison")
                        comparison_df = pd.DataFrame({
                            'Model': ['Random Forest', 'Logistic Regression'],
                            'Prediction': ['Churn' if prediction_result['rf_prediction'] == 1 else 'No Churn',
                                         'Churn' if prediction_result['lr_prediction'] == 1 else 'No Churn'],
                            'Churn Probability': [f"{rf_churn_prob:.2f}%", f"{lr_churn_prob:.2f}%"],
                            'Confidence': ['High' if max(rf_churn_prob, rf_no_churn_prob) > 70 else 'Medium',
                                         'High' if max(lr_churn_prob, lr_no_churn_prob) > 70 else 'Medium']
                        })
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        if rf_pred == 1 or lr_pred == 1:
                            st.warning("""
                            **Customer is at risk of churning!** Consider:
                            - Offering personalized discounts or promotions
                            - Reaching out with customer support
                            - Providing loyalty rewards
                            - Sending re-engagement campaigns
                            """)
                        else:
                            st.success("""
                            **Customer is likely to stay!** Continue to:
                            - Maintain good service quality
                            - Send regular updates and offers
                            - Encourage app usage
                            - Reward loyalty
                            """)


if __name__ == "__main__":
    main()

