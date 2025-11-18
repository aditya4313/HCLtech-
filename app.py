"""
Customer Churn Prediction - Streamlit Web Application
A professional and minimal frontend for the ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import kagglehub
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
        self.label_encoders = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.results = {}
        
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
        """Train multiple models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.model = results[best_model_name]['model']
        self.y_pred = results[best_model_name]['y_pred']
        self.y_pred_proba = results[best_model_name]['y_pred_proba']
        self.results = results
        
        return results, best_model_name


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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Data Analysis", "🤖 Model Training", "📈 Results"])
    
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
            
            # Step 4: Split and Scale
            status_text.text("📊 Step 4/6: Splitting and Scaling Data...")
            progress_bar.progress(70)
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
            )
            X_train_scaled = predictor.scaler.fit_transform(X_train)
            X_test_scaled = predictor.scaler.transform(X_test)
            predictor.X_test = X_test
            predictor.y_test = y_test
            
            status_text.text("✅ Step 4/6: Data Split and Scaled")
            progress_bar.progress(80)
            
            # Step 5: Train Models
            status_text.text("🤖 Step 5/6: Training Models...")
            progress_bar.progress(90)
            results, best_model = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            status_text.text("✅ Step 5/6: Models Trained")
            progress_bar.progress(100)
            
            st.session_state.predictor = predictor
            st.session_state.model_trained = True
            st.session_state.train_model = False
            
            st.success("✅ Model Training Completed Successfully!")
            st.balloons()
            
            # Display model comparison
            st.markdown("### Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                'Precision': [results[m]['precision'] for m in results.keys()],
                'Recall': [results[m]['recall'] for m in results.keys()],
                'F1-Score': [results[m]['f1'] for m in results.keys()],
                'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
            })
            comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
            
            st.info(f"🏆 **Best Model:** {best_model} (ROC-AUC: {results[best_model]['roc_auc']:.4f})")
        else:
            st.info("👆 Click 'Train Model' button in the sidebar to start training")
    
    with tab4:
        st.markdown("### Model Results & Visualizations")
        
        if st.session_state.model_trained and st.session_state.predictor:
            predictor = st.session_state.predictor
            
            # Metrics
            accuracy = accuracy_score(predictor.y_test, predictor.y_pred)
            precision = precision_score(predictor.y_test, predictor.y_pred, average='weighted', zero_division=0)
            recall = recall_score(predictor.y_test, predictor.y_pred, average='weighted', zero_division=0)
            f1 = f1_score(predictor.y_test, predictor.y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(predictor.y_test, predictor.y_pred_proba)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
            
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


if __name__ == "__main__":
    main()

