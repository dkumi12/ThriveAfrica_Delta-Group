"""
Diabetes Prediction Model Training Script
==========================================
This script trains a logistic regression model to predict diabetes risk
using the Pima Indians Diabetes dataset.

Author: Delta Group - Thrive Africa
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve
)
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    """Load and clean the diabetes dataset."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Data Cleaning: Replace zeros with median for specific columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())
    
    print(f"Dataset loaded: {df.shape}")
    print("\nClass distribution:")
    print(df['Outcome'].value_counts(normalize=True))
    
    return df


def create_visualizations(df):
    """Create and save EDA visualizations."""
    print("Creating visualizations...")
    
    # Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig('../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for i, col in enumerate(numerical_cols[:9]):
        row, col_idx = divmod(i, 3)
        axes[row, col_idx].hist(df[col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col_idx].set_title(f'{col} Distribution')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def feature_engineering(df):
    """Create new features from existing ones."""
    print("Engineering features...")
    
    # Glucose/Insulin Ratio (handle division by zero)
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1e-6)
    df['Glucose_Insulin_Ratio'] = np.clip(df['Glucose_Insulin_Ratio'], 0, 1e6)
    
    # SkinThickness/BMI Ratio
    df['SkinThickness_BMI_Ratio'] = df['SkinThickness'] / (df['BMI'] + 1e-6)
    
    # Glucose * BMI (log-transformed)
    df['Glucose_BMI'] = np.log1p(df['Glucose'] * df['BMI'])
    
    # Age Group (categorical)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                            labels=['<25', '25-40', '40-60', '>60']).astype('category')
    
    print("Feature engineering completed.")
    return df


def train_model(X_train, X_test, y_train, y_test):
    """Train the logistic regression model."""
    print("Training model...")
    
    # Define preprocessing pipeline
    num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
    cat_cols = ['Age_Group']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Fit preprocessing pipeline
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train model
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Model Training Completed!")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, preprocessor, y_pred, y_pred_prob, auc_score


def create_model_visualizations(y_test, y_pred, y_pred_prob):
    """Create model performance visualizations."""
    print("Creating model performance visualizations...")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/precision_recall_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_model_artifacts(model, preprocessor):
    """Save trained model and preprocessor."""
    print("Saving model artifacts...")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save model and preprocessor
    joblib.dump(model, '../models/diabetes_model.pkl')
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    
    print("Model artifacts saved to ../models/")


def main():
    """Main training pipeline."""
    print("=== Diabetes Prediction Model Training Pipeline ===\n")
    
    # Load and clean data
    df = load_and_clean_data('../data/diabetes.csv')
    
    # Create EDA visualizations
    create_visualizations(df.copy())
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Train-test split
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train model
    model, preprocessor, y_pred, y_pred_prob, auc_score = train_model(
        X_train, X_test, y_train, y_test
    )
    
    # Create performance visualizations
    create_model_visualizations(y_test, y_pred, y_pred_prob)
    
    # Save model artifacts
    save_model_artifacts(model, preprocessor)
    
    print(f"\n=== Training Complete! ===")
    print(f"Final ROC-AUC Score: {auc_score:.4f}")
    print("Check the visualizations/ folder for charts and graphs.")
    print("Model saved to models/ folder.")


if __name__ == "__main__":
    main()
