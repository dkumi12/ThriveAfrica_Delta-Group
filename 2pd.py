import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\diabetes.csv')

# KNN Imputation for zeros
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = KNNImputer(n_neighbors=5)
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# Feature Engineering
df['Glucose_Insulin_Ratio'] = df['Glucose'] / df['Insulin']
df['Glucose_Insulin_Ratio'] = df['Glucose_Insulin_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['BMI_Age'] = df['BMI'] * df['Age']
df['Obese'] = (df['BMI'] >= 30).astype(int)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                        labels=['<25', '25-40', '40-60', '>60']).astype('category')
df['High_BP'] = (df['BloodPressure'] >= 80).astype(int)
df['SkinThickness_BMI_Ratio'] = df['SkinThickness'] / df['BMI']
df['SkinThickness_BMI_Ratio'] = df['SkinThickness_BMI_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
df['Pregnancies_Age'] = df['Pregnancies'] * df['Age']
df['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405
df['HOMA_IR'] = df['HOMA_IR'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['Age_DiabetesPedigree'] = df['Age'] * df['DiabetesPedigreeFunction']
df['High_Glucose'] = (df['Glucose'] >= 140).astype(int)
df['Age_Group_Fine'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                              labels=['<30', '30-40', '40-50', '50-60', '>60']).astype('category')
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], 
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese']).astype('category')
df['Log_Insulin'] = np.log1p(df['Insulin'])
df['Log_DiabetesPedigree'] = np.log1p(df['DiabetesPedigreeFunction'])

# Cap large values in multiplication-based features
large_features = ['BMI_Age', 'Glucose_BMI', 'Insulin_Glucose', 'Pregnancies_Age', 
                 'HOMA_IR', 'Age_DiabetesPedigree']
for feature in large_features:
    df[feature] = np.clip(df[feature], a_min=None, a_max=1e6)
    df[feature] = np.log1p(df[feature])

# Drop redundant or low-importance features (based on previous RF feature importance)
df = df.drop(columns=['DiabetesPedigreeFunction', 'BMI_Age'])  # Dropping BMI_Age due to redundancy with Glucose_BMI

# Train-Test Split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = ['Age_Group', 'Age_Group_Fine', 'BMI_Category']

# Debugging: Check for infinite or NaN values in X_train
print("Checking for infinite or NaN values in X_train:")
print(X_train[num_cols].isna().sum())
print(np.isinf(X_train[num_cols]).sum())

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Define feature names for importance plotting
feature_names = num_cols + [f"{col}_{cat}" for col in cat_cols for cat in preprocessor.named_transformers_['cat'].categories_[cat_cols.index(col)][1:]]

# XGBoost with Hyperparameter Tuning
xgb = XGBClassifier(random_state=42, scale_pos_weight=2)  # Increase weight for positive class
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'reg_lambda': [1, 10]  # L2 regularization
}
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_xgb.fit(X_train_balanced, y_train_balanced)
print("Best XGBoost Parameters:", grid_search_xgb.best_params_)

# Train the best XGBoost model
best_xgb = grid_search_xgb.best_estimator_

# Cross-Validation AUC
cv_scores_xgb = cross_val_score(best_xgb, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
print("XGBoost CV AUC Scores:", cv_scores_xgb)
print("Mean CV AUC:", cv_scores_xgb.mean(), "±", cv_scores_xgb.std())

# Predictions
y_pred_prob_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Evaluation
print("XGBoost AUC:", roc_auc_score(y_test, y_pred_prob_xgb))
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix (Threshold = 0.5)')
plt.show()

# Test Thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    y_pred_thresh = (y_pred_prob_xgb >= thresh).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"Threshold: {thresh}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

# Feature Importance
feature_importance_xgb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_xgb.feature_importances_
})
feature_importance_xgb = feature_importance_xgb.sort_values(by='Importance', ascending=False)
print("XGBoost Feature Importance (Top 10):")
print(feature_importance_xgb.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_xgb.head(10))
plt.title('Top 10 Feature Importance (XGBoost)')
plt.show()