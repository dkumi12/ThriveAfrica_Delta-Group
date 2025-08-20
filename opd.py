import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\diabetes.csv')

# Replace inf with NaN in the raw data to avoid seaborn warnings
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# --- Visual 1: Correlation Heatmap (Before Imputation and Feature Engineering) ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Original Features")
plt.tight_layout()
plt.savefig(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\correlation_heatmap.png')
plt.close()

# KNN Imputation for zeros
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = KNNImputer(n_neighbors=5)
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# --- Visual 2: Feature Distribution Histograms ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution of Key Numeric Features After Imputation")
for i, col in enumerate(cols_with_zeros):
    sns.histplot(df[col].replace([np.inf, -np.inf], np.nan), bins=30, kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(col)
axes[1, 2].axis('off')  # Hide extra subplot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\feature_distributions.png')
plt.close()

# Feature Engineering
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1e-6)
df['Glucose_Insulin_Ratio'] = np.clip(df['Glucose_Insulin_Ratio'], a_min=None, a_max=1e6)
df['Obese'] = (df['BMI'] >= 30).astype(int)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                         labels=['<25', '25-40', '40-60', '>60']).astype('category')
df['High_BP'] = (df['BloodPressure'] >= 80).astype(int)
df['SkinThickness_BMI_Ratio'] = df['SkinThickness'] / (df['BMI'] + 1e-6)
df['SkinThickness_BMI_Ratio'] = np.clip(df['SkinThickness_BMI_Ratio'], a_min=None, a_max=1e6)
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
df['Pregnancies_Age'] = df['Pregnancies'] * df['Age']
df['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405
df['HOMA_IR'] = np.clip(df['HOMA_IR'], a_min=None, a_max=1e6)
df['High_Glucose'] = (df['Glucose'] >= 140).astype(int)
df['Log_Insulin'] = np.log1p(df['Insulin'])

# Cap large values and apply log transformation
large_features = ['Glucose_BMI', 'Insulin_Glucose', 'Pregnancies_Age', 'HOMA_IR']
for feature in large_features:
    df[feature] = np.clip(df[feature], a_min=None, a_max=1e6)
    df[feature] = np.log1p(df[feature])

# Drop high-correlation and low-importance features
columns_to_drop = ['Insulin_Glucose', 'HOMA_IR', 'Pregnancies_Age', 'DiabetesPedigreeFunction', 'Log_Insulin']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Check for multicollinearity
num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Outcome']
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print("Features with high correlation (>0.8):", to_drop)

# Train-Test Split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = ['Age_Group']

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

# Hyperparameter Tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear', random_state=42), 
                           param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_balanced, y_train_balanced)
best_C = grid_search.best_params_['C']
print("Best C:", best_C)

# Train Logistic Regression with optimized parameters
lr = LogisticRegression(C=best_C, penalty='l2', solver='liblinear', random_state=42)
lr.fit(X_train_balanced, y_train_balanced)

# Cross-Validation AUC
cv_scores_lr = cross_val_score(lr, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
print("Logistic Regression CV AUC Scores:", cv_scores_lr)
print("Mean CV AUC:", cv_scores_lr.mean(), "±", cv_scores_lr.std())

# Predictions
y_pred_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (y_pred_prob_lr >= 0.5).astype(int)

# Evaluation
print("Logistic Regression AUC:", roc_auc_score(y_test, y_pred_prob_lr))
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")

# --- Visual 3: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\roc_curve.png')
plt.close()

# --- Visual 4: Feature Importance Bar Plot ---
feature_names = preprocessor.get_feature_names_out()
importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': np.abs(lr.coef_[0])})
importance = importance.sort_values('Coefficient', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
plt.title('Top 10 Feature Importance (Absolute Coefficients)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\feature_importance.png')
plt.close()

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix (Threshold = 0.5)')
plt.savefig(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\confusion_matrix.png')
plt.close()

# Save the model and preprocessor
joblib.dump(lr, r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\logistic_regression_model.pkl')
joblib.dump(preprocessor, r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\preprocessor.pkl')
print("Model and preprocessor saved successfully!")