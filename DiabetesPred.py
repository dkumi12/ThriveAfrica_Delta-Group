import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (adjust the file path as needed)
df = pd.read_csv(r'C:\Users\abami\OneDrive\Desktop\DeltaGroup\diabetes.csv')  # Use raw string for Windows paths

# Data Cleaning: Replace zeros with median for specific columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())

# Exploratory Data Analysis (EDA)
# Check class distribution
print("Class distribution:")
print(df['Outcome'].value_counts(normalize=True))

# Feature Correlation Heatmap (numerical columns only)
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Feature Distributions (numerical columns only)
df.select_dtypes(include=[np.number]).hist(figsize=(12, 8), bins=15)
plt.tight_layout()
plt.show()

# Feature Engineering
# Glucose/Insulin Ratio (handle division by zero)
df['Glucose_Insulin_Ratio'] = df['Glucose'] / df['Insulin']
df['Glucose_Insulin_Ratio'] = df['Glucose_Insulin_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# BMI x Age
df['BMI_Age'] = df['BMI'] * df['Age']

# Obese indicator (BMI ≥ 30)
df['Obese'] = (df['BMI'] >= 30).astype(int)

# Age Group (categorical)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                         labels=['<25', '25-40', '40-60', '>60']).astype('category')

# High Blood Pressure (BP ≥ 80)
df['High_BP'] = (df['BloodPressure'] >= 80).astype(int)

# SkinThickness/BMI Ratio
df['SkinThickness_BMI_Ratio'] = df['SkinThickness'] / df['BMI']

# Train-Test Split
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Define numerical and categorical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = ['Age_Group']

# Create a preprocessing pipeline with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),         # Scale numerical columns
        ('cat', OneHotEncoder(drop='first'), cat_cols)  # One-hot encode categorical columns
    ])

# Apply preprocessing to training and test data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Model Training: Logistic Regression
model_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]  # Probabilities for AUC
y_pred_lr = model_lr.predict(X_test_scaled)                  # Binary predictions

# Evaluation: ROC AUC
print("Logistic Regression AUC:", roc_auc_score(y_test, y_pred_prob_lr))