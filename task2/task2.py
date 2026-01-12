# ======================================================
# Level 1 - Task 2: Data Cleaning & Preprocessing
# Codveda Data Science Internship
# ======================================================

# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 2. File paths (CSV in same directory)
INPUT_PATH = "house Prediction Data Set.csv"
OUTPUT_PATH = "cleaned_house_data.csv"

# 3. Load dataset
print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully")
print("Initial shape:", df.shape)

# 4. Dataset overview
print("\nDataset info:")
df.info()

# 5. Handle missing values
print("\nHandling missing values...")

for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled")

# 6. Remove outliers using IQR (NUMERIC COLUMNS ONLY)
print("\nRemoving outliers using IQR method...")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
          (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Outliers removed")
print("Shape after outlier removal:", df.shape)

# 7. Encode categorical variables
print("\nEncoding categorical variables...")

df = pd.get_dummies(df, drop_first=True)

print("Categorical variables encoded")

# 8. Normalize numerical features
print("\nNormalizing numerical features...")

scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

print("Normalization completed")

# 9. Final check
print("\nFinal dataset info:")
df.info()

# 10. Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nCleaned dataset saved successfully as '{OUTPUT_PATH}'")
