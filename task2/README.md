# Level 1 – Task 2: Data Cleaning & Preprocessing

## Description
This task focuses on cleaning and preprocessing a raw dataset to make it suitable for analysis and machine learning models.

## Dataset
House Prediction Data Set

## Tools & Libraries
- Python
- pandas
- numpy
- scikit-learn

## Data Cleaning Steps
- Loaded the raw dataset using pandas
- Handled missing values:
  - Numerical features filled with mean values
  - Categorical features filled with mode
- Detected and removed outliers using the IQR method
- Converted categorical variables into numerical format using One-Hot Encoding
- Standardized numerical features using StandardScaler

## Output
- `cleaned_house_data.csv`: Cleaned and preprocessed dataset ready for machine learning

## How to Run
```bash
python task2.py
