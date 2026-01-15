import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Load dataset (space-separated)
# =========================
DATA_PATH = "house Prediction Data Set.csv"

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
print("Dataset loaded successfully")

# Column names
df.columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT", "MEDV"
]

print(df.head())

# =========================
# Features and target
# =========================
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Linear Regression
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
