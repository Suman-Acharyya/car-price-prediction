import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- 1. Load cleaned data ----
df = pd.read_csv("data/processed/cars_cleaned.csv")

features = ["brand", "model", "age", "mileage", "transmission", "fuelType", "tax", "mpg", "engineSize"]
target = "price"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 2. Define which columns are numeric vs categorical ----
numeric_features = ["age", "mileage", "tax", "mpg", "engineSize"]
categorical_features = ["brand", "model", "transmission", "fuelType"]

# ---- 3. Build a preprocessor that handles both types automatically ----
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---- 4. Build full pipelines: preprocessing + model, as ONE object ----
linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,          # caps how deep each tree can grow
        min_samples_leaf=2,    # requires at least 2 samples per leaf (less overfitting)
        random_state=42,
        n_jobs=-1
    )),
])

# ---- 5. A helper function so we don't repeat ourselves ----
def evaluate(pipeline, name):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name} -> R2: {round(r2,4)}  MAE: £{round(mae,2)}  RMSE: £{round(rmse,2)}")
    return pipeline

linear_pipeline = evaluate(linear_pipeline, "Linear Regression")
rf_pipeline = evaluate(rf_pipeline, "Random Forest")

# ---- 6. Save the WINNING pipeline (preprocessing + model, together) ----
joblib.dump(rf_pipeline, "models/car_price_model.pkl", compress=3)
print("Model saved")