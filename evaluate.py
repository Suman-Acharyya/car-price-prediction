import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/processed/cars_cleaned.csv")

features = ["brand", "model", "age", "mileage", "transmission", "fuelType", "tax", "mpg", "engineSize"]
target = "price"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved trained model
model = joblib.load("models/car_price_model.pkl")

# Predict on the test set
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.2, s=10, color="royalblue")

max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5)

plt.xlabel("Actual Price (£)")
plt.ylabel("Predicted Price (£)")
plt.title("Random Forest: Actual vs. Predicted Car Prices")
plt.tight_layout()

plt.savefig("images/actual_vs_predicted.png", dpi=200)
plt.show()

print("Plot saved as actual_vs_predicted.png")