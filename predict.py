import pandas as pd
import joblib

# Load the saved pipeline (preprocessing + model, together)
model = joblib.load("models/car_price_model.pkl")

# Describe a new car you want a price for
new_car = pd.DataFrame([{
    "brand": "bmw",
    "model": "3 Series",
    "age": 2,
    "mileage": 25000,
    "transmission": "Automatic",
    "fuelType": "Diesel",
    "tax": 145,
    "mpg": 50.4,
    "engineSize": 2.0,
}])

predicted_price = model.predict(new_car)[0]
print(f"Predicted price: £{predicted_price: ,.2f}")