import pandas as pd

brand_files = {
    "audi": "audi.csv",
    "bmw": "bmw.csv",
    "ford": "ford.csv",
    "hyundai": "hyundi.csv",
    "mercedes": "merc.csv",
    "skoda": "skoda.csv",
    "toyota": "toyota.csv",
    "vauxhall": "vauxhall.csv",
    "volkswagen": "vw.csv",
}

all_dataframes = []

for brand_name, filename in brand_files.items():
    df = pd.read_csv(f"data/raw/{filename}")
    df = df.rename(columns={"tax(£)": "tax"})   # fix Hyundai's inconsistent column name
    df["brand"] = brand_name
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)

print(combined_df.shape)
print(combined_df.columns.tolist())
print(combined_df.isnull().sum())   # check for missing values in every column

print("Before cleaning:", combined_df.shape)

# 1. Remove exact duplicate rows
combined_df = combined_df.drop_duplicates()

# 2. Keep only plausible years (1996-2020, based on what we saw in value_counts)
combined_df = combined_df[(combined_df["year"] >= 1996) & (combined_df["year"] <= 2020)]

# 3. Keep only plausible prices (drop anything absurdly cheap, likely a data error)
combined_df = combined_df[(combined_df["price"] >= 500) & (combined_df["price"] <= 150000)]

# 4. Keep only plausible mileage (drop negative or impossibly high values)
combined_df = combined_df[(combined_df["mileage"] >= 0) & (combined_df["mileage"] <= 400000)]

print("After cleaning:", combined_df.shape)

# Create an 'age' feature: how many years old the car is
# We use 2020 as the reference point since that's the latest year in this dataset
combined_df["age"] = 2020 - combined_df["year"]

print(combined_df[["year", "age"]].head())
print(combined_df["age"].describe())


combined_df.to_csv("data/processed/cars_cleaned.csv", index=False)
print("Saved cleaned dataset:", combined_df.shape)