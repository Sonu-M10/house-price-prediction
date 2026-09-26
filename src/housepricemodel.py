# --------------------------------------------------
# Imports
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv("data/kc_house_data.csv")


# --------------------------------------------------
# Explore Original Data
# --------------------------------------------------

print(df.head())

print("\nOriginal shape:")
print(df.shape)

print("\nData information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nCorrelation with price:")
print(
    df.corr(numeric_only=True)["price"]
    .sort_values(ascending=False)
)


# --------------------------------------------------
# Clean Data
# --------------------------------------------------

print("\n--- Data Cleaning ---")

original_rows = len(df)

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

# Remove clearly invalid values
invalid_rows = (
    (df["price"] <= 0)
    | (df["sqft_living"] <= 0)
    | (df["sqft_lot"] <= 0)
    | (df["bedrooms"] <= 0)
    | (df["bathrooms"] <= 0)
    | (df["floors"] <= 0)
)

invalid_count = invalid_rows.sum()

df = df[~invalid_rows].copy()

# Remove the extreme 33-bedroom anomaly
bedroom_anomaly_count = (df["bedrooms"] > 15).sum()

df = df[df["bedrooms"] <= 15].copy()

cleaned_rows = len(df)

print("Duplicate rows removed:", duplicate_count)
print("Invalid rows removed:", invalid_count)
print("Bedroom anomaly rows removed:", bedroom_anomaly_count)
print("Original rows:", original_rows)
print("Cleaned rows:", cleaned_rows)
print("Total rows removed:", original_rows - cleaned_rows)


# --------------------------------------------------
# Explore Cleaned Data
# --------------------------------------------------

print("\n--- Cleaned Data ---")

print("Cleaned shape:")
print(df.shape)

print("\nCleaned descriptive statistics:")
print(df.describe())


# --------------------------------------------------
# Visualize Living Space vs. Price
# --------------------------------------------------

plt.scatter(
    df["sqft_living"],
    df["price"]
)

plt.xlabel("Living Space (sqft)")
plt.ylabel("Price")
plt.title("Living Space vs. House Price")

plt.savefig(
    "graphs/living_space_vs_price.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Prepare Features and Target
# --------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

# Remove columns that are not useful for prediction
X = X.drop(
    ["id", "date"],
    axis=1
)

# Convert ZIP codes into separate columns
X = pd.get_dummies(
    X,
    columns=["zipcode"]
)

print("\nFeature shape:")
print(X.shape)

print("Target shape:")
print(y.shape)

print("\nShape after encoding:")
print(X.shape)


# --------------------------------------------------
# Split Data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Train/Test Split ---")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# Train and Evaluate Models
# --------------------------------------------------

# Linear Regression
linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# Decision Tree
tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_predictions = tree_model.predict(X_test)

tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = mean_squared_error(
    y_test,
    tree_predictions
) ** 0.5

tree_r2 = r2_score(
    y_test,
    tree_predictions
)


# Random Forest
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(
    X_train,
    y_train
)

forest_predictions = forest_model.predict(X_test)

forest_mae = mean_absolute_error(
    y_test,
    forest_predictions
)

forest_rmse = mean_squared_error(
    y_test,
    forest_predictions
) ** 0.5

forest_r2 = r2_score(
    y_test,
    forest_predictions
)


# Tuned Random Forest
tuned_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

tuned_rf.fit(
    X_train,
    y_train
)

tuned_predictions = tuned_rf.predict(X_test)

tuned_rf_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rf_rmse = mean_squared_error(
    y_test,
    tuned_predictions
) ** 0.5

tuned_rf_r2 = r2_score(
    y_test,
    tuned_predictions
)


# Gradient Boosting
gradient_model = GradientBoostingRegressor(
    random_state=42
)

gradient_model.fit(
    X_train,
    y_train
)

gradient_predictions = gradient_model.predict(X_test)

gradient_mae = mean_absolute_error(
    y_test,
    gradient_predictions
)

gradient_rmse = mean_squared_error(
    y_test,
    gradient_predictions
) ** 0.5

gradient_r2 = r2_score(
    y_test,
    gradient_predictions
)


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        linear_mae,
        tree_mae,
        forest_mae,
        tuned_rf_mae,
        gradient_mae
    ],
    "RMSE": [
        linear_rmse,
        tree_rmse,
        forest_rmse,
        tuned_rf_rmse,
        gradient_rmse
    ],
    "R²": [
        linear_r2,
        tree_r2,
        forest_r2,
        tuned_rf_r2,
        gradient_r2
    ]
})

print("\n--- Model Comparison ---")
print(model_results)


# --------------------------------------------------
# Final Model
# --------------------------------------------------

final_model = tuned_rf
final_predictions = tuned_predictions


# --------------------------------------------------
# Prediction Error Analysis
# --------------------------------------------------

residuals = y_test - final_predictions

absolute_errors = abs(residuals)

percentage_errors = (
    absolute_errors / y_test
) * 100

print("\n--- Prediction Error Analysis ---")

print(
    "Median Error:",
    residuals.median()
)

print(
    "Mean Absolute Error:",
    absolute_errors.mean()
)

print(
    "Median Absolute Percentage Error:",
    percentage_errors.median()
)

print(
    "90th Percentile Absolute Percentage Error:",
    percentage_errors.quantile(0.90)
)

print(
    "95th Percentile Absolute Percentage Error:",
    percentage_errors.quantile(0.95)
)


# --------------------------------------------------
# Error by Price Range
# --------------------------------------------------

error_analysis = pd.DataFrame({
    "actual_price": y_test.values,
    "predicted_price": final_predictions,
    "absolute_error": absolute_errors.values,
    "percentage_error": percentage_errors.values
})

price_bins = [
    0,
    250000,
    500000,
    750000,
    1000000,
    float("inf")
]

price_labels = [
    "$0-$250k",
    "$250k-$500k",
    "$500k-$750k",
    "$750k-$1M",
    "$1M+"
]

error_analysis["price_range"] = pd.cut(
    error_analysis["actual_price"],
    bins=price_bins,
    labels=price_labels,
    right=False
)

error_by_price = (
    error_analysis
    .groupby(
        "price_range",
        observed=False
    )
    .agg(
        houses=("actual_price", "count"),
        median_error=("absolute_error", "median"),
        mean_error=("absolute_error", "mean"),
        median_percentage_error=(
            "percentage_error",
            "median"
        ),
        percentile_90_error=(
            "absolute_error",
            lambda x: x.quantile(0.90)
        ),
        percentile_90_percentage=(
            "percentage_error",
            lambda x: x.quantile(0.90)
        )
    )
)

print("\n--- Error by Price Range ---")
print(error_by_price)


# --------------------------------------------------
# Error by Price Range Graph
# --------------------------------------------------

plt.bar(
    error_by_price.index,
    error_by_price["median_error"]
)

plt.xlabel("Actual House Price")
plt.ylabel("Median Absolute Error ($)")
plt.title("Prediction Error by House Price Range")
plt.xticks(rotation=30)

plt.savefig(
    "graphs/error_by_price_range.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Percentage Error by Price Range Graph
# --------------------------------------------------

plt.bar(
    error_by_price.index,
    error_by_price["median_percentage_error"]
)

plt.xlabel("Actual House Price")
plt.ylabel("Median Absolute Percentage Error (%)")
plt.title("Percentage Error by House Price Range")
plt.xticks(rotation=30)

plt.savefig(
    "graphs/percentage_error_by_price_range.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# House Price Prediction Function
# --------------------------------------------------

def predict_house_price(house):

    prediction = final_model.predict(house)

    return prediction[0]


# --------------------------------------------------
# Model Comparison Graph
# --------------------------------------------------

plt.bar(
    model_results["Model"],
    model_results["R²"]
)

plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.ylim(0, 1)

plt.savefig(
    "graphs/model_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Actual vs. Predicted Prices
# --------------------------------------------------

plt.scatter(
    y_test,
    final_predictions
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.savefig(
    "graphs/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

feature_importance = pd.Series(
    final_model.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)

print("\n--- Top 10 Feature Importances ---")
print(feature_importance.head(10))


# --------------------------------------------------
# Feature Importance Graph
# --------------------------------------------------

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")

plt.savefig(
    "graphs/feature_importance.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Sample House Prediction
# --------------------------------------------------

sample_house = X_test.iloc[0:1]

predicted_price = predict_house_price(
    sample_house
)

actual_price = y_test.iloc[0]

print("\n--- House Price Prediction ---")

print(
    "Predicted Price: $",
    round(predicted_price, 2)
)

print(
    "Actual Price: $",
    actual_price
)


# --------------------------------------------------
# Price-Based Prediction Range
# --------------------------------------------------

def get_prediction_range(predicted_price):

    predicted_price_range = pd.cut(
        [predicted_price],
        bins=price_bins,
        labels=price_labels,
        right=False
    )[0]

    range_data = error_by_price.loc[
        predicted_price_range
    ]

    percentile_90_error = range_data[
        "percentile_90_error"
    ]

    lower_price = (
        predicted_price - percentile_90_error
    )

    upper_price = (
        predicted_price + percentile_90_error
    )

    return (
        predicted_price_range,
        max(0, lower_price),
        upper_price,
        percentile_90_error
    )


(
    sample_price_range,
    sample_lower,
    sample_upper,
    sample_error
) = get_prediction_range(
    predicted_price
)

print("\n--- Price-Based Prediction Range ---")

print(
    "Price Range Used:",
    sample_price_range
)

print(
    "90% Historical Error: $",
    round(sample_error, 2)
)

print(
    "Estimated Price Range: $",
    round(sample_lower, 2),
    "to $",
    round(sample_upper, 2)
)

print(
    "The range is based on historical prediction "
    "errors for houses with similar predicted prices."
)


# --------------------------------------------------
# What-If Analysis
# --------------------------------------------------

original_house = X_test.iloc[0:1].copy()

larger_house = original_house.copy()

larger_house["sqft_living"] = (
    larger_house["sqft_living"] + 500
)

original_prediction = predict_house_price(
    original_house
)

larger_prediction = predict_house_price(
    larger_house
)

difference = (
    larger_prediction - original_prediction
)

print("\n--- What-If Analysis ---")

print(
    "Original predicted price: $",
    round(original_prediction, 2)
)

print(
    "Predicted price with +500 sqft: $",
    round(larger_prediction, 2)
)

print(
    "Estimated price difference: $",
    round(difference, 2)
)