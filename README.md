# House Price Prediction

A machine learning project that predicts house prices using the King County House Sales dataset.

## Project Overview

This project uses supervised machine learning to estimate the price of a home based on features such as:

- Living space
- Home quality
- Location
- Number of bedrooms and bathrooms
- Lot size
- Year built
- Waterfront status
- View quality

The dataset contains house sales from King County, Washington.

## Data Cleaning

The dataset was checked for invalid values and unusual records.

The following rows were removed:

- Rows with duplicate data
- Houses with invalid or zero values for important property features
- One extreme bedroom-count anomaly

After cleaning, the dataset contained **21,596 houses**.

## Models

The following regression models were tested:

- Linear Regression
- Decision Tree
- Random Forest
- Tuned Random Forest
- Gradient Boosting

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | $97,309 | $162,152 | 0.799 |
| Decision Tree | $99,300 | $184,990 | 0.739 |
| Random Forest | $67,444 | $122,406 | 0.886 |
| Tuned Random Forest | $67,381 | $123,082 | 0.884 |
| Gradient Boosting | $76,265 | $124,629 | 0.881 |

## Prediction Analysis

Prediction errors varied depending on the price range of the home.

The model generally produced smaller percentage errors for homes in the middle price ranges, while lower- and higher-priced homes showed wider variation.

The model also showed that features such as home quality, living space, and location were important when making predictions.

## Streamlit App

The project includes a Streamlit web application that allows users to enter information about a home and receive an estimated price.

The app also provides:

- An estimated price range
- A historical error estimate
- A +500 sqft what-if comparison
- A feature importance chart

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit


## Project Structure

```text
house-price-prediction/
├── data/
│   └── kc_house_data.csv
├── graphs/
├── notebooks/
├── src/
│   ├── app.py
│   └── housepricemodel.py
├── README.md
├── requirements.txt
└── .gitignore
