import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------
# Page Setup
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")

st.write(
    "Enter information about a home to get an estimated market value."
)

st.divider()


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv("data/kc_house_data.csv")


# --------------------------------------------------
# Prepare Data
# --------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

X = X.drop("id", axis=1)
X = X.drop("date", axis=1)

X = pd.get_dummies(X, columns=["zipcode"])


# --------------------------------------------------
# Train Model
# --------------------------------------------------

@st.cache_resource(show_spinner="🧠 Preparing the house price prediction model...")
def train_model(X, y):

    model = GradientBoostingRegressor(random_state=42)

    model.fit(X, y)

    return model


model = train_model(X, y)


# --------------------------------------------------
# Prediction Error Information
# --------------------------------------------------

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


error_data = {
    "$0-$250k": 97723.190312,
    "$250k-$500k": 93356.358897,
    "$500k-$750k": 138547.834465,
    "$750k-$1M": 201448.592540,
    "$1M+": 501261.347545
}

error_by_price = pd.Series(error_data)


def get_prediction_range(predicted_price):

    predicted_price_range = pd.cut(
        [predicted_price],
        bins=price_bins,
        labels=price_labels,
        right=False
    )[0]

    percentile_90_error = error_by_price.loc[
        predicted_price_range
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


# --------------------------------------------------
# Home Information
# --------------------------------------------------

st.subheader("🏡 Tell Us About the Home")

st.write(
    "Enter the details below. The model will use this information "
    "to estimate the home's value."
)

st.markdown("### 🏠 Basic Information")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input(
        "🛏️ Number of Bedrooms",
        0,
        10,
        3
    )

with col2:
    bathrooms = st.number_input(
        "🛁 Number of Bathrooms",
        0.0,
        8.0,
        2.0,
        step=0.25
    )

with col3:
    floors = st.number_input(
        "🏢 Number of Floors",
        1.0,
        3.5,
        1.0,
        step=0.5
    )


st.markdown("### 📐 Property Size")

col1, col2, col3 = st.columns(3)

with col1:
    sqft_living = st.number_input(
        "📐 Home Size (sqft)",
        290,
        14000,
        2000
    )

with col2:
    sqft_lot = st.number_input(
        "🌳 Lot Size (sqft)",
        500,
        900000,
        7500
    )

with col3:
    sqft_basement = st.number_input(
        "⬇️ Basement Size (sqft)",
        0,
        5000,
        0
    )


st.markdown("### ⭐ Home Quality")

col1, col2, col3 = st.columns(3)

with col1:
    grade = st.number_input(
        "⭐ Overall Home Quality",
        1,
        13,
        7,
        help="Higher numbers represent better construction and design quality."
    )

with col2:
    condition = st.number_input(
        "🏠 Home Condition",
        1,
        5,
        3,
        help="1 = Poor condition, 5 = Excellent condition"
    )

with col3:
    yr_built = st.number_input(
        "📅 Year Built",
        1900,
        2026,
        1980
    )


st.markdown("### 📍 Location & Features")

col1, col2, col3 = st.columns(3)

with col1:
    zipcode = st.selectbox(
        "📍 ZIP Code",
        sorted(df["zipcode"].unique())
    )

with col2:
    view = st.number_input(
        "👀 View Quality",
        0,
        4,
        0,
        help="0 = No view, 4 = Excellent view"
    )

with col3:
    waterfront = st.selectbox(
        "🌊 Waterfront Property?",
        ["No", "Yes"]
    )


st.divider()


# --------------------------------------------------
# Estimate Home Value
# --------------------------------------------------

if st.button(
    "💰 Estimate Home Value",
    use_container_width=True
):

    # Start with median values for features
    # the user is not entering
    house = X.median().to_frame().T

    # Get the typical latitude and longitude
    # for the selected ZIP code
    zip_location = df[
        df["zipcode"] == zipcode
    ][["lat", "long"]].median()

    house["lat"] = zip_location["lat"]
    house["long"] = zip_location["long"]

    # Replace median values with user inputs
    house["bedrooms"] = bedrooms
    house["bathrooms"] = bathrooms
    house["sqft_living"] = sqft_living
    house["sqft_lot"] = sqft_lot
    house["floors"] = floors
    house["grade"] = grade
    house["yr_built"] = yr_built
    house["view"] = view
    house["waterfront"] = 1 if waterfront == "Yes" else 0
    house["condition"] = condition
    house["sqft_basement"] = sqft_basement

    # Activate the selected ZIP code
    zipcode_column = "zipcode_" + str(zipcode)

    if zipcode_column in house.columns:
        house[zipcode_column] = True

    # Make prediction
    prediction = model.predict(house)[0]


    # --------------------------------------------------
    # Prediction Range
    # --------------------------------------------------

    (
        prediction_price_range,
        lower_price,
        upper_price,
        historical_error
    ) = get_prediction_range(
        prediction
    )


    st.subheader("💰 Estimated Home Value")

    st.success(f"${prediction:,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"Lower Range: ${lower_price:,.0f}"
        )

    with col2:
        st.error(
            f"Higher Range: ${upper_price:,.0f}"
        )

    st.caption(
        f"Based on historical prediction errors for homes "
        f"with predicted prices in the {prediction_price_range} range."
    )

    st.info(
        "This prediction is an estimate. The model can have significant "
        "prediction errors, especially for unusual or high-value homes."
    )


    # --------------------------------------------------
    # Try a Different Home
    # --------------------------------------------------

    st.subheader("🔮 Try a Different Home")

    larger_house = house.copy()

    larger_house["sqft_living"] = (
        larger_house["sqft_living"] + 500
    )

    larger_prediction = model.predict(larger_house)[0]

    difference = larger_prediction - prediction

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Home Value",
            f"${prediction:,.0f}"
        )

    with col2:
        st.metric(
            "With +500 sqft",
            f"${larger_prediction:,.0f}"
        )

    with col3:
        st.metric(
            "Estimated Change",
            f"${difference:,.0f}"
        )


    # --------------------------------------------------
    # What Matters Most?
    # --------------------------------------------------

    st.subheader("🧠 What Matters Most?")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    # Get top 8 features and convert importance to percentages
    top_features = importance.head(8).sort_values() * 100

    # Rename technical feature names
    feature_names = {
        "sqft_living": "Living Space",
        "grade": "Home Quality",
        "lat": "Latitude",
        "long": "Longitude",
        "waterfront": "Waterfront",
        "yr_built": "Year Built",
        "view": "View",
        "sqft_living15": "Nearby Home Size"
    }

    top_features.index = [
        feature_names.get(name, name)
        for name in top_features.index
    ]

    st.bar_chart(top_features)

    st.caption(
        "Higher percentages mean the model relied more heavily on that factor "
        "when estimating home values."
    )

    st.markdown("""
    **📖 How to read this graph**

    - **Living Space** → Size of the home itself
    - **Home Quality** → Overall construction and design quality
    - **Latitude** → North/south location
    - **Longitude** → East/west location
    - **Waterfront** → Whether the home has a waterfront view
    - **Year Built** → Age of the home
    - **View** → Quality of the property's view
    - **Nearby Home Size** → Average living space of nearby homes

    **Important:** These percentages show how much the model relied on each factor.
    They do **not** mean that a factor directly causes a higher home value.
    """)