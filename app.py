import streamlit as st
import pandas as pd
import requests
import joblib

# Load trained model
model = joblib.load("rain_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️"
)

st.title("🌧️ Rainfall Prediction")
st.write(
    "Enter a location to predict the probability of rainfall tomorrow."
)

st.divider()

location = st.text_input(
    "Enter Location",
    placeholder="e.g. Kochi, Kerala"
)

if st.button("Predict Rainfall", use_container_width=True):

    if not location:
        st.warning("Please enter a location.")
        st.stop()

    # -------------------------------
    # Get coordinates from location
    # -------------------------------

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(
        geo_url,
        params=geo_params
    )

    if geo_response.status_code != 200:
        st.error("Unable to find the location.")
        st.stop()

    geo_data = geo_response.json()

    if "results" not in geo_data:
        st.error("Location not found.")
        st.stop()

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    # -------------------------------
    # Get current weather
    # -------------------------------

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation"
        ],
        "timezone": "auto"
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params
    )

    if weather_response.status_code != 200:
        st.error("Unable to fetch weather data.")
        st.stop()

    weather_data = weather_response.json()
    current = weather_data["current"]

    # -------------------------------
    # Display fetched weather
    # -------------------------------

    st.subheader("Current Weather")

    st.write(f"**Location:** {location}")
    st.write(
        f"**Temperature:** {current['temperature_2m']} °C"
    )
    st.write(
        f"**Humidity:** {current['relative_humidity_2m']}%"
    )
    st.write(
        f"**Pressure:** {current['surface_pressure']} hPa"
    )
    st.write(
        f"**Wind Speed:** {current['wind_speed_10m']} km/h"
    )
    st.write(
        f"**Cloud Cover:** {current['cloud_cover']}%"
    )

    # -------------------------------
    # Prepare model input
    # -------------------------------

    input_data = {}

    for feature in feature_columns:

        if feature == "RainToday":
            input_data[feature] = (
                1 if current["precipitation"] > 0 else 0
            )

        elif feature == "Temp3pm":
            input_data[feature] = current["temperature_2m"]

        elif feature == "Temp9am":
            input_data[feature] = current["temperature_2m"]

        elif feature == "Humidity3pm":
            input_data[feature] = current["relative_humidity_2m"]

        elif feature == "Humidity9am":
            input_data[feature] = current["relative_humidity_2m"]

        elif feature == "Pressure3pm":
            input_data[feature] = current["surface_pressure"]

        elif feature == "Pressure9am":
            input_data[feature] = current["surface_pressure"]

        elif feature == "WindSpeed3pm":
            input_data[feature] = current["wind_speed_10m"]

        elif feature == "WindSpeed9am":
            input_data[feature] = current["wind_speed_10m"]

        elif feature == "Cloud3pm":
            input_data[feature] = current["cloud_cover"]

        elif feature == "Cloud9am":
            input_data[feature] = current["cloud_cover"]

        elif feature == "Rainfall":
            input_data[feature] = current["precipitation"]

        elif feature == "MaxTemp":
            input_data[feature] = current["temperature_2m"]

        elif feature == "MinTemp":
            input_data[feature] = current["temperature_2m"]

        else:
            st.error(
                f"The model requires '{feature}', "
                "but the weather API does not currently provide it."
            )
            st.stop()

    input_df = pd.DataFrame([input_data])

    # Ensure exact feature order
    input_df = input_df[feature_columns]

    # -------------------------------
    # Prediction
    # -------------------------------

    probabilities = model.predict_proba(input_df)[0]

    rain_probability = probabilities[1] * 100

    st.divider()

    st.subheader("Prediction")

    if rain_probability >= 50:
        st.success(
            f"🌧️ Rain is likely tomorrow"
        )
    else:
        st.info(
            f"☀️ Rain is unlikely tomorrow"
        )

    st.metric(
        "Probability of Rain Tomorrow",
        f"{rain_probability:.2f}%"
    )