import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from geopy.distance import geodesic
import joblib

# Load your trained model
model = load_model('ETA/enhanced_dl_model2.h5')  # Replace with the path to your model

# Load the scaler
scaler = joblib.load('ETA/scaler.pkl')

# Streamlit app title
st.title("ETA Prediction for Mumbai Routes")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    # Input sliders for coordinates
    start_lat = st.slider("Start Latitude", 18.87, 19.30, 19.0)
    start_lon = st.slider("Start Longitude", 72.77, 72.98, 72.85)
    end_lat = st.slider("End Latitude", 18.87, 19.30, 19.1)
    end_lon = st.slider("End Longitude", 72.77, 72.98, 72.90)

    # Number input for turns and traffic lights
    num_turns = st.number_input("Number of Turns", min_value=0, max_value=100, value=5)
    num_traffic_lights = st.number_input("Number of Traffic Lights", min_value=0, max_value=200, value=5)

with col2:
    # Number input for speed
    avg_speed = st.number_input("Average Speed (km/h)", min_value=10, max_value=100, value=50)

    # Dropdown for Time of Day
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    time_of_day_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    time_of_day = time_of_day_mapping[time_of_day]

    # Dropdown for Day of Week
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    day_of_week_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_of_week = day_of_week_mapping[day_of_week]

# Calculate distance between start and end points
def calculate_distance(start_lat, start_lon, end_lat, end_lon):
    start_coords = (start_lat, start_lon)
    end_coords = (end_lat, end_lon)
    return geodesic(start_coords, end_coords).km

distance_km = calculate_distance(start_lat, start_lon, end_lat, end_lon)

# Display the calculated distance
st.write(f"Calculated Distance: {distance_km:.2f} km")

# Simulate traffic conditions
def simulate_traffic_conditions(time_of_day, day_of_week):
    if day_of_week in [0, 1, 2, 3, 4]:  # Weekdays
        if time_of_day in [0, 2]:  # Peak hours
            return np.random.uniform(1.5, 2)
        else:  # Off-peak hours
            return np.random.uniform(1.0, 1.5)
    else:  # Weekends
        if time_of_day in [0, 2]:  # Peak hours
            return np.random.uniform(1.0, 1.5)
        else:  # Off-peak hours
            return np.random.uniform(0.5, 0.8)

traffic_factor = simulate_traffic_conditions(time_of_day, day_of_week)

# Calculate additional features
route_complexity_score = num_turns + num_traffic_lights
distance_time_interaction = distance_km / traffic_factor

# Create the feature array for prediction
features = np.array([[distance_km, num_turns, num_traffic_lights, avg_speed, traffic_factor, route_complexity_score, distance_time_interaction]])

# Standardize features
features_scaled = scaler.transform(features)  # Use the same scaler

# Predict ETA
with col2:

    if st.button("Predict ETA"):
        eta_prediction = model.predict(features_scaled).flatten()[0]
        st.write(f"Estimated Time of Arrival (ETA): {eta_prediction:.2f} minutes")
