# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Title and description of the app
st.title("Site Selection for Wind and Solar Farms")
st.markdown("""
This app demonstrates how machine learning can be used to optimize site selection for wind and solar farms.
You can adjust the parameters to see how they affect the prediction of whether a site is suitable or not.
""")

# Sidebar: Inputs for generating synthetic data
st.sidebar.header("Simulate New Data")
n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 10.0, 0.1)
solar_irradiance = st.sidebar.slider("Solar Irradiance (kWh/m²/day)", 0.0, 10.0, 5.0, 0.1)
terrain_slope = st.sidebar.slider("Terrain Slope (degrees)", 0.0, 30.0, 10.0, 0.1)
proximity_to_grid = st.sidebar.slider("Proximity to Grid (km)", 0, 100, 50)
environmental_impact = st.sidebar.slider("Environmental Impact (scale 0-10)", 0.0, 10.0, 5.0, 0.1)

# Simulate synthetic dataset
def generate_data(n_samples, wind_speed, solar_irradiance, terrain_slope, proximity_to_grid, environmental_impact):
    np.random.seed(42)
    wind_speed_data = np.random.uniform(2, 15, n_samples) + np.random.normal(0, 1, n_samples)
    solar_irradiance_data = np.random.uniform(2, 8, n_samples) + np.random.normal(0, 0.5, n_samples)
    terrain_slope_data = np.random.uniform(0, 20, n_samples) + np.random.normal(0, 1, n_samples)
    proximity_to_grid_data = np.random.uniform(0, 100, n_samples) + np.random.normal(0, 5, n_samples)
    environmental_impact_data = np.random.uniform(0, 10, n_samples) + np.random.normal(0, 1, n_samples)

    labels = ((wind_speed_data > 6) | (solar_irradiance_data > 5)) & (terrain_slope_data < 10) & (proximity_to_grid_data < 50)

    data = pd.DataFrame({
        'wind_speed': wind_speed_data,
        'solar_irradiance': solar_irradiance_data,
        'terrain_slope': terrain_slope_data,
        'proximity_to_grid': proximity_to_grid_data,
        'environmental_impact': environmental_impact_data,
        'suitable': labels.astype(int)
    })

    return data

# Generate synthetic data
data = generate_data(n_samples, wind_speed, solar_irradiance, terrain_slope, proximity_to_grid, environmental_impact)

# Split data into training and testing sets
X = data.drop('suitable', axis=1)
y = data['suitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")

st.write("### Classification Report")
report = classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

# Feature importance visualization
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices])
ax.set_xlabel('Relative Importance')
st.pyplot(fig)

# Interactive predictions with sliders
st.subheader("Try Custom Inputs")

user_wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, wind_speed, 0.1, key='wind_speed_slider')
user_solar_irradiance = st.slider("Solar Irradiance (kWh/m²/day)", 0.0, 10.0, solar_irradiance, 0.1, key='solar_irradiance_slider')
user_terrain_slope = st.slider("Terrain Slope (degrees)", 0.0, 30.0, terrain_slope, 0.1, key='terrain_slope_slider')
user_proximity_to_grid = st.slider("Proximity to Grid (km)", 0, 100, proximity_to_grid, key='proximity_to_grid_slider')
user_environmental_impact = st.slider("Environmental Impact (scale 0-10)", 0.0, 10.0, environmental_impact, 0.1, key='environmental_impact_slider')

# Make predictions based on user inputs
user_data = pd.DataFrame({
    'wind_speed': [user_wind_speed],
    'solar_irradiance': [user_solar_irradiance],
    'terrain_slope': [user_terrain_slope],
    'proximity_to_grid': [user_proximity_to_grid],
    'environmental_impact': [user_environmental_impact]
})

user_data_scaled = scaler.transform(user_data)
user_prediction = model.predict(user_data_scaled)

# Display prediction
if user_prediction[0] == 1:
    st.success("This site is **suitable** for a wind or solar farm!")
else:
    st.error("This site is **not suitable** for a wind or solar farm.")
