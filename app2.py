import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox('Company', df['Company'].unique())
typename = st.selectbox('TypeName', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
opsys = st.selectbox('Operating System', df['OpSys'].unique())
ips = st.selectbox('IPS Display', [0, 1])
touchscreen = st.selectbox('Has Touchscreen', [0, 1])
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600',
    '2560x1440', '2304x1440'
])
cpu_new = st.selectbox('CPU', df['Cpu_new'].unique())
gpu_new = st.selectbox('GPU', df['Gpu_new'].unique())
memory_int = st.selectbox('Memory (int)', sorted(df['Memory_int'].unique()))

# Use the correct weight column - it's 'Weight' based on your screenshot
weight_value = float(df['Weight'].mean())
weight_new = st.number_input('Weight (kg)', min_value=0.0, value=weight_value, step=0.1)

screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.6)


def calculate_ppi(resolution, screen_size):
    x_res, y_res = map(int, resolution.split('x'))
    return ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size


ppi = calculate_ppi(resolution, screen_size)

if st.button('Predict Price'):
    # Prepare input in the same order as your training dataframe
    # Use 'Weight' instead of 'Weight_new' to match your model's training data
    input_df = pd.DataFrame([[
        company, typename, ram, opsys, ips, touchscreen,
        ppi, cpu_new, gpu_new, memory_int, weight_new
    ]], columns=[
        'Company', 'TypeName', 'Ram', 'OpSys', 'IPS', 'HasTouchscreen',
        'PPI', 'Cpu_new', 'Gpu_new', 'Memory_int', 'Weight'
    ])

    # Make prediction
    prediction = pipe.predict(input_df)[0]

    st.success(f"Predicted Price (log scale): {prediction:.2f}")

    # Convert from log scale to actual price
    st.info(f"Predicted Price: {np.exp(prediction):.0f}")

