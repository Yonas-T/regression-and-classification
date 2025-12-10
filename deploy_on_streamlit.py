import streamlit as st
import joblib
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Load models
model = joblib.load('models/audio_emotion_classifier_random_forest.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
encoder = joblib.load('models/label_encoder.pkl')


st.title('Audio Emotion Classifier')

uploaded_file = st.file_uploader('Upload Audio File', type=['wav'])

if uploaded_file:
    prediction, probabilities = predict_emotion(
        uploaded_file, model, scaler, encoder
    )
    st.write(f'Predicted Emotion: {prediction}')
    st.bar_chart(probabilities)



# Title of the app
st.title("Sales Prediction Model")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Date input
selected_date = st.sidebar.date_input("Select a Date", datetime.date.today())

# Button to trigger prediction
if st.sidebar.button("Predict Sales"):
    # Process the date into features (example: convert to ordinal for simple regression)
    # You may need to adjust this based on how your model was trained (e.g., extract day, month, year, weekday, etc.)
    date_ordinal = selected_date.toordinal()  # Simple feature: date as ordinal number
    
    # Load your pre-trained model (assuming it's saved as 'sales_model.pkl')
    # Replace this with your actual model loading code
    try:
        model = joblib.load('models/retail_demand_prediction_gradient_boosting.pkl')
    except FileNotFoundError:
        # If no model file, use a dummy model for demonstration
        st.warning("Model file not found. Using a dummy linear regression model for demo.")
        # Dummy training data (replace with your actual training logic if needed)
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        X = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
        y = np.random.randint(100, 1000, size=365) + (X.flatten() * 0.1)  # Simulated sales increasing over time
        model = LinearRegression()
        model.fit(X, y)

    # Prepare input for prediction
    input_features = np.zeros(51)
    input_features[1] = selected_date.year
    input_features[2] = selected_date.month
    input_features[3] = selected_date.day
    input_features = np.array([input_features])  # Use the full feature array

    # Make prediction
    predicted_sales = model.predict(input_features)[0]

    # Display the result
    st.header("Predicted Sales")
    st.write(f"For the date **{selected_date}**, the predicted sales are: **${predicted_sales:.2f}**")