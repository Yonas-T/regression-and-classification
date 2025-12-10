import streamlit as st
import joblib

# Load models
model = joblib.load('models/audio_emotion_classifier_*.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
encoder = joblib.load('models/label_encoder.pkl')
regressor = joblib.load('models/retail_demand_prediction_gradient_boosting.pkl')


st.title('Audio Emotion Classifier')

uploaded_file = st.file_uploader('Upload Audio File', type=['wav'])

if uploaded_file:
    prediction, probabilities = predict_emotion(
        uploaded_file, model, scaler, encoder,regressor
    )
    st.write(f'Predicted Emotion: {prediction}')
    st.bar_chart(probabilities)