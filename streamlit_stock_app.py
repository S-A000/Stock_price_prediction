import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load the model (you can load it outside the function for efficiency)
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading
def load_model():
    model = keras.models.load_model('stock_price_m.h5')  # Load your model from .h5 file
    return model

# Set up the app title and description
st.title("Stock Price Prediction")
st.write("Input a date, click 'Predict', and view the predicted stock price along with a graph.")

# Date input widget in Streamlit
input_date = st.date_input("Select a date for prediction")

# Load model once the app runs
model = load_model()

# Button to trigger prediction
if st.button("Predict"):
    
    # Define your prediction function using the loaded model
    def predict_price(date, model):
        # Here you need to preprocess the input based on your model's expected input format
        # Convert date into the format your model expects (this depends on your model's input)
        # For now, we simulate a random input feature vector as an example:
        input_features = np.random.rand(1, 10)  # Example input, replace with real features
        prediction = model.predict(input_features)
        return prediction[0][0]

    # Call the prediction function with the selected date and model
    predicted_price = predict_price(input_date, model)

    # Display the predicted price
    st.write(f"Predicted price for {input_date}: ${predicted_price:.2f}")

    # Plotting the predicted price (replace with actual plotting logic)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)) + predicted_price)  # Mock data for graph

    # Display the plot
    st.pyplot(fig)
