import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("iris_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Enter measurements to predict Iris species")

# Sidebar sliders
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width  = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width  = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Input array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

# Class names
classes = ["setosa", "versicolor", "virginica"]

if st.button("Predict 🌼"):
    st.success(f"Predicted Iris Species: {classes[prediction[0]]}")

st.write("---")
st.write("Built using Streamlit 🚀")