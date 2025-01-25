import joblib
import streamlit as st
import pandas as pd

# Cargar el modelo
model = joblib.load("03_PKL/calculo_repartidores.pkl")


# Función para la predicción
def predict_repartidores(order_hour, grouped_category, total_outstanding_orders, model):
    # Crear el dataframe para la predicción
    example_data = pd.DataFrame([[order_hour, grouped_category, total_outstanding_orders]],
                                columns=['order_hour', 'grouped_category', 'total_outstanding_orders'])
    
    # Predecir el número de repartidores
    repartidores_pred = model.predict(example_data)
    
    return repartidores_pred[0]

# Configuración de la aplicación Streamlit
st.title("Predicción de Repartidores")
st.write("Esta aplicación predice el número de repartidores necesarios según el tipo de restaurante, la hora del pedido y los pedidos pendientes.")

# Inputs de usuario
order_hour = st.slider("Hora del pedido (0-23):", min_value=0, max_value=23)
grouped_category = st.selectbox("Selecciona el tipo de restaurante:", 
                               ["American", "Asian", "Beverages", "Desserts", "European", "Fast Food", 
                                "Healthy", "Indian", "Italian", "Latin", "Mediterranean", "Mexican", "Other"])
total_outstanding_orders = st.number_input("Pedidos pendientes:", min_value=0)

# Realizar la predicción
if st.button("Predecir número de repartidores"):
    predicted_repartidores = predict_repartidores(order_hour, grouped_category, total_outstanding_orders, model)
    st.write(f"El número estimado de repartidores necesarios es: {predicted_repartidores:.2f}")