import joblib
import streamlit as st
import pandas as pd

# Cargar el modelo
model = joblib.load("03_PKL/calculo_repartidores.pkl")

# Función para calcular la densidad de repartidores
def calculate_partner_density(total_outstanding_orders, order_hour, grouped_category):
    # Esta es una estimación simple, ya que no sabemos exactamente cómo se calcula la densidad
    # Puedes mejorar este cálculo según los datos y el modelo entrenado
    return total_outstanding_orders / (order_hour + 1)  # Ejemplo de cálculo (ajustar según el modelo)

# Función para la predicción
def predict_repartidores(order_hour, grouped_category, total_outstanding_orders, model):
    # Calcular partner_density antes de hacer la predicción
    partner_density = calculate_partner_density(total_outstanding_orders, order_hour, grouped_category)
    
    # Crear el dataframe para la predicción con la columna 'partner_density' calculada
    example_data = pd.DataFrame([[order_hour, grouped_category, total_outstanding_orders, partner_density]],
                                columns=['order_hour', 'grouped_category', 'total_outstanding_orders', 'partner_density'])
    
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