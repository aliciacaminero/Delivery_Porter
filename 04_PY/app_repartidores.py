import joblib
import streamlit as st
import pandas as pd

# Cargar el modelo
model = joblib.load("03_PKL/calculo_repartidores.pkl")

# Función para calcular la densidad de repartidores ajustada por tipo de restaurante
def calculate_partner_density(total_outstanding_orders, order_hour, grouped_category):
    # Definir un factor de ajuste por tipo de restaurante
    adjustment_factors = {
        "American": 1.2,    # Ejemplo: más pedidos por hora en restaurantes americanos
        "Asian": 1.1,
        "Beverages": 0.9,   # Ejemplo: menos pedidos por hora en bebidas
        "Desserts": 1.0,    # Neutral, no se ajusta
        "European": 1.3,
        "Fast Food": 1.4,   # Ejemplo: más pedidos en restaurantes de comida rápida
        "Healthy": 1.1,
        "Indian": 1.2,
        "Italian": 1.0,
        "Latin": 1.1,
        "Mediterranean": 1.0,
        "Mexican": 1.2,
        "Other": 1.0
    }

    # Factor de ajuste para el tipo de restaurante
    adjustment_factor = adjustment_factors.get(grouped_category, 1.0)

    # Ajuste de la densidad de repartidores con base en el tipo de restaurante
    return total_outstanding_orders / (order_hour + 1) * adjustment_factor  # Ejemplo de ajuste (ajustar según el modelo)

# Función para la predicción
def predict_repartidores(order_hour, grouped_category, total_outstanding_orders, model):
    # Calcular partner_density antes de hacer la predicción
    partner_density = calculate_partner_density(total_outstanding_orders, order_hour, grouped_category)
    
    # Crear el dataframe para la predicción con la columna 'partner_density' calculada
    example_data = pd.DataFrame([[order_hour, grouped_category, total_outstanding_orders, partner_density]],
                                columns=['order_hour', 'grouped_category', 'total_outstanding_orders', 'partner_density'])
    
    # Predecir el número de repartidores
    repartidores_pred = model.predict(example_data)
    
    # Redondear el resultado a un número entero
    return round(repartidores_pred[0])

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
    st.write(f"El número estimado de repartidores necesarios es: {predicted_repartidores}")

    # Mostrar el icono del repartidor según el número estimado
    repartidor_icon = "🛵"  # Icono de repartidor
    st.write(f"Iconos de repartidores: {repartidor_icon * predicted_repartidores}")