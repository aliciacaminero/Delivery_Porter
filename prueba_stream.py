import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos
modelo_tiempo_entrega_path = os.path.abspath('03_PKL/m_tiempo_pedido_normal.pkl')
modelo_calculo_repartidores_path = os.path.abspath('03_PKL/calculo_repartidores.pkl')

try:
    modelo_tiempo_entrega = joblib.load(modelo_tiempo_entrega_path)
    modelo_calculo_repartidores = joblib.load(modelo_calculo_repartidores_path)
except Exception as e:
    st.error(f"Error cargando modelos: {e}")

# Función para calcular la densidad de repartidores ajustada
def calculate_partner_density(total_outstanding_orders, order_hour, grouped_category):
    adjustment_factors = {
        "American": 1.2, "Asian": 1.1, "Beverages": 0.9, "Desserts": 1.0, 
        "European": 1.3, "Fast Food": 1.4, "Healthy": 1.1, "Indian": 1.2, 
        "Italian": 1.0, "Latin": 1.1, "Mediterranean": 1.0, "Mexican": 1.2, 
        "Other": 1.0
    }
    adjustment_factor = adjustment_factors.get(grouped_category, 1.0)
    return total_outstanding_orders / (order_hour + 1) * adjustment_factor

# Función para predecir repartidores
def predict_repartidores(order_hour, grouped_category, total_outstanding_orders, model):
    partner_density = calculate_partner_density(total_outstanding_orders, order_hour, grouped_category)
    example_data = pd.DataFrame([{
        'order_hour': order_hour,
        'grouped_category': grouped_category,
        'total_outstanding_orders': total_outstanding_orders,
        'partner_density': partner_density
    }])
    repartidores_pred = model.predict(example_data)
    return round(repartidores_pred[0])

# Título de la app
st.title("Predicción de Tiempo de Entrega y Repartidores 🚚")

# Contenedor de entrada de usuario
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        store_primary_category = st.selectbox('Categoría de Tienda', [
            'American', 'Asian', 'Beverages', 'Desserts', 'European', 
            'Fast Food', 'Healthy', 'Indian', 'Italian', 'Latin', 
            'Mediterranean', 'Mexican', 'Other'
        ])
        order_day = st.selectbox('Día del Pedido', [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday', 'Sunday'
        ])
        order_hour = st.slider('Hora del Pedido', min_value=0, max_value=23, value=12)
    
    with col2:
        total_onshift_partners = st.number_input('Repartidores Disponibles', min_value=1, max_value=50, value=10)
        total_busy_partners = st.number_input('Repartidores Ocupados', min_value=0, max_value=50, value=5)
        total_outstanding_orders = st.number_input('Pedidos Pendientes', min_value=0, max_value=100, value=20)

# Botón para predecir
if st.button("Predecir"):
    try:
        # Cálculo de la densidad de repartidores
        partner_density = calculate_partner_density(total_outstanding_orders, order_hour, store_primary_category)
        
        # Crear DataFrame para la predicción del tiempo de entrega
        datos_tiempo = pd.DataFrame([{
            'store_primary_category': store_primary_category,
            'total_onshift_partners': total_onshift_partners,
            'total_busy_partners': total_busy_partners,
            'total_outstanding_orders': total_outstanding_orders,
            'order_day': order_day,
            'order_hour': order_hour
        }])

        # Predicción del tiempo de entrega
        prediccion_tiempo = modelo_tiempo_entrega.predict(datos_tiempo)

        # Predicción de repartidores
        prediccion_repartidores = predict_repartidores(
            order_hour, store_primary_category, total_outstanding_orders, modelo_calculo_repartidores
        )

        # Mostrar resultados
        st.subheader("Resultados de la Predicción")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Duración Estimada", f"{prediccion_tiempo[0]:.2f} minutos")
            st.metric("Categoría de Tienda", store_primary_category)
            st.metric("Repartidores Necesarios", prediccion_repartidores)
        
        with col2:
            st.metric("Repartidores Disponibles", total_onshift_partners)
            st.metric("Pedidos Pendientes", total_outstanding_orders)

        # Mostrar DataFrame de entrada
        st.subheader("Datos de Entrada")
        st.dataframe(datos_tiempo)

        # Gráfico de distribución simulado
        st.subheader("Distribución de Tiempos de Entrega")
        st.bar_chart(pd.DataFrame({
            'Tiempo de Entrega': np.random.normal(prediccion_tiempo[0], 5, 100)
        }))

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
