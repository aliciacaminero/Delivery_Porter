import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import streamlit as st
from sklearn.pipeline import Pipeline

# URL del archivo del modelo de tiempo de entrega
url_modelo_tiempo_entrega = 'http://s68-77.furanet.com/ironhack/m_tiempo_pedido_normal.pkl'

# Descargar el archivo del modelo de tiempo de entrega
response_tiempo_entrega = requests.get(url_modelo_tiempo_entrega)
if response_tiempo_entrega.status_code == 200:
    try:
        # Cargar el modelo entrenado (pipeline completo)
        mejor_modelo = joblib.load(BytesIO(response_tiempo_entrega.content))
        st.success('Modelo de tiempo de entrega cargado correctamente.')
    except Exception as e:
        st.error(f'Error al cargar el modelo de tiempo de entrega: {e}')
else:
    st.error('No se pudo cargar el modelo de tiempo de entrega desde la URL proporcionada.')

# Diccionario de mapeo de valores en inglés a español para 'grouped_category'
category_map = {
    'Italian': 'Italiana',
    'Mexican': 'Mexicana',
    'Fast Food': 'Comida Rápida',
    'American': 'Americana',
    'Asian': 'Asiática',
    'Mediterranean': 'Mediterránea',
    'Indian': 'India',
    'European': 'Europea',
    'Healthy': 'Saludable',
    'Drinks': 'Bebidas',
    'Other': 'Otros',
    'Desserts': 'Postres'
}

# Diccionario de mapeo de valores en inglés a español para 'order_day'
day_map = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}

# Función para transformar los datos de entrada
def transformar_datos(datos):
    # Rellenar los valores NaN en las columnas categóricas con un valor predeterminado
    datos['grouped_category'] = datos['grouped_category'].fillna('Desconocido')
    datos['order_day'] = datos['order_day'].fillna('Desconocido')
    
    # Mapear los valores de 'grouped_category' y 'order_day' a español
    datos['grouped_category'] = datos['grouped_category'].map(category_map).fillna('Desconocido')
    datos['order_day'] = datos['order_day'].map(day_map).fillna('Desconocido')
    
    # Asegurarnos de que las columnas estén presentes para el modelo
    return datos

# Título de la app
st.title('Predicción de Tiempo de Entrega 🚚')

# Contenedor principal para parámetros de entrada
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        store_primary_category = st.selectbox('Categoría de Tienda', [
            'Italiana', 'Mexicana', 'Comida Rápida', 'Americana', 'Asiática', 
            'Mediterránea', 'India', 'Europea', 'Saludable', 'Bebidas',
            'Otros', 'Postres'
        ])

        order_day = st.selectbox('Día del Pedido', [
            'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'
        ])

        order_hour = st.slider('Hora del Pedido', min_value=0, max_value=23, value=12)

    with col2:
        total_onshift_partners = st.number_input('Total Repartidores Disponibles', min_value=1, max_value=50, value=10)
        total_busy_partners = st.number_input('Total Repartidores Asignados', min_value=0, max_value=50, value=5)
        total_outstanding_orders = st.number_input('Pedidos Pendientes', min_value=0, max_value=100, value=20)

# Botón de predicción en la barra lateral
if st.sidebar.button('Predecir Duración de Entrega del Pedido'):
    try:
        # Crear DataFrame de entrada
        datos = pd.DataFrame([{
            'grouped_category': store_primary_category,
            'total_onshift_partners': total_onshift_partners,
            'total_busy_partners': total_busy_partners,
            'total_outstanding_orders': total_outstanding_orders,
            'order_day': order_day,
            'order_hour': order_hour
        }])

        # Transformar los datos de entrada
        datos_transformados = transformar_datos(datos)

        # Usar el pipeline completo para la predicción
        prediccion_tiempo = mejor_modelo.predict(datos_transformados)

        # Mostrar resultados
        st.subheader('Resultados de la Predicción')

        col1, col2 = st.columns(2)

        with col1:
            # Mostrar duración del pedido en formato "minutos:segundos"
            st.metric('Duración Estimada', f'{prediccion_tiempo[0] // 60} minutos {int(prediccion_tiempo[0] % 60)} segundos')

        with col2:
            st.metric('Repartidores Disponibles', total_onshift_partners)
            st.metric('Pedidos Pendientes', total_outstanding_orders)

        # Mostrar DataFrame de inputs
        st.subheader('Detalles del Pedido')
        st.dataframe(datos_transformados)
        
    except Exception as e:
        st.error(f'Error en la predicción: {e}')
