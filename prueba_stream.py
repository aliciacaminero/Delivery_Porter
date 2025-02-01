import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import streamlit as st

# URL del modelo
url_modelo_tiempo_entrega = 'http://s68-77.furanet.com/ironhack/m_tiempo_pedido_normal.pkl'

# Cargar el modelo desde la URL
try:
    response_tiempo_entrega = requests.get(url_modelo_tiempo_entrega)
    if response_tiempo_entrega.status_code == 200:
        modelo_datos = BytesIO(response_tiempo_entrega.content)
        mejor_modelo = joblib.load(modelo_datos)  # Cargar el modelo correctamente
        st.success('Modelo de tiempo de entrega cargado correctamente.')
    else:
        st.error('No se pudo descargar el modelo de la URL proporcionada.')
        st.stop()
except Exception as e:
    st.error(f'Error en la conexión: {str(e)}')
    st.stop()

# Diccionarios de mapeo
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

day_map = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}

# Mapeo inverso para convertir de español a inglés
reverse_category_map = {v: k for k, v in category_map.items()}
reverse_day_map = {v: k for k, v in day_map.items()}

# Interfaz de usuario
st.title('Predicción de Tiempo de Entrega 🚚')

# Contenedor principal para parámetros de entrada
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        store_primary_category = st.selectbox('Categoría de Tienda', list(category_map.values()))
        order_day = st.selectbox('Día del Pedido', list(day_map.values()))
        order_hour = st.slider('Hora del Pedido', min_value=0, max_value=23, value=12)

    with col2:
        total_onshift_partners = st.number_input('Total Repartidores Disponibles', min_value=1, max_value=50, value=10)
        total_busy_partners = st.number_input('Total Repartidores Asignados', min_value=0, max_value=50, value=5)
        total_outstanding_orders = st.number_input('Pedidos Pendientes', min_value=0, max_value=100, value=20)

# Botón de predicción
if st.sidebar.button('Predecir Duración de Entrega del Pedido'):
    try:
        # Crear DataFrame con las columnas en el orden correcto
        columnas_modelo = ['grouped_category', 'order_day', 'order_hour', 
                           'total_onshift_partners', 'total_busy_partners', 
                           'total_outstanding_orders']
        
        datos_entrada = pd.DataFrame([[
            reverse_category_map[store_primary_category],  # Convertir de español a inglés
            reverse_day_map[order_day],
            order_hour,
            total_onshift_partners,
            total_busy_partners,
            total_outstanding_orders
        ]], columns=columnas_modelo)  # Asegurar que las columnas coincidan

        # Verificar que las columnas coincidan antes de predecir
        if set(datos_entrada.columns) != set(columnas_modelo):
            st.error("Error: Los nombres de columnas no coinciden con los esperados por el modelo.")
            st.write("Esperado:", columnas_modelo)
            st.write("Recibido:", list(datos_entrada.columns))
            st.stop()

        # Verificar si el modelo es un Pipeline válido
        if hasattr(mejor_modelo, "predict"):
            # Hacer la predicción con el modelo
            prediccion_tiempo = mejor_modelo.predict(datos_entrada)[0]

            # Convertir el tiempo a minutos y redondear
            minutos = max(1, round(float(prediccion_tiempo) / 60))

            # Mostrar resultado
            st.subheader('Resultados de la Predicción')
            st.metric('Tiempo Estimado de Entrega', f'{minutos} minutos')
        else:
            st.error("El modelo cargado no es un Pipeline válido.")

    except Exception as e:
        st.error(f'Error al procesar los datos: {str(e)}')
        st.write('Detalles del error:')
        st.write(e)