import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# URL del archivo del modelo de tiempo de entrega
url_modelo_tiempo_entrega = 'http://s68-77.furanet.com/ironhack/m_tiempo_pedido_normal.pkl'

# Descargar el archivo del modelo de tiempo de entrega
response_tiempo_entrega = requests.get(url_modelo_tiempo_entrega)
if response_tiempo_entrega.status_code == 200:
    try:
        # Cargar el modelo entrenado (pipeline completo)
        mejor_modelo = joblib.load(BytesIO(response_tiempo_entrega.content))
        #st.success('Modelo de tiempo de entrega cargado correctamente.')
    except Exception as e:
        st.error(f'Error al cargar el modelo de tiempo de entrega: {e}')
else:
    st.error('No se pudo cargar el modelo de tiempo de entrega desde la URL proporcionada.')

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

def transformar_datos(datos):
    # Convertir categorías de español a inglés para el modelo
    datos['grouped_category'] = datos['grouped_category'].map(reverse_category_map)
    datos['order_day'] = datos['order_day'].map(reverse_day_map)
    
    # Asegurarse de que todas las columnas necesarias estén presentes
    columnas_numericas = ['order_hour', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']
    columnas_categoricas = ['grouped_category', 'order_day']
    
    return datos[columnas_numericas + columnas_categoricas]

# Título de la app
st.title('Predicción de Tiempo de Entrega 🚚')

# Insertamos el bloque de CSS con st.markdown()
st.markdown("""
    <style>
    /* Estilo para el slider */
    div[data-testid="stSlider"] input[type="range"] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 5px;
        background-color: #0041f5;
        /* Barra del slider */
    }
    div[data-testid="stSlider"] input[type="range"]:focus {
        outline: none;
    }

    /* Estilo para el thumb del slider */
    div[data-testid="stSlider"] .stSlider__thumb input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: #0041f5;
        cursor: pointer;
        border: 2px solid #0041f5;
    }

    /* Estilo cuando el thumb está en hover */
    div[data-testid="stSlider"] .stSlider__thumb input[type="range"]:hover {
        background-color: #0031b0;
    }

    /* Valor del slider */
    div[data-testid="stSlider"] > div[role="slider"] > div[data-testid="stSliderThumbValue"] {
        color: #0041f5 !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)   

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
if st.button('Tiempo de Entrega Estimado'):
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

        # Transformar los datos
        datos_transformados = transformar_datos(datos)

        # Realizar la predicción
        prediccion_tiempo = mejor_modelo.predict(datos_transformados)

        # Mostrar resultados
        st.subheader('Tiempo de Entrega Estimado')

        # Convertir el tiempo de entrega a minutos
        total_minutos = int(prediccion_tiempo[0])
        horas = total_minutos // 60
        minutos = total_minutos % 60

        # Mostrar el tiempo en horas y minutos
        if horas > 0:
            st.metric('Duración Estimada', f'{horas} h {minutos} min')
        else:
            st.metric('Duración Estimada', f'{minutos} min')

    except Exception as e:
        st.error(f'Error en la predicción: {e}')
        st.error('Detalles del error para debugging:')
        st.write(datos_transformados.head())
