import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn

# Mostrar la versión de scikit-learn
st.sidebar.write(f"Versión de scikit-learn: {sklearn.__version__}")

# URL del archivo del modelo de tiempo de entrega
url_modelo_tiempo_entrega = 'http://s68-77.furanet.com/ironhack/m_tiempo_pedido_normal.pkl'

# Intentar cargar el modelo con manejo de errores específico
try:
    response_tiempo_entrega = requests.get(url_modelo_tiempo_entrega)
    if response_tiempo_entrega.status_code == 200:
        # Intentar cargar el modelo con configuración especial para versiones antiguas
        try:
            import pickle
            mejor_modelo = pickle.loads(response_tiempo_entrega.content)
            st.success('Modelo de tiempo de entrega cargado correctamente usando pickle.')
        except:
            try:
                mejor_modelo = joblib.load(BytesIO(response_tiempo_entrega.content))
                st.success('Modelo de tiempo de entrega cargado correctamente usando joblib.')
            except Exception as e:
                st.error(f'Error al cargar el modelo: {str(e)}')
                st.stop()
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

def preparar_datos_para_prediccion(datos):
    """
    Prepara los datos para la predicción asegurando el formato correcto.
    """
    # Convertir categorías de español a inglés
    datos['grouped_category'] = datos['grouped_category'].map(reverse_category_map)
    datos['order_day'] = datos['order_day'].map(reverse_day_map)
    
    # Asegurar el orden de las columnas
    columnas_deseadas = [
        'grouped_category', 'order_day', 'order_hour',
        'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders'
    ]
    
    # Verificar que todas las columnas necesarias estén presentes
    for col in columnas_deseadas:
        if col not in datos.columns:
            st.error(f'Falta la columna: {col}')
            return None
            
    return datos[columnas_deseadas]

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

# Botón de predicción en la barra lateral
if st.sidebar.button('Predecir Duración de Entrega del Pedido'):
    try:
        # Crear DataFrame de entrada
        datos_entrada = pd.DataFrame([{
            'grouped_category': store_primary_category,
            'order_day': order_day,
            'order_hour': order_hour,
            'total_onshift_partners': total_onshift_partners,
            'total_busy_partners': total_busy_partners,
            'total_outstanding_orders': total_outstanding_orders
        }])

        # Preparar datos
        datos_preparados = preparar_datos_para_prediccion(datos_entrada)
        
        if datos_preparados is not None:
            # Realizar predicción
            try:
                prediccion_tiempo = mejor_modelo.predict(datos_preparados)
                
                # Mostrar resultados
                st.subheader('Resultados de la Predicción')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    minutos = int(prediccion_tiempo[0] // 60)
                    segundos = int(prediccion_tiempo[0] % 60)
                    st.metric('Duración Estimada', f'{minutos} minutos {segundos} segundos')
                
                with col2:
                    st.metric('Repartidores Disponibles', total_onshift_partners)
                    st.metric('Pedidos Pendientes', total_outstanding_orders)

                # Mostrar datos utilizados
                st.subheader('Datos Utilizados para la Predicción')
                st.dataframe(datos_entrada)
                
            except Exception as e:
                st.error(f'Error en la predicción: {str(e)}')
                st.write('Datos preparados:')
                st.write(datos_preparados)

    except Exception as e:
        st.error(f'Error al procesar los datos: {str(e)}')