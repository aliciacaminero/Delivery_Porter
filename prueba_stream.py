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
    from sklearn.preprocessing import LabelEncoder

    # Codificación de 'store_primary_category' con LabelEncoder
    encoder_category = LabelEncoder()
    datos['store_primary_category_encoded'] = encoder_category.fit_transform(datos['store_primary_category'])

    # Codificación de 'order_day' con LabelEncoder
    encoder_day = LabelEncoder()
    datos['order_day_encoded'] = encoder_day.fit_transform(datos['order_day'])

    # Crear 'is_high_duration' (ejemplo simple)
    datos['is_high_duration'] = datos['order_hour'] > 18  

    # Cálculo de partner_density
    datos['partner_density'] = datos['total_onshift_partners'] / (datos['total_outstanding_orders'] + 1)

    # Convertir columnas numéricas a formato correcto
    columnas_numericas = ['total_outstanding_orders', 'total_onshift_partners', 'total_busy_partners', 'order_hour']
    datos[columnas_numericas] = datos[columnas_numericas].apply(pd.to_numeric, errors='coerce')

    # Revisar que los datos siguen siendo un DataFrame
    if not isinstance(datos, pd.DataFrame):
        print("⚠ Error: datos_transformados no es un DataFrame")
        datos = pd.DataFrame(datos)

    print("✅ Datos transformados correctamente:")
    print(datos.dtypes)
    print(datos.head())

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

        # Verificar que es un DataFrame válido
            if not isinstance(datos_transformados, pd.DataFrame):
                st.error("Error: La transformación de datos no devolvió un DataFrame.")
                st.stop()

# Realizar predicción de tiempo de entrega
prediccion_tiempo = modelo_tiempo_entrega.predict(datos_transformados)


        # Crear un ColumnTransformer para manejar OneHotEncoding con 'handle_unknown="ignore"'
        preprocesador = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['grouped_category', 'order_day']),  # Codificación one-hot con manejo de categorías desconocidas
                ('num', 'passthrough', ['order_hour', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders'])  # Pasar columnas numéricas sin cambio
            ])

        # Asegurarse de que los datos sean un DataFrame antes de pasarlos al ColumnTransformer
        datos_transformados = preprocesador.fit_transform(datos_transformados)

        # Usar el pipeline para hacer la predicción
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
