import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn
import pickle

# Configuración de la página
st.set_page_config(page_title="Predicción de Tiempo de Entrega", page_icon="🛵", layout="centered")

# URL del archivo del modelo de tiempo de entrega
url_modelo_tiempo_entrega = 'http://s68-77.furanet.com/ironhack/m_tiempo_pedido_normal.pkl'

# Intentar cargar el modelo con manejo de errores específico
try:
    response_tiempo_entrega = requests.get(url_modelo_tiempo_entrega)
    if response_tiempo_entrega.status_code == 200:
        try:
            # Intentar cargar el modelo directamente como un array numpy
            modelo_datos = BytesIO(response_tiempo_entrega.content)
            mejor_modelo = np.load(modelo_datos, allow_pickle=True)
            st.success('Modelo de tiempo de entrega cargado correctamente.')
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

def preparar_caracteristicas(datos):
    """
    Prepara las características para la predicción.
    """
    # Convertir categorías de español a inglés
    datos['grouped_category'] = datos['grouped_category'].map(reverse_category_map)
    datos['order_day'] = datos['order_day'].map(reverse_day_map)
    
    # Crear el codificador one-hot para las variables categóricas
    cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Separar características categóricas y numéricas
    cat_features = ['grouped_category', 'order_day']
    num_features = ['order_hour', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']
    
    # Codificar variables categóricas
    cat_encoded = cat_encoder.fit_transform(datos[cat_features])
    
    # Obtener nombres de características codificadas
    cat_feature_names = []
    for i, feature in enumerate(cat_features):
        feature_names = cat_encoder.get_feature_names_out([feature])
        cat_feature_names.extend(feature_names)
    
    # Crear DataFrame con características codificadas
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_feature_names)
    
    # Combinar con características numéricas
    num_df = datos[num_features]
    
    # Concatenar todas las características
    final_features = pd.concat([cat_encoded_df, num_df], axis=1)
    
    return final_features.values

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

        # Preparar características
        X = preparar_caracteristicas(datos_entrada)
        
        # Realizar predicción usando el array numpy directamente
        prediccion_tiempo = mejor_modelo.dot(X.flatten())
        
        # Mostrar resultados
        st.subheader('Resultados de la Predicción')
        
        # Convertir el tiempo a minutos y redondear al minuto más cercano
        minutos = max(1, round(prediccion_tiempo / 60))
        st.metric('Tiempo Estimado de Entrega', f'{minutos} minutos')
            
    except Exception as e:
        st.error(f'Error al procesar los datos: {str(e)}')
        st.write('Detalles del error:')
        st.write(e)