import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar modelos
modelo_tiempo_entrega = joblib.load('03_PKL/m_tiempo_pedido_normal.pkl')
modelo_calculo_repartidores = joblib.load('03_PKL/calculo_repartidores.pkl')

# Función para transformar los datos de entrada
def transformar_datos(datos):
    # Codificación de 'store_primary_category' con LabelEncoder
    encoder_category = LabelEncoder()
    datos['store_primary_category_encoded'] = encoder_category.fit_transform(datos['store_primary_category'])
    
    # Codificación de 'order_day' con LabelEncoder
    encoder_day = LabelEncoder()
    datos['order_day_encoded'] = encoder_day.fit_transform(datos['order_day'])

    # Crear 'grouped_category' (agrupamos las categorías de tienda)
    def crear_grouped_category(categoria):
        if categoria in ['Fast Food', 'Mexicana', 'Italiana']:
            return 'Comida rápida'
        elif categoria in ['Saludable', 'Mediterránea', 'Asiática']:
            return 'Comida saludable'
        else:
            return 'Otros'

    datos['grouped_category'] = datos['store_primary_category'].apply(crear_grouped_category)

    # Crear 'is_high_duration' (por ejemplo, si la hora del pedido es mayor a un umbral)
    datos['is_high_duration'] = datos['order_hour'] > 18  # Ejemplo simple

    # Crear 'log_delivery_duration' y 'delivery_duration' (ejemplo ficticio)
    # Supongamos que delivery_duration se calcula de alguna manera, como la duración del pedido
    datos['delivery_duration'] = datos['order_hour'] + np.random.randint(10, 30, size=len(datos))  # Ficticio
    datos['log_delivery_duration'] = np.log(datos['delivery_duration'])  # Logaritmo de la duración

    # Cálculo de partner_density (densidad de repartidores)
    datos['partner_density'] = datos['total_onshift_partners'] / (datos['total_outstanding_orders'] + 1)

    # Cálculo de subtotal (puedes modificar esta fórmula según el valor real de los productos)
    datos['subtotal'] = np.random.uniform(10, 100, size=len(datos))  # Asume un valor aleatorio por ahora

    # Crear otras características faltantes necesarias
    datos['order_period_encoded'] = datos['order_hour'] // 6  # Ficticio, ajusta según tu caso
    datos['max_item_price'] = np.random.uniform(10, 100, size=len(datos))  # Ficticio
    datos['num_distinct_items'] = 3  # Solo para ilustración

    # Asegurarse de que los datos tengan las columnas correctas y en el orden correcto
    expected_columns = [
        'log_delivery_duration', 'is_high_duration', 'total_outstanding_orders',
        'subtotal', 'order_period_encoded', 'num_distinct_items', 'max_item_price', 'total_busy_partners',
        'order_hour', 'partner_density', 'grouped_category'
    ]

    # Asegurarse de que el DataFrame tenga las columnas correctas en el orden adecuado
    datos = datos[expected_columns]

    return datos

# Título de la app
st.title('Predicción de Tiempo de Entrega 🚚')

# Contenedor principal para parámetros de entrada
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        store_primary_category = st.selectbox('Categoría de Tienda', [
            'Italiana', 'Mexicana', 'Fast Food', 'Americana', 'Asiática', 
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
            'store_primary_category': store_primary_category,
            'total_onshift_partners': total_onshift_partners,
            'total_busy_partners': total_busy_partners,
            'total_outstanding_orders': total_outstanding_orders,
            'order_day': order_day,
            'order_hour': order_hour
        }])

        # Transformar los datos de entrada para que coincidan con lo que espera el modelo
        datos_transformados = transformar_datos(datos)

        # Realizar predicción de tiempo de entrega
        prediccion_tiempo = modelo_tiempo_entrega.predict(datos_transformados)

        # Realizar predicción de repartidores
        prediccion_repartidores = modelo_calculo_repartidores.predict(datos_transformados)

        # Mostrar resultados
        st.subheader('Resultados de la Predicción')

        col1, col2 = st.columns(2)

        with col1:
            st.metric('Duración Estimada', f'{prediccion_tiempo[0]:.2f} minutos')
            st.metric('Categoría de Tienda', store_primary_category)
            st.metric('Repartidores Estimados', f'{prediccion_repartidores[0]:.0f}')

        with col2:
            st.metric('Repartidores Disponibles', total_onshift_partners)
            st.metric('Pedidos Pendientes', total_outstanding_orders)

        # Mostrar DataFrame de inputs
        st.subheader('Detalles del Pedido')
        st.dataframe(datos_transformados)

        # Gráfico de distribución simulado
        st.subheader('Distribución de Tiempos de Entrega')
        st.bar_chart(pd.DataFrame({
            'Tiempo de Entrega': np.random.normal(prediccion_tiempo[0], 5, 100)
        }))
        
    except Exception as e:
        st.error(f'Error en la predicción: {e}')
