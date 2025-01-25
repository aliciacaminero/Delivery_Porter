import os
import streamlit as st
import pandas as pd
import joblib as joblib 
import numpy as np

print(os.getcwd())

# Cargar modelos con manejo de errores
try:
    modelo_tiempo_entrega = os.path.abspath('/03_PKL/m_tiempo_pedido_normal.pkl')
    modelo_tiempo_entrega = os.path.abspath('/03_PKL/calculo_repartidores.pkl')
    
except Exception as e:
    st.error(f'Error cargando modelo: {e}')

# Se inicia el título inicial de la app
st.title('Predicción de Tiempo de Entrega🚚')

# Sidebar para inputs
st.sidebar.header('Parámetros de Entrada')

# Función para generar inputs
def get_inputs():
    inputs = {}
    
    # Categoría de tienda
    inputs['store_primary_category'] = st.sidebar.selectbox('Categoría de Tienda', [
        'Italiana', 'Mexicana', 'Fast Food', 'Americana', 'Asiatica', 
        'Mediterranea', 'India', 'Europea', 'Saludable', 'Bebidas',
        'Otros', 'Postres' 
    ])
    
    # Partners
    inputs['total_onshift_partners'] = st.sidebar.number_input('Total Repartidores Disponibles', min_value=1, max_value=50, value=10)
    inputs['total_busy_partners'] = st.sidebar.number_input('Total Repartidores Asignados', min_value=0, max_value=50, value=5)
    inputs['total_outstanding_orders'] = st.sidebar.number_input('Pedidos Pendientes', min_value=0, max_value=100, value=20)
    
    # Día y hora
    inputs['order_day'] = st.sidebar.selectbox('Día del Pedido', [
        'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'
    ])
    inputs['order_hour'] = st.sidebar.slider('Hora del Pedido', min_value=0, max_value=23, value=12)

    

    
    return pd.DataFrame([inputs])

# Botón de predicción
if st.sidebar.button('Predecir Duración de Entrega del Pedido'):
    try:
        # Obtener inputs
        datos = get_inputs()
        
        # Realizar predicción de tiempo de entrega
        prediccion_tiempo = modelo_tiempo_entrega.predict(datos)
        
        # Columnas para mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Duración Estimada', f'{prediccion_tiempo[0]:.2f} minutos')
            st.metric('Categoría de Tienda', datos['store_primary_category'][0])
        
        with col2:
            st.metric('Total de Artículos', datos['total_items'][0])
            st.metric('Subtotal', f'${datos["subtotal"][0]:.2f}')
        
        # Mostrar DataFrame de inputs
        st.subheader('Detalles del Pedido')
        st.dataframe(datos)
        
        # Gráfico de distribución simulado
        st.subheader('Distribución de Tiempos de Entrega')
        st.bar_chart(pd.DataFrame({
            'Tiempo de Entrega': np.random.normal(prediccion_tiempo[0], 5, 100)
        }))
    
    except Exception as e:
        st.error(f'Error en la predicción: {e}')

# Información adicional
st.sidebar.markdown("""
### Información Adicional
- Este es un ejemplo de una aplicación de Streamlit para predecir el tiempo de entrega de un pedido.
- Los modelos utilizados fueron entrenados con datos reales y pueden ser usados para predecir el tiempo de entrega de pedidos en tiendas de comida.
""")