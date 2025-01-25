import os
import streamlit as st
import pandas as pd
import joblib as joblib 
import numpy as np

print(os.getcwd())

modelo_tiempo_entrega_path = os.path.abspath('03_PKL/m_tiempo_pedido_normal.pkl')
modelo_calculo_repartidores_path = os.path.abspath('03_PKL/calculo_repartidores.pkl')

# Cargar modelos con manejo de errores
try:
    if os.path.exists(modelo_tiempo_entrega_path) and os.path.exists(modelo_calculo_repartidores_path):
        modelo_tiempo_entrega = joblib.load(modelo_tiempo_entrega_path)
        modelo_calculo_repartidores = joblib.load(modelo_calculo_repartidores_path)
    else:
        raise FileNotFoundError("No se encontraron los archivos de modelos en las rutas especificadas.")   
except Exception as e:
    st.error(f'Error cargando modelo: {e}')

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

        # Realizar predicción de tiempo de entrega
        prediccion_tiempo = modelo_tiempo_entrega.predict(datos)

        # Realizar predicción de repartidores
        prediccion_repartidores = modelo_calculo_repartidores.predict(datos)

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
        st.dataframe(datos)

        # Gráfico de distribución simulado
        st.subheader('Distribución de Tiempos de Entrega')
        st.bar_chart(pd.DataFrame({
            'Tiempo de Entrega': np.random.normal(prediccion_tiempo[0], 5, 100)
        }))

    except Exception as e:
        st.error(f'Error en la predicción: {e}')

# Información adicional en la barra lateral
st.sidebar.markdown("""
### Información Adicional
- Este es un ejemplo de una aplicación de Streamlit para predecir el tiempo de entrega de un pedido.
- Los modelos utilizados fueron entrenados con datos reales y pueden ser usados para predecir 
el tiempo de entrega de pedidos en tiendas de comida.
""")