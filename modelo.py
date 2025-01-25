import joblib
import os 
import streamlit as st

# Obtener la ruta absoluta de los modelos
modelo_tiempo_entrega_path = os.path.abspath('03_PKL/m_tiempo_pedido_normal.pkl')
modelo_calculo_repartidores_path = os.path.abspath('03_PKL/calculo_repartidores.pkl')

# Cargar modelos con manejo de errores
try:
    if os.path.exists(modelo_tiempo_entrega_path) and os.path.exists(modelo_calculo_repartidores_path):
        modelo_tiempo_entrega = joblib.load(modelo_tiempo_entrega_path)
        modelo_calculo_repartidores = joblib.load(modelo_calculo_repartidores_path)
        st.success("Modelos cargados correctamente.")
    else:
        raise FileNotFoundError("No se encontraron los archivos de modelos en las rutas especificadas.")
except Exception as e:
    modelo_tiempo_entrega = None
    modelo_calculo_repartidores = None
    st.error(f'Error cargando modelo: {e}')
    

print(os.path.exists('03_PKL/m_tiempo_pedido_normal.pkl'))
print(os.path.exists('03_PKL/calculo_repartidores.pkl'))