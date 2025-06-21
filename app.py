import streamlit as st
import numpy as np
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('modelo_dashboard.pkl')

# Título del panel
st.set_page_config(page_title="Predicción de Lluvia", layout="centered")
st.title("🌦️ Predicción de Lluvia para Mañana")
st.markdown("Este panel utiliza condiciones meteorológicas actuales para predecir si lloverá al día siguiente.")

# Entradas del usuario
st.sidebar.header("🛠️ Ajusta las condiciones del clima")
humedad = st.sidebar.slider("Humedad a las 3PM (%)", 0, 100, 65)
presion = st.sidebar.slider("Presión a las 3PM (hPa)", 980, 1040, 1015)
nubes = st.sidebar.slider("Nubosidad a las 3PM (octavos del cielo)", 0, 9, 4)
sol = st.sidebar.slider("Horas de sol", 0.0, 14.0, 7.0, step=0.1)
viento = st.sidebar.slider("Velocidad de ráfaga de viento (km/h)", 10, 100, 35)
lluvio_hoy = st.sidebar.radio("¿Llovió hoy?", ["No", "Sí"])

# Codificar la respuesta de lluvia
lluvio_hoy_bin = 1 if lluvio_hoy == "Sí" else 0

# Preparar los datos para predicción
input_usuario = np.array([[humedad, presion, nubes, sol, viento, lluvio_hoy_bin]])

# Realizar la predicción
prediccion = modelo.predict(input_usuario)[0]
probabilidad = modelo.predict_proba(input_usuario)[0][1]

# Mostrar el resultado
st.subheader("🌤️ Resultado de la predicción:")
if prediccion == 1:
    st.success("🌧️ *Sí lloverá mañana*")
else:
    st.info("🌤️ *No lloverá mañana*")

# Mostrar probabilidad si se desea
st.write(f"📊 Probabilidad estimada de lluvia: *{probabilidad:.2%}*")