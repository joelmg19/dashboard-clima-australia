import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Configurar la página
st.set_page_config(page_title="Predicción de Lluvia", layout="wide")

# Cargar el modelo entrenado
modelo = joblib.load('modelo_dashboard.pkl')

# Crear columnas (principal y visualización)
col1, col2 = st.columns([2, 1])

# Panel de entrada y predicción
with col1:
    st.title("🌦️ Predicción de Lluvia para Mañana")
    st.markdown("Este panel utiliza condiciones meteorológicas actuales para predecir si lloverá al día siguiente.")

    st.header("🛠️ Ajusta las condiciones del clima")
    humedad = st.slider("Humedad a las 3PM (%)", 0, 100, 65)
    presion = st.slider("Presión a las 3PM (hPa)", 980, 1040, 1015)
    nubes = st.slider("Nubosidad a las 3PM (octavos del cielo)", 0, 9, 4)
    sol = st.slider("Horas de sol", 0.0, 14.0, 7.0, step=0.1)
    viento = st.slider("Velocidad de ráfaga de viento (km/h)", 10, 100, 35)
    lluvio_hoy = st.radio("¿Llovió hoy?", ["No", "Sí"])

    lluvio_hoy_bin = 1 if lluvio_hoy == "Sí" else 0
    input_usuario = np.array([[humedad, presion, nubes, sol, viento, lluvio_hoy_bin]])

    prediccion = modelo.predict(input_usuario)[0]
    probabilidad = modelo.predict_proba(input_usuario)[0][1]

    st.subheader("🌤️ Resultado de la predicción:")
    if prediccion == 1:
        st.success("🌧️ **Sí lloverá mañana**")
    else:
        st.info("🌤️ **No lloverá mañana**")

    st.write(f"📊 Probabilidad estimada de lluvia: **{probabilidad:.2%}**")

# Visualizaciones inspiradas en tu presentación
with col2:
    st.markdown("### 📈 Visualizaciones del clima")

    # Gráfico de torta: distribución de predicciones históricas simuladas
    st.markdown("**Distribución de días con y sin lluvia**")
    labels = ['Sin lluvia', 'Con lluvia']
    values = [70, 30]
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Gráfico de barras: comparación de humedad por cluster
    st.markdown("**Comparación de humedad por tipo de día**")
    fig2, ax2 = plt.subplots()
    tipos = ['Soleado', 'Nublado', 'Lluvioso']
    humedades = [45, 65, 85]
    ax2.bar(tipos, humedades, color=['#FEE08B', '#91BFDB', '#4575B4'])
    ax2.set_ylabel('% Humedad')
    st.pyplot(fig2)

    # Gráfico de líneas: probabilidad de lluvia en la semana (simulado)
    st.markdown("**Tendencia semanal simulada**")
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    probs = [0.2, 0.4, 0.65, 0.85, 0.6, 0.35, 0.15]
    fig3, ax3 = plt.subplots()
    ax3.plot(dias, probs, marker='o', color='green')
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Prob. de lluvia')
    st.pyplot(fig3)
