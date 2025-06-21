import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('modelo_dashboard.pkl')

# Configurar página con ancho expandido
st.set_page_config(page_title="Predicción de Lluvia", layout="wide")

# Crear dos columnas: izquierda (formulario) y derecha (visualizaciones)
col1, col2 = st.columns([1.2, 1])

# ----- COLUMNA 1: Formulario e inferencia -----
with col1:
    st.title("🌦️ Predicción de Lluvia para Mañana")
    st.markdown("Este panel utiliza condiciones climáticas actuales para predecir si lloverá al día siguiente.")

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
        st.success("🌧️ *Sí lloverá mañana*")
    else:
        st.info("🌤️ *No lloverá mañana*")

    st.write(f"📊 Probabilidad estimada de lluvia: *{probabilidad:.2%}*")

# ----- COLUMNA 2: Visualización basada en inputs -----
with col2:
    st.markdown("### 📈 Visualización de condiciones ingresadas")

    # Gráfico de barras horizontal con valores ingresados
    df_condiciones = pd.DataFrame({
        'Variable': ['Humedad', 'Presión', 'Nubosidad', 'Horas de Sol', 'Viento'],
        'Valor': [humedad, presion, nubes, sol, viento]
    })

    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.barh(df_condiciones['Variable'], df_condiciones['Valor'], color='#4C72B0')
    ax1.set_xlabel('Valor')
    ax1.set_title('📊 Condiciones meteorológicas actuales')
    st.pyplot(fig1)

    # Gráfico de línea con tendencia hipotética (probabilidad semanal)
    st.markdown("### 📉 Comparación con tendencia simulada")
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Hoy']
    prob_tendencia = [0.3, 0.4, 0.25, 0.55, 0.6, 0.35, probabilidad]

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(dias, prob_tendencia, marker='o', color='#55A868')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probabilidad de lluvia')
    ax2.set_title('📈 Tendencia semanal simulada')
    st.pyplot(fig2)
