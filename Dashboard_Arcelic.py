import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos simulados
data = {
    "Fecha": ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06",
              "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12",
              "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
              "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
              "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
              "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
              "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
              "2025-07", "2025-08", "2025-09"],
    "Fallas Ocurridas": [12, 14, 10, 15, 13, 16, 12, 18, 17, 14, 11, 13,
                         9, 10, 8, 12, 11, 14, 9, 13, 10, 8, 6, 9,
                         7, 8, 5, 6, 7, 6, 5, 7, 4, 5, 3, 5,
                         4, 3, 2, 2, 3, 2, 2, 1, 1],
    "Fallas Prevenidas": [2, 3, 1, 4, 3, 5, 3, 6, 7, 5, 4, 5,
                          5, 6, 4, 8, 6, 8, 7, 9, 6, 5, 3, 6,
                          4, 6, 3, 4, 5, 4, 3, 5, 3, 3, 2, 3,
                          3, 2, 1, 1, 2, 1, 1, 1, 0],
    "Consumo Energía (kWh)": [200, 210, 195, 220, 215, 225, 230, 240, 245, 235, 225, 215,
                              190, 195, 185, 205, 200, 210, 195, 205, 195, 185, 180, 190,
                              175, 180, 170, 175, 180, 175, 165, 170, 160, 165, 155, 160,
                              150, 155, 145, 150, 145, 140, 135, 130, 125],
    "Vibraciones (Hz)": [1.5, 1.6, 1.4, 1.7, 1.5, 1.8, 1.6, 1.9, 1.8, 1.7, 1.6, 1.5,
                         1.3, 1.4, 1.2, 1.5, 1.4, 1.6, 1.3, 1.5, 1.3, 1.2, 1.1, 1.3,
                         1.1, 1.2, 1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 0.9, 1.0, 0.8, 0.9,
                         0.8, 0.7, 0.6, 0.6, 0.7, 0.6, 0.5, 0.4, 0.4],
    "Costo Mantenimiento Correctivo ($)": [8000, 8500, 7800, 8700, 8600, 9000, 8800, 9500, 9700, 9200, 8900, 8600,
                                            7500, 7800, 7300, 8100, 7900, 8300, 7700, 8100, 7800, 7400, 7000, 7500,
                                            6800, 7100, 6600, 6800, 6900, 6700, 6200, 6400, 5900, 6100, 5700, 5900,
                                            5500, 5700, 5200, 5300, 5100, 5000, 4900, 4700, 4600]
}

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Configuración de Streamlit
st.title("Dashboard de KPIs Normales de Arçelik")
st.header("📊 Seguimiento de Fallas y Mantenimiento Predictivo")

# Presentación de resultados
st.subheader("1. Objetivo del OKR")
st.write("**Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.**")

st.subheader("2. Key Results")
st.write("- KR1: Implementar modelos predictivos en el 100% de las líneas críticas en un año.")
st.write("- KR2: Lograr una reducción del 70% en fallas no programadas en 24 meses.")
st.write("- KR3: Obtener una reducción del 30% en costos de mantenimiento correctivo.")

st.subheader("3. KPIs y su importancia")
st.write("**Número de Fallas Evitadas vs. Fallas Ocurridas**")
st.write("Este KPI mide la efectividad del modelo de predicción en la reducción de fallas no programadas. Un mayor número de fallas evitadas indica que el sistema de gemelo digital está funcionando correctamente.")
st.write("**Costo de Mantenimiento Correctivo**")
st.write("Este KPI muestra el impacto financiero del mantenimiento predictivo. Su disminución refleja eficiencia en la reducción de fallas inesperadas.")

st.subheader("4. Visualización de Datos")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Fecha"], df["Fallas Ocurridas"], marker="o", label="Fallas Ocurridas", linestyle="-")
ax.plot(df["Fecha"], df["Fallas Prevenidas"], marker="s", label="Fallas Prevenidas", linestyle="dashed")
ax.set_title("¿Cómo impacta el mantenimiento predictivo en la reducción de fallas?")
ax.set_xlabel("Fecha")
ax.set_ylabel("Número de Fallas")
ax.legend()
st.pyplot(fig)

st.subheader("5. Datos y Fuentes")
st.write("**Variable: Fallas Ocurridas**")
st.write("Fuente de datos: Registros de mantenimiento de planta")
st.write("Método de obtención: Datos históricos de fallas no programadas")
st.write("Transformaciones necesarias: Promedio mensual y tendencias")

st.subheader("6. Acciones Necesarias")
st.write("✔️ Ampliar el uso de sensores IoT para monitoreo en tiempo real.")
st.write("✔️ Ajustar modelos predictivos con datos más precisos y entrenamiento continuo.")
st.write("✔️ Integrar dashboards de visualización con alertas automáticas.")

st.success("Dashboard dinámico construido en Streamlit.")
