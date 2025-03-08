import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Arçelik - Dashboard de Predicción de Fallas",
    page_icon="🏭",
    layout="wide"
)

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        # Cargar el archivo CSV desde el mismo directorio
        df = pd.read_csv("KPIs_normales_Arcelik.csv")
        
        # Verificar la columna de fecha
        date_column_candidates = ['fecha', 'date', 'Fecha', 'Date', 'FECHA', 'DATE']
        for col in date_column_candidates:
            if col in df.columns:
                df['fecha'] = pd.to_datetime(df[col])
                break
        else:
            st.warning("No se encontró una columna de fecha. Creando fechas simuladas.")
            df['fecha'] = pd.date_range(start='2022-01-01', periods=len(df), freq='M')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

# Cargar datos
df = load_data()

# Verificar si los datos están cargados
if df.empty:
    st.error("No se pudieron cargar los datos. Verifique el archivo CSV.")
    st.stop()

# Sidebar filtros
st.sidebar.header("Filtros")

# Rango de fechas
min_date = df['fecha'].min()
max_date = df['fecha'].max()
selected_date_range = st.sidebar.date_input("Rango de Fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = df[(df['fecha'] >= pd.Timestamp(start_date)) & (df['fecha'] <= pd.Timestamp(end_date))]
else:
    filtered_df = df

# Filtro de máquinas
selected_machines = st.sidebar.multiselect("Seleccionar Máquinas", options=df['maquina'].unique(), default=df['maquina'].unique())
if selected_machines:
    filtered_df = filtered_df[filtered_df['maquina'].isin(selected_machines)]

# Layout principal
st.title("🏭 Dashboard de KPIs: Predicción de Fallas mediante Gemelos Digitales")

# Sección de OKR
st.subheader("1️⃣ Objetivo del OKR")
st.info("Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.")

# Sección de KRs
st.subheader("2️⃣ Key Results (KR)")
st.markdown("✅ **KR1**: Implementar modelos predictivos de fallas en el 100% de las líneas críticas en un año.")
st.markdown("✅ **KR2**: Lograr una reducción del 70% en fallas no programadas en 24 meses.")
st.markdown("✅ **KR3**: Obtener una reducción del 30% en costos de mantenimiento correctivo.")

# KPI 1: Número de fallas evitadas vs. fallas ocurridas
st.subheader("3️⃣ KPIs y Visualizaciones")
st.markdown("### 📌 KPI 1: Número de fallas evitadas vs. fallas ocurridas")

# Agregación de datos
kpi1_data = filtered_df.groupby('fecha').agg({'fallas_ocurridas': 'sum', 'fallas_evitadas': 'sum'}).reset_index()
kpi1_data['tasa_prevencion'] = (kpi1_data['fallas_evitadas'] / (kpi1_data['fallas_ocurridas'] + kpi1_data['fallas_evitadas']) * 100).round(1)

# Crear gráfico
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=kpi1_data['fecha'], y=kpi1_data['fallas_ocurridas'], mode='lines+markers', name='Fallas Ocurridas', marker=dict(color='red')))
fig1.add_trace(go.Scatter(x=kpi1_data['fecha'], y=kpi1_data['fallas_evitadas'], mode='lines+markers', name='Fallas Evitadas', marker=dict(color='green')))
fig1.update_layout(title='Evolución de Fallas Ocurridas vs. Fallas Evitadas', xaxis_title='Fecha', yaxis_title='Número de Fallas')
st.plotly_chart(fig1, use_container_width=True)

# KPI 2: Consumo energético y vibraciones
st.markdown("### 📌 KPI 2: Consumo energético y vibraciones anómalas")
fig2 = px.scatter(filtered_df, x="vibraciones_anomalas", y="consumo_energia", color="maquina", title="Correlación entre Vibraciones y Consumo Energético")
st.plotly_chart(fig2, use_container_width=True)

# KPI 3: Costo de mantenimiento correctivo
st.markdown("### 📌 KPI 3: Costo de mantenimiento correctivo")
kpi3_data = filtered_df.groupby('fecha').agg({'costo_mantenimiento_correctivo': 'sum'}).reset_index()
fig3 = px.bar(kpi3_data, x='fecha', y='costo_mantenimiento_correctivo', title='Evolución de Costos de Mantenimiento Correctivo')
st.plotly_chart(fig3, use_container_width=True)

# Descarga de datos
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Descargar Datos en CSV", data=csv, file_name="arcelik_datos_gemelo_digital.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("*Dashboard desarrollado para Arçelik - Iniciativa de Gemelos Digitales para Predicción de Fallas*")
