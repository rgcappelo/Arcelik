import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar

# Set page configuration
st.set_page_config(
    page_title="Arçelik - Dashboard de Predicción de Fallas",
    page_icon="🏭",
    layout="wide"
)
# Function to load data
@st.cache_data
def load_data():
    try:
        # Leer el archivo CSV directamente desde el mismo directorio
        df = pd.read_csv("KPIs_normales_Arcelik.csv")
        
        # Mostrar información sobre las columnas disponibles para depuración
        st.sidebar.write("Columnas disponibles en el CSV:", df.columns.tolist())
        
        # Verificar si existe una columna de fecha (pueden tener diferentes nombres)
        date_column_candidates = ['fecha', 'date', 'Fecha', 'Date', 'FECHA', 'DATE']
        date_column = None
        
        for col in date_column_candidates:
            if col in df.columns:
                date_column = col
                break
        
        # Si encontramos una columna de fecha, convertirla a datetime
        if date_column:
            df['fecha'] = pd.to_datetime(df[date_column])
        # Si no hay columna de fecha, crear una fecha ficticia para evitar errores
        else:
            st.warning("No se encontró una columna de fecha en el CSV. Creando fechas simuladas.")
            df['fecha'] = pd.date_range(start='2022-01-01', periods=len(df), freq='M')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        
        # Crear un DataFrame simple para evitar errores
        sample_dates = pd.date_range(start='2022-01-01', periods=36, freq='M')
        sample_df = pd.DataFrame({
            'fecha': sample_dates,
            'mes_anio': [d.strftime('%b %Y') for d in sample_dates],
            'maquina': ['Máquina A'] * 36,
            'temperatura_equipo': [65] * 36,
            'vibraciones_anomalas': [120] * 36,
            'consumo_energia': [450] * 36,
            'fallas_ocurridas': [10] * 36,
            'fallas_evitadas': [5] * 36,
            'costo_mantenimiento_correctivo': [5000] * 36,
            'tiempo_respuesta_alertas': [30] * 36
        })
        return sample_df


    
    df = pd.DataFrame(data)
    return df

# Load data
df = load_data()

# Main dashboard layout
st.title("🏭 Dashboard de KPIs: Predicción de Fallas mediante Gemelos Digitales")

# Sidebar filters
st.sidebar.header("Filtros")

# Date range filter
min_date = df['fecha'].min().to_pydatetime()
max_date = df['fecha'].max().to_pydatetime()
selected_date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = df[(df['fecha'] >= pd.Timestamp(start_date)) & 
                     (df['fecha'] <= pd.Timestamp(end_date))]
else:
    filtered_df = df

# Machine filter
selected_machines = st.sidebar.multiselect(
    "Seleccionar Máquinas",
    options=df['maquina'].unique(),
    default=df['maquina'].unique()
)
if selected_machines:
    filtered_df = filtered_df[filtered_df['maquina'].isin(selected_machines)]

# Metric summary
st.subheader("1️⃣ Objetivo del OKR")
st.info("Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.")

# KR section
st.subheader("2️⃣ Key Results (KR)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("✅ **KR1**: Implementar modelos predictivos de fallas en el 100% de las líneas críticas en un año.")
with col2:
    st.markdown("✅ **KR2**: Lograr una reducción del 70% en fallas no programadas en 24 meses.")
with col3:
    st.markdown("✅ **KR3**: Obtener una reducción del 30% en costos de mantenimiento correctivo.")

st.subheader("3️⃣ KPIs y Visualizaciones")

# KPI 1: Número de fallas evitadas vs. fallas ocurridas
st.markdown("### 📌 KPI 1: Número de fallas evitadas vs. fallas ocurridas")

# Aggregate data by month for KPI 1
kpi1_data = filtered_df.groupby('fecha').agg({
    'fallas_ocurridas': 'sum',
    'fallas_evitadas': 'sum'
}).reset_index()

# Calculate prevention rate
kpi1_data['tasa_prevencion'] = (kpi1_data['fallas_evitadas'] / 
                               (kpi1_data['fallas_ocurridas'] + kpi1_data['fallas_evitadas']) * 100).round(1)

# Create metrics cards
col1, col2, col3 = st.columns(3)
with col1:
    total_failures = kpi1_data['fallas_ocurridas'].sum()
    st.metric(
        label="Total Fallas Ocurridas",
        value=f"{total_failures:,}",
        delta=None
    )
with col2:
    total_prevented = kpi1_data['fallas_evitadas'].sum()
    st.metric(
        label="Total Fallas Evitadas",
        value=f"{total_prevented:,}",
        delta=f"{total_prevented/(total_failures+total_prevented)*100:.1f}%"
    )
with col3:
    latest_prevention_rate = kpi1_data.iloc[-1]['tasa_prevencion']
    first_prevention_rate = kpi1_data.iloc[0]['tasa_prevencion']
    st.metric(
        label="Tasa de Prevención Actual",
        value=f"{latest_prevention_rate:.1f}%",
        delta=f"{latest_prevention_rate - first_prevention_rate:.1f}%"
    )

# Create interactive plot for KPI 1
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=kpi1_data['fecha'],
    y=kpi1_data['fallas_ocurridas'],
    mode='lines+markers',
    name='Fallas Ocurridas',
    marker=dict(color='red')
))
fig1.add_trace(go.Scatter(
    x=kpi1_data['fecha'],
    y=kpi1_data['fallas_evitadas'],
    mode='lines+markers',
    name='Fallas Evitadas',
    marker=dict(color='green')
))
fig1.update_layout(
    title='Evolución Mensual de Fallas Ocurridas vs. Fallas Prevenidas',
    xaxis_title='Fecha',
    yaxis_title='Número de Fallas',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(fig1, use_container_width=True)

# Explanation
with st.expander("📊 Detalles del KPI 1"):
    st.markdown("""
    **Cómo se construyó:** Se obtiene comparando las fallas ocurridas en los últimos 36 meses con la cantidad de fallas que pudieron ser prevenidas mediante modelos predictivos basados en gemelos digitales.
    
    **Importancia:** Permite evaluar la efectividad del modelo predictivo y determinar si la estrategia de mantenimiento preventivo está funcionando correctamente.
    
    **Factores que lo afectan:** Condiciones de operación, vibraciones, temperatura, frecuencia de mantenimiento, y calidad de la calibración de modelos predictivos.
    
    **Pregunta clave:** ¿Cómo ha evolucionado la tasa de fallas y qué impacto ha tenido la predicción en su reducción?
    """)

# KPI 2: Consumo energético y vibraciones anómalas
st.markdown("### 📌 KPI 2: Consumo energético y vibraciones anómalas")

# Create metrics cards for KPI 2
col1, col2 = st.columns(2)
with col1:
    avg_energy = filtered_df['consumo_energia'].mean()
    st.metric(
        label="Consumo Energético Promedio (kWh)",
        value=f"{avg_energy:.2f}"
    )
with col2:
    avg_vibration = filtered_df['vibraciones_anomalas'].mean()
    st.metric(
        label="Vibraciones Anómalas Promedio (Hz)",
        value=f"{avg_vibration:.2f}"
    )

# Create scatter plot for KPI 2
fig2 = px.scatter(
    filtered_df,
    x="vibraciones_anomalas",
    y="consumo_energia",
    color="maquina",
    size="temperatura_equipo",
    hover_name="maquina",
    hover_data=["fecha", "temperatura_equipo", "fallas_ocurridas"],
    title="Correlación entre Vibraciones Anómalas y Consumo Energético",
    labels={
        "vibraciones_anomalas": "Vibraciones Anómalas (Hz)",
        "consumo_energia": "Consumo Energético (kWh)",
        "temperatura_equipo": "Temperatura (°C)"
    }
)
fig2.update_layout(
    xaxis_title="Vibraciones Anómalas (Hz)",
    yaxis_title="Consumo Energético (kWh)"
)
st.plotly_chart(fig2, use_container_width=True)

# Explanation
with st.expander("📊 Detalles del KPI 2"):
    st.markdown("""
    **Cómo se construyó:** Se analizan los datos de sensores IoT en las máquinas críticas, evaluando patrones de consumo de energía en relación con vibraciones mecánicas fuera del rango normal.
    
    **Importancia:** Permite correlacionar problemas operativos con posibles fallas, facilitando la detección temprana antes de que ocurran daños severos.
    
    **Factores que lo afectan:** Edad del equipo, eficiencia del motor, frecuencia de operación y ajustes mecánicos inadecuados.
    
    **Pregunta clave:** ¿Cómo influyen las vibraciones anómalas en el consumo energético y qué relación tienen con la aparición de fallas?
    """)

# KPI 3: Costo de mantenimiento correctivo
st.markdown("### 📌 KPI 3: Costo de mantenimiento correctivo")

# Calculate before/after implementation date (assuming middle point of the date range)
implementation_date = min_date + (max_date - min_date) / 2
before_implementation = filtered_df[filtered_df['fecha'] < pd.Timestamp(implementation_date)]
after_implementation = filtered_df[filtered_df['fecha'] >= pd.Timestamp(implementation_date)]

# Calculate metrics for KPI 3
before_avg_cost = before_implementation['costo_mantenimiento_correctivo'].mean()
after_avg_cost = after_implementation['costo_mantenimiento_correctivo'].mean()
cost_reduction_pct = ((before_avg_cost - after_avg_cost) / before_avg_cost * 100) if before_avg_cost > 0 else 0

# Create metrics cards for KPI 3
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Costo Promedio Antes ($)",
        value=f"{before_avg_cost:,.2f}"
    )
with col2:
    st.metric(
        label="Costo Promedio Después ($)",
        value=f"{after_avg_cost:,.2f}",
        delta=f"-{(before_avg_cost - after_avg_cost):,.2f}",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Reducción de Costos",
        value=f"{cost_reduction_pct:.1f}%"
    )

# Aggregate maintenance costs by month and machine
kpi3_data = filtered_df.groupby(['fecha', 'maquina']).agg({
    'costo_mantenimiento_correctivo': 'sum'
}).reset_index()

# Create bar chart for KPI 3
fig3 = px.bar(
    kpi3_data,
    x='fecha',
    y='costo_mantenimiento_correctivo',
    color='maquina',
    title='Evolución de Costos de Mantenimiento Correctivo por Máquina',
    labels={
        'fecha': 'Fecha',
        'costo_mantenimiento_correctivo': 'Costo de Mantenimiento Correctivo ($)',
        'maquina': 'Máquina'
    }
)
fig3.add_vline(
    x=implementation_date, 
    line_dash="dash", 
    line_color="red",
    annotation_text="Implementación del Modelo Predictivo",
    annotation_position="top right"
)
st.plotly_chart(fig3, use_container_width=True)

# Explanation
with st.expander("📊 Detalles del KPI 3"):
    st.markdown("""
    **Cómo se construyó:** Se comparan los costos de mantenimiento correctivo antes y después de la implementación del modelo de predicción de fallas.
    
    **Importancia:** Permite visualizar el ahorro generado por mantenimiento preventivo en comparación con el correctivo.
    
    **Factores que lo afectan:** Tiempo de respuesta ante fallas, tipo de avería, disponibilidad de repuestos y eficiencia del mantenimiento programado.
    
    **Pregunta clave:** ¿Cuánto ha disminuido el costo de mantenimiento correctivo gracias a la implementación del gemelo digital?
    """)

# Data sources and description
st.subheader("4️⃣ Descripción de los Datos")
with st.expander("Ver Descripción de Datos"):
    data_desc = pd.DataFrame({
        'Variable': ['Fecha de la medición', 'Máquina monitoreada', 'Temperatura del equipo (°C)', 'Vibraciones anómalas (Hz)', 
                     'Consumo de energía (kWh)', 'Número de fallas detectadas', 'Número de fallas evitadas', 
                     'Costo de mantenimiento correctivo ($)', 'Tiempo medio de respuesta ante alertas (min)'],
        'Fuente de Datos': ['Sensores IoT y registros de mantenimiento', 'Inventario de equipos en la planta', 'Sensores IoT', 
                            'Sensores IoT (acelerómetros)', 'Sensores IoT', 'Reportes de mantenimiento', 'Modelos de predicción', 
                            'ERP y Finanzas', 'Sistema de monitoreo'],
        'Método de Obtención': ['Registro mensual en base de datos', 'Relación con datos IoT', 'Captura cada minuto, agregado mensual', 
                               'Captura en tiempo real, agregado diario', 'Registros de uso por máquina', 'Registro manual', 
                               'Simulación con gemelo digital', 'Reportes contables', 'Captura automática'],
        'Transformaciones Necesarias': ['Ninguna, solo formato de tiempo', 'Asignación a cada serie de datos', 'Media mensual', 
                                       'Desviaciones estándar y medias', 'Media mensual', 'Frecuencia de ocurrencia', 
                                       'Comparación con histórico', 'Conversión a USD mensual', 'Cálculo de promedios']
    })
    st.table(data_desc)

# Actions 
st.subheader("5️⃣ Acciones Necesarias")
st.markdown("""
🔹 **Optimizar el monitoreo en tiempo real** de fallas con alertas automáticas basadas en umbrales de vibraciones y temperatura.

🔹 **Actualizar el modelo de predicción** para reducir el margen de error en la detección temprana de anomalías.

🔹 **Integrar dashboards en toda la organización**, permitiendo a los operadores visualizar tendencias en tiempo real y actuar antes de que ocurran fallas.
""")

# View and download data
st.subheader("Datos Utilizados")
with st.expander("Ver Datos"):
    st.dataframe(filtered_df)

# Download button for reports
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar Datos en CSV",
    data=csv,
    file_name="arcelik_datos_gemelo_digital.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown("*Dashboard desarrollado para Arçelik - Iniciativa de Gemelos Digitales para Predicción de Fallas*")
