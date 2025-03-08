import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Detección de Fallas - Arçelik",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #3498db;
    }
    .target-text {
        color: #7f8c8d;
        font-size: 14px;
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Datos proporcionados (convertidos a un formato más completo)
# Generar datos mensuales desde enero 2022 hasta septiembre 2025 (45 meses)
fechas = []
for year in range(2022, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 9:  # Solo hasta septiembre 2025
            break
        fechas.append(f"{year}-{month:02d}")

# Datos simulados de fallas (los datos tienen una reducción gradual y proyección a futuro)
np.random.seed(42)  # Para reproducibilidad
fallas_ocurridas = [15, 17, 16, 14, 15, 18, 20, 17, 15, 14, 12, 13,  # 2022
                    14, 12, 13, 11, 12, 10, 9, 11, 10, 9, 8, 7,      # 2023
                    9, 8, 7, 8, 6, 7, 6, 5, 6, 5, 6, 4,              # 2024
                    5, 6, 4, 3, 4, 3]                                # 2025 (hasta septiembre)
                    
fallas_prevenidas = [5, 6, 7, 8, 7, 9, 10, 11, 12, 13, 14, 15,      # 2022
                     14, 15, 16, 15, 14, 16, 17, 16, 18, 19, 18, 20, # 2023
                     18, 19, 21, 20, 22, 21, 23, 24, 22, 23, 24, 25, # 2024
                     23, 24, 25, 24, 25, 26]                         # 2025 (hasta septiembre)

# Datos de máquinas - 5 máquinas diferentes con sus propios patrones de fallas
maquinas = ["Línea A", "Línea B", "Línea C", "Línea D", "Línea E"]

# Distribuir las fallas entre las máquinas
data = []
for i, fecha in enumerate(fechas):
    total_ocurridas = fallas_ocurridas[i]
    total_prevenidas = fallas_prevenidas[i]
    
    # Distribuir fallas entre máquinas (sumando hasta el total)
    dist_ocurridas = np.random.multinomial(total_ocurridas, [0.3, 0.25, 0.2, 0.15, 0.1])
    dist_prevenidas = np.random.multinomial(total_prevenidas, [0.2, 0.25, 0.15, 0.3, 0.1])
    
    for j, maquina in enumerate(maquinas):
        data.append({
            "Fecha": fecha,
            "Máquina": maquina,
            "Fallas Ocurridas": dist_ocurridas[j],
            "Fallas Prevenidas": dist_prevenidas[j]
        })

# Crear DataFrame
df = pd.DataFrame(data)
df['Fecha_dt'] = pd.to_datetime(df['Fecha'])
df['Año'] = df['Fecha_dt'].dt.year
df['Mes'] = df['Fecha_dt'].dt.month
df['Mes_Nombre'] = df['Fecha_dt'].dt.month_name()

# Calcular métricas adicionales
df['Fallas Totales'] = df['Fallas Ocurridas'] + df['Fallas Prevenidas']
df['Tasa de Prevención'] = np.round((df['Fallas Prevenidas'] / df['Fallas Totales']) * 100, 1)

# Agrupar por fecha para los totales
df_totales = df.groupby('Fecha').agg({
    'Fallas Ocurridas': 'sum',
    'Fallas Prevenidas': 'sum',
    'Fallas Totales': 'sum'
}).reset_index()

df_totales['Tasa de Prevención'] = np.round((df_totales['Fallas Prevenidas'] / df_totales['Fallas Totales']) * 100, 1)
df_totales['Fecha_dt'] = pd.to_datetime(df_totales['Fecha'])
df_totales['Mes'] = df_totales['Fecha_dt'].dt.month
df_totales['Año'] = df_totales['Fecha_dt'].dt.year
df_totales['Proyección'] = df_totales['Fecha_dt'] >= '2024-10-01'

# Sidebar para filtros
st.sidebar.image("https://via.placeholder.com/150x80?text=Arçelik", width=150)
st.sidebar.markdown("### Filtros del Dashboard")

# Filtro de fechas
min_date = df['Fecha_dt'].min()
max_date = df['Fecha_dt'].max()

fecha_inicio = st.sidebar.date_input("Fecha Inicio", 
                                    min_value=min_date,
                                    max_value=max_date,
                                    value=min_date)
fecha_fin = st.sidebar.date_input("Fecha Fin", 
                                 min_value=min_date,
                                 max_value=max_date,
                                 value=max_date)

# Convertir a datetime para filtrar
fecha_inicio = pd.to_datetime(fecha_inicio)
fecha_fin = pd.to_datetime(fecha_fin)

# Filtro de máquinas
maquinas_seleccionadas = st.sidebar.multiselect("Seleccionar Máquinas", 
                                              options=maquinas,
                                              default=maquinas)

# Aplicar filtros a los datos
filtro_fecha = (df['Fecha_dt'] >= fecha_inicio) & (df['Fecha_dt'] <= fecha_fin)
df_filtrado = df[filtro_fecha & df['Máquina'].isin(maquinas_seleccionadas)]

filtro_fecha_totales = (df_totales['Fecha_dt'] >= fecha_inicio) & (df_totales['Fecha_dt'] <= fecha_fin)
df_totales_filtrado = df_totales[filtro_fecha_totales]

# Marcar datos de proyección
inicio_proyeccion = pd.to_datetime('2024-10-01')
df_filtrado['Es Proyección'] = df_filtrado['Fecha_dt'] >= inicio_proyeccion
df_totales_filtrado['Es Proyección'] = df_totales_filtrado['Fecha_dt'] >= inicio_proyeccion

# Cabecera del Dashboard
st.markdown('<div class="main-header">Dashboard de Detección de Fallas en Maquinaria - Arçelik</div>', unsafe_allow_html=True)

# Información del OKR
with st.expander("📌 Objetivo del OKR y Key Results", expanded=False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Objetivo Principal")
        st.markdown("**Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.**")
    
    with col2:
        st.markdown("### Key Results (KR)")
        st.markdown("""
        - **KR1:** Implementar modelos predictivos de fallas en el **100% de las líneas críticas** en un año.
        - **KR2:** Lograr una reducción del **70% en fallas no programadas** en 24 meses.
        - **KR3:** Obtener una reducción del **30% en costos de mantenimiento correctivo**.
        """)

# Resumen de métricas
st.markdown('<div class="sub-header">Resumen de Métricas</div>', unsafe_allow_html=True)

# Calcular métricas de resumen
total_fallas_ocurridas = df_filtrado['Fallas Ocurridas'].sum()
total_fallas_prevenidas = df_filtrado['Fallas Prevenidas'].sum()
total_fallas = total_fallas_ocurridas + total_fallas_prevenidas
tasa_prevencion = np.round((total_fallas_prevenidas / total_fallas) * 100, 1) if total_fallas > 0 else 0

# Calcular tendencias (comparación con periodo anterior de igual duración)
dias_periodo = (fecha_fin - fecha_inicio).days
fecha_periodo_anterior_fin = fecha_inicio - pd.Timedelta(days=1)
fecha_periodo_anterior_inicio = fecha_periodo_anterior_fin - pd.Timedelta(days=dias_periodo)

# Filtrar para el periodo anterior
filtro_fecha_anterior = (df['Fecha_dt'] >= fecha_periodo_anterior_inicio) & (df['Fecha_dt'] <= fecha_periodo_anterior_fin)
filtro_maquinas = df['Máquina'].isin(maquinas_seleccionadas)
df_periodo_anterior = df[filtro_fecha_anterior & filtro_maquinas]

# Calcular métricas del periodo anterior
total_fallas_ocurridas_anterior = df_periodo_anterior['Fallas Ocurridas'].sum()
total_fallas_prevenidas_anterior = df_periodo_anterior['Fallas Prevenidas'].sum()

# Calcular variaciones porcentuales
var_fallas_ocurridas = np.round(((total_fallas_ocurridas - total_fallas_ocurridas_anterior) / total_fallas_ocurridas_anterior) * 100, 1) if total_fallas_ocurridas_anterior > 0 else 100
var_fallas_prevenidas = np.round(((total_fallas_prevenidas - total_fallas_prevenidas_anterior) / total_fallas_prevenidas_anterior) * 100, 1) if total_fallas_prevenidas_anterior > 0 else 100

# Mostrar métricas en tarjetas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">Fallas Ocurridas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="text-align: center;">{total_fallas_ocurridas}</div>', unsafe_allow_html=True)
    if var_fallas_ocurridas < 0:
        st.markdown(f'<div style="text-align: center; color: green;">▼ {abs(var_fallas_ocurridas)}% vs período anterior</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: center; color: red;">▲ {var_fallas_ocurridas}% vs período anterior</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">Fallas Prevenidas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="text-align: center;">{total_fallas_prevenidas}</div>', unsafe_allow_html=True)
    if var_fallas_prevenidas > 0:
        st.markdown(f'<div style="text-align: center; color: green;">▲ {var_fallas_prevenidas}% vs período anterior</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: center; color: red;">▼ {abs(var_fallas_prevenidas)}% vs período anterior</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">Total Fallas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="text-align: center;">{total_fallas}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;" class="target-text">Meta: Reducción del 60%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">Tasa de Prevención</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="text-align: center;">{tasa_prevencion}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;" class="target-text">Proporción de fallas prevenidas</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Gráficos
st.markdown('<div class="sub-header">Evolución de Fallas Detectadas y Prevenidas</div>', unsafe_allow_html=True)

# Crear gráfico de evolución temporal con Plotly
fig = go.Figure()

# Añadir línea de fallas ocurridas
fig.add_trace(go.Scatter(
    x=df_totales_filtrado['Fecha_dt'],
    y=df_totales_filtrado['Fallas Ocurridas'],
    mode='lines+markers',
    name='Fallas Ocurridas',
    line=dict(color='#e74c3c', width=2),
    marker=dict(size=8)
))

# Añadir línea de fallas prevenidas
fig.add_trace(go.Scatter(
    x=df_totales_filtrado['Fecha_dt'],
    y=df_totales_filtrado['Fallas Prevenidas'],
    mode='lines+markers',
    name='Fallas Prevenidas',
    line=dict(color='#2ecc71', width=2, dash='dash'),
    marker=dict(size=8)
))

# Añadir línea vertical para proyección
inicio_proyeccion_str = inicio_proyeccion.strftime('%Y-%m-%d')
if (df_totales_filtrado['Fecha_dt'] >= inicio_proyeccion).any():
    fig.add_vline(x=inicio_proyeccion, line_width=2, line_dash="dot", line_color="grey")
    fig.add_annotation(x=inicio_proyeccion, y=1.05, yref="paper",
                      text="Inicio de Proyección", showarrow=True,
                      arrowhead=1, arrowcolor="grey")

# Personalizar el gráfico
fig.update_layout(
    title='Evolución de Fallas a lo Largo del Tiempo',
    xaxis_title='Fecha',
    yaxis_title='Número de Fallas',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=500,
    margin=dict(l=20, r=20, t=70, b=20),
)

# Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)

# Añadir información sobre la proyección
if (df_filtrado['Es Proyección'] == True).any():
    st.markdown('<div class="highlight">📊 Los datos a partir de octubre 2024 son proyecciones basadas en tendencias y modelos predictivos.</div>', unsafe_allow_html=True)

# Mostrar análisis por máquina
st.markdown('<div class="sub-header">Análisis por Máquina</div>', unsafe_allow_html=True)

# Agrupar por máquina para el análisis
df_por_maquina = df_filtrado.groupby('Máquina').agg({
    'Fallas Ocurridas': 'sum',
    'Fallas Prevenidas': 'sum'
}).reset_index()

df_por_maquina['Total Fallas'] = df_por_maquina['Fallas Ocurridas'] + df_por_maquina['Fallas Prevenidas']
df_por_maquina['Tasa de Prevención (%)'] = np.round((df_por_maquina['Fallas Prevenidas'] / df_por_maquina['Total Fallas']) * 100, 1)

# Gráfico de barras comparativas
col1, col2 = st.columns([2, 1])

with col1:
    # Crear gráfico de barras agrupadas
    fig_barras = go.Figure()
    
    fig_barras.add_trace(go.Bar(
        x=df_por_maquina['Máquina'],
        y=df_por_maquina['Fallas Ocurridas'],
        name='Fallas Ocurridas',
        marker_color='#e74c3c'
    ))
    
    fig_barras.add_trace(go.Bar(
        x=df_por_maquina['Máquina'],
        y=df_por_maquina['Fallas Prevenidas'],
        name='Fallas Prevenidas',
        marker_color='#2ecc71'
    ))
    
    fig_barras.update_layout(
        title='Comparativa de Fallas por Máquina',
        xaxis_title='Máquina',
        yaxis_title='Número de Fallas',
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    
    st.plotly_chart(fig_barras, use_container_width=True)

with col2:
    # Tabla de métricas por máquina
    st.markdown("### Métricas por Máquina")
    st.dataframe(df_por_maquina, hide_index=True)

# Análisis tendencial de la tasa de prevención
st.markdown('<div class="sub-header">Tendencia de la Tasa de Prevención</div>', unsafe_allow_html=True)

# Preparar datos para el gráfico de tasa de prevención
df_tendencia = df_totales_filtrado.copy()
df_tendencia['Periodo'] = df_tendencia['Fecha_dt'].dt.strftime('%Y-%m')

# Calcular media móvil de 3 meses
df_tendencia = df_tendencia.sort_values('Fecha_dt')
df_tendencia['Tasa Media Móvil 3M'] = df_tendencia['Tasa de Prevención'].rolling(window=3, min_periods=1).mean()

# Gráfico de línea con área para tasa de prevención
fig_tasa = go.Figure()

# Añadir área de tasa de prevención
fig_tasa.add_trace(go.Scatter(
    x=df_tendencia['Fecha_dt'],
    y=df_tendencia['Tasa de Prevención'],
    mode='lines',
    name='Tasa de Prevención (%)',
    line=dict(color='#3498db', width=2),
    fill='tozeroy',
    fillcolor='rgba(52, 152, 219, 0.2)'
))

# Añadir línea de media móvil
fig_tasa.add_trace(go.Scatter(
    x=df_tendencia['Fecha_dt'],
    y=df_tendencia['Tasa Media Móvil 3M'],
    mode='lines',
    name='Media Móvil 3 Meses',
    line=dict(color='#e67e22', width=2, dash='dot')
))

# Añadir línea vertical para proyección
if (df_tendencia['Es Proyección'] == True).any():
    fig_tasa.add_vline(x=inicio_proyeccion, line_width=2, line_dash="dot", line_color="grey")

# Personalizar el gráfico
fig_tasa.update_layout(
    title='Tendencia de la Tasa de Prevención de Fallas',
    xaxis_title='Fecha',
    yaxis_title='Tasa de Prevención (%)',
    yaxis=dict(range=[0, 100]),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=400,
    margin=dict(l=20, r=20, t=70, b=20),
)

# Mostrar el gráfico
st.plotly_chart(fig_tasa, use_container_width=True)

# Tabla de datos mensuales para el período seleccionado
with st.expander("Ver datos mensuales detallados", expanded=False):
    # Agrupar por año y mes
    df_mensual = df_filtrado.groupby(['Año', 'Mes', 'Mes_Nombre']).agg({
        'Fallas Ocurridas': 'sum',
        'Fallas Prevenidas': 'sum'
    }).reset_index()
    
    df_mensual['Total Fallas'] = df_mensual['Fallas Ocurridas'] + df_mensual['Fallas Prevenidas']
    df_mensual['Tasa de Prevención (%)'] = np.round((df_mensual['Fallas Prevenidas'] / df_mensual['Total Fallas']) * 100, 1)
    
    # Ordenar por año y mes
    df_mensual = df_mensual.sort_values(['Año', 'Mes'])
    
    # Crear columna para mostrar período
    df_mensual['Período'] = df_mensual['Mes_Nombre'] + ' ' + df_mensual['Año'].astype(str)
    
    # Seleccionar y reordenar columnas para mostrar
    df_mensual_mostrar = df_mensual[['Período', 'Fallas Ocurridas', 'Fallas Prevenidas', 
                                     'Total Fallas', 'Tasa de Prevención (%)']]
    
    st.dataframe(df_mensual_mostrar, hide_index=True)

# Análisis de Metas y Progreso
st.markdown('<div class="sub-header">Progreso Hacia los Key Results</div>', unsafe_allow_html=True)

# Cálculo simplificado del progreso hacia las metas
# KR1: Implementar modelos predictivos en el 100% de líneas críticas
kr1_objetivo = 100  # 100% de las líneas críticas
kr1_progreso = len(maquinas_seleccionadas) / len(maquinas) * 100

# KR2: Reducción del 70% en fallas no programadas
kr2_objetivo = 70  # 70% de reducción
# Calcular el % de reducción en fallas ocurridas (primeros 12 meses vs últimos 12 meses disponibles)
primeros_12m = df[(df['Fecha_dt'] >= pd.to_datetime('2022-01-01')) & (df['Fecha_dt'] <= pd.to_datetime('2022-12-31'))]
ultimos_12m = df[(df['Fecha_dt'] >= pd.to_datetime('2024-01-01')) & (df['Fecha_dt'] <= pd.to_datetime('2024-12-31'))]

fallas_primer_año = primeros_12m['Fallas Ocurridas'].sum()
fallas_ultimo_año = ultimos_12m['Fallas Ocurridas'].sum()

kr2_progreso = ((fallas_primer_año - fallas_ultimo_año) / fallas_primer_año) * 100 if fallas_primer_año > 0 else 0

# KR3: Reducción del 30% en costos de mantenimiento correctivo
kr3_objetivo = 30  # 30% de reducción
# Simulamos que el costo de mantenimiento correctivo está directamente relacionado con las fallas ocurridas
kr3_progreso = kr2_progreso  # Simplificación: misma reducción en % que las fallas

# Mostrar métricas en gráficos de progreso
col1, col2, col3 = st.columns(3)

with col1:
    fig_kr1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=kr1_progreso,
        title={'text': "KR1: Implementación en Líneas Críticas"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "rgba(52, 152, 219, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    fig_kr1.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_kr1, use_container_width=True)
    st.markdown(f"**Meta:** 100% de líneas críticas con modelos predictivos")

with col2:
    fig_kr2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=kr2_progreso,
        title={'text': "KR2: Reducción de Fallas No Programadas"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "rgba(46, 204, 113, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig_kr2.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_kr2, use_container_width=True)
    st.markdown(f"**Meta:** 70% de reducción en 24 meses")

with col3:
    fig_kr3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=kr3_progreso,
        title={'text': "KR3: Reducción en Costos de Mantenimiento"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#f39c12"},
            'steps': [
                {'range': [0, 15], 'color': "lightgray"},
                {'range': [15, 30], 'color': "lightyellow"},
                {'range': [30, 100], 'color': "rgba(243, 156, 18, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    fig_kr3.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_kr3, use_container_width=True)
    st.markdown(f"**Meta:** 30% de reducción en costos")

# Acciones recomendadas basadas en los datos
st.markdown('<div class="sub-header">Acciones Recomendadas</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("#### Acciones Prioritarias")
    st.markdown("""
    1. **Expandir la integración de sensores IoT** en todas las líneas críticas restantes.
    2. **Mejorar la precisión del modelo predictivo** para la Línea C, que muestra la menor tasa de prevención.
    3. **Implementar alertas automatizadas
