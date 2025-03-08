import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de KPIs Arçelik",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("Dashboard de KPIs - Arçelik")
st.markdown("### Sistema de Monitoreo de Gemelos Digitales para Detección de Fallas")

# Sidebar con información de OKR y KRs
with st.sidebar:
    st.header("Objetivo del OKR")
    st.info("📌 Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.")
    
    st.header("Key Results (KR)")
    st.success("✅ **KR1:** Implementar modelos predictivos de fallas en el **100% de las líneas críticas** en un año.")
    st.success("✅ **KR2:** Lograr una reducción del **70% en fallas no programadas** en 24 meses.")
    st.success("✅ **KR3:** Obtener una reducción del **30% en costos de mantenimiento correctivo**.")
    
    st.header("Pregunta clave")
    st.markdown("🧐 **¿Cómo impacta el mantenimiento predictivo en la reducción de fallas?**")

# Generación de datos simulados
def generate_mock_data():
    # Fechas para los últimos 36 meses + 6 proyectados
    dates = pd.date_range(start='2022-01-01', periods=42, freq='MS')
    
    # Fallas ocurridas (con tendencia decreciente después de implementación)
    fallas_ocurridas = [12, 14, 10, 15, 13, 16, 12, 18, 14, 15, 17, 16, 
                        14, 12, 11, 10, 9, 8, 7, 9, 8, 6, 7, 5,
                        4, 3, 4, 3, 2, 3, 2, 1, 2, 2, 1, 1,
                        # Valores proyectados
                        1, 1, 0, 1, 0, 1]
    
    # Fallas prevenidas (aumentando con el tiempo a medida que el modelo mejora)
    fallas_prevenidas = [2, 3, 1, 4, 3, 5, 3, 6, 4, 5, 6, 7,
                         8, 10, 12, 14, 16, 18, 20, 19, 21, 23, 24, 26,
                         28, 29, 30, 31, 32, 33, 34, 36, 35, 37, 38, 39,
                         # Valores proyectados
                         40, 41, 42, 43, 44, 45]
    
    # Consumo de energía (kWh) (disminuyendo gradualmente)
    consumo_energia = [200, 210, 205, 215, 220, 225, 218, 230, 225, 228, 232, 230,
                       225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170,
                       165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 125, 120,
                       # Valores proyectados
                       118, 115, 112, 110, 108, 105]
    
    # Vibraciones (Hz) (disminuyendo gradualmente, correlacionado con fallas)
    vibraciones = [1.5, 1.6, 1.5, 1.7, 1.8, 1.9, 1.8, 2.0, 1.9, 1.8, 1.7, 1.6,
                  1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5,
                  0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2,
                  # Valores proyectados
                  0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    
    # Costo de mantenimiento correctivo ($)
    costo_mantenimiento = [8000, 8500, 8200, 9000, 8800, 9500, 9200, 10000, 9800, 9500, 10200, 10000,
                           9500, 9000, 8500, 8000, 7500, 7000, 6500, 6000, 5500, 5200, 5000, 4800,
                           4600, 4500, 4400, 4200, 4000, 4100, 4000, 3900, 3800, 3700, 3600, 3500,
                           # Valores proyectados
                           3400, 3300, 3200, 3100, 3000, 2900]
    
    # Crear DataFrame
    data = pd.DataFrame({
        'fecha': dates,
        'fallas_ocurridas': fallas_ocurridas,
        'fallas_prevenidas': fallas_prevenidas,
        'consumo_energia': consumo_energia,
        'vibraciones': vibraciones,
        'costo_mantenimiento': costo_mantenimiento,
    })
    
    # Añadir columnas derivadas
    data['mes_año'] = data['fecha'].dt.strftime('%Y-%m')
    data['año'] = data['fecha'].dt.year
    data['mes'] = data['fecha'].dt.month
    data['es_proyeccion'] = data.index >= 36
    data['pre_implementacion'] = data.index < 12  # Primeros 12 meses antes de la implementación
    
    # Calcular eficiencia del modelo
    data['eficiencia_modelo'] = (data['fallas_prevenidas'] / (data['fallas_ocurridas'] + data['fallas_prevenidas']) * 100).round(1)
    data.loc[data['eficiencia_modelo'].isna(), 'eficiencia_modelo'] = 0
    
    return data

# Generar datos y guardarlos en session_state para evitar regeneración en cada interacción
if 'data' not in st.session_state:
    st.session_state.data = generate_mock_data()

data = st.session_state.data

# Filtros interactivos
col1, col2, col3 = st.columns(3)

with col1:
    # Selector de rango de fechas
    start_date = st.date_input("Fecha de inicio", 
                               value=pd.to_datetime('2022-01-01'), 
                               min_value=pd.to_datetime('2022-01-01'),
                               max_value=pd.to_datetime('2025-09-01'))

with col2:
    end_date = st.date_input("Fecha de fin", 
                             value=pd.to_datetime('2025-09-01'), 
                             min_value=pd.to_datetime('2022-01-01'),
                             max_value=pd.to_datetime('2025-09-01'))

with col3:
    # Selector para incluir proyecciones
    include_projections = st.checkbox("Incluir proyecciones", value=True)

# Filtrar datos según los inputs del usuario
filtered_data = data[(data['fecha'] >= pd.Timestamp(start_date)) & 
                     (data['fecha'] <= pd.Timestamp(end_date))]

if not include_projections:
    filtered_data = filtered_data[filtered_data['es_proyeccion'] == False]

# KPIs principales en tarjetas
st.markdown("## Resumen de KPIs")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Calcular valores para KPIs
total_fallas_evitadas = filtered_data['fallas_prevenidas'].sum()
total_fallas_ocurridas = filtered_data['fallas_ocurridas'].sum()
porcentaje_reduccion = ((filtered_data[filtered_data['pre_implementacion'] == True]['fallas_ocurridas'].mean() - 
                         filtered_data[filtered_data['pre_implementacion'] == False]['fallas_ocurridas'].mean()) / 
                         filtered_data[filtered_data['pre_implementacion'] == True]['fallas_ocurridas'].mean() * 100)
reduccion_costos = ((filtered_data[filtered_data['pre_implementacion'] == True]['costo_mantenimiento'].mean() - 
                    filtered_data[filtered_data['pre_implementacion'] == False]['costo_mantenimiento'].mean()) / 
                    filtered_data[filtered_data['pre_implementacion'] == True]['costo_mantenimiento'].mean() * 100)
eficiencia_promedio = filtered_data['eficiencia_modelo'].mean()

# Mostrar KPIs en tarjetas
with kpi1:
    st.metric("Fallas Evitadas", f"{int(total_fallas_evitadas)}", 
              delta=f"{int(total_fallas_evitadas - total_fallas_ocurridas)} vs. Ocurridas")

with kpi2:
    st.metric("Reducción de Fallas", f"{porcentaje_reduccion:.1f}%", 
              delta="Desde implementación")

with kpi3:
    st.metric("Reducción Costos Mant.", f"{reduccion_costos:.1f}%", 
              delta="Desde implementación")

with kpi4:
    st.metric("Eficiencia del Modelo", f"{eficiencia_promedio:.1f}%", 
              delta="Predicción de fallas")

# Primer gráfico: Tendencia de fallas ocurridas vs prevenidas
st.markdown("## 1️⃣ Tendencia de Fallas Ocurridas vs. Fallas Prevenidas")

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(filtered_data['fecha'], filtered_data['fallas_ocurridas'], 'b-', linewidth=2, label='Fallas Ocurridas')
ax1.plot(filtered_data['fecha'], filtered_data['fallas_prevenidas'], 'g--', linewidth=2, label='Fallas Prevenidas')

# Marcar área de proyección si está incluida
if include_projections and any(filtered_data['es_proyeccion']):
    proyeccion_start = filtered_data[filtered_data['es_proyeccion']]['fecha'].min()
    ax1.axvline(x=proyeccion_start, color='gray', linestyle='--', alpha=0.7)
    ax1.text(proyeccion_start, ax1.get_ylim()[1] * 0.9, ' Inicio de proyecciones', 
             color='gray', fontsize=10, verticalalignment='top')

# Ajustar formato del eje X
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Añadir títulos y leyenda
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Número de Fallas')
ax1.set_title('Fallas Ocurridas vs. Fallas Prevenidas', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mostrar el punto de implementación del modelo predictivo
implementation_date = pd.Timestamp('2023-01-01')
if implementation_date >= filtered_data['fecha'].min() and implementation_date <= filtered_data['fecha'].max():
    ax1.axvline(x=implementation_date, color='red', linestyle='-', alpha=0.5)
    ax1.text(implementation_date, ax1.get_ylim()[1] * 0.95, ' Implementación del modelo predictivo', 
             color='red', fontsize=10, verticalalignment='top')

# Añadir anotación para la meta del OKR
ax1.text(0.02, 0.02, 'Meta OKR: Reducción del 60% en fallas', 
         transform=ax1.transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

st.pyplot(fig1)

# Segundo gráfico: Relación entre Consumo de Energía y Vibraciones
st.markdown("## 2️⃣ Relación entre Consumo de Energía y Vibraciones")

fig2, ax2 = plt.subplots(figsize=(12, 6))
scatter = ax2.scatter(filtered_data['consumo_energia'], 
                     filtered_data['vibraciones'],
                     c=filtered_data['vibraciones'], 
                     cmap='YlOrRd', 
                     s=100, 
                     alpha=0.7)

# Añadir línea de tendencia
z = np.polyfit(filtered_data['consumo_energia'], filtered_data['vibraciones'], 1)
p = np.poly1d(z)
ax2.plot(filtered_data['consumo_energia'], p(filtered_data['consumo_energia']), 
         "r--", alpha=0.7, label=f"Tendencia (y={z[0]:.4f}x+{z[1]:.4f})")

# Añadir colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Nivel de Vibraciones (Hz)')

# Añadir etiquetas para algunos puntos
for i, row in filtered_data.iloc[::6].iterrows():  # Etiqueta cada 6 puntos para no sobrecargar
    ax2.annotate(row['mes_año'], 
                (row['consumo_energia'], row['vibraciones']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8)

# Ajustar títulos y etiquetas
ax2.set_xlabel('Consumo de Energía (kWh)')
ax2.set_ylabel('Vibraciones (Hz)')
ax2.set_title('Relación entre Consumo de Energía y Vibraciones', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Añadir anotación explicativa
ax2.text(0.02, 0.96, 'Mayor vibración = Mayor riesgo de falla', 
         transform=ax2.transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightskyblue', alpha=0.5))

st.pyplot(fig2)

# Tercer gráfico: Costo de Mantenimiento Correctivo por Año
st.markdown("## 3️⃣ Costo de Mantenimiento Correctivo por Año")

# Preparar datos para el gráfico de barras
yearly_cost = filtered_data.groupby(['año', 'pre_implementacion'])['costo_mantenimiento'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(12, 6))

# Definir colores
colores = ['#ff9999', '#66b3ff']  # rojo claro para pre-implementación, azul para post-implementación

# Crear barras
sns.barplot(x='año', y='costo_mantenimiento', hue='pre_implementacion', 
            data=yearly_cost, palette=colores, ax=ax3)

# Añadir etiquetas encima de las barras
for i, p in enumerate(ax3.patches):
    height = p.get_height()
    ax3.text(p.get_x() + p.get_width()/2., height + 100,
            f'${height:.0f}',
            ha='center', fontsize=10)

# Personalizar leyenda
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, ['Post-implementación', 'Pre-implementación'], 
           title='Estado del Modelo', loc='upper right')

# Ajustar títulos y etiquetas
ax3.set_xlabel('Año')
ax3.set_ylabel('Costo Promedio de Mantenimiento Correctivo ($)')
ax3.set_title('Costo de Mantenimiento Correctivo por Año', fontsize=14)

# Añadir línea de meta OKR
promedio_pre = yearly_cost[yearly_cost['pre_implementacion']]['costo_mantenimiento'].mean()
meta_reduccion = promedio_pre * 0.7  # 30% de reducción
ax3.axhline(y=meta_reduccion, color='green', linestyle='--', alpha=0.7)
ax3.text(0.02, 0.04, 'Meta OKR: 30% reducción de costos', 
         transform=ax3.transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))

st.pyplot(fig3)

# Tabla de datos filtrados
with st.expander("Ver datos detallados"):
    st.dataframe(filtered_data[['mes_año', 'fallas_ocurridas', 'fallas_prevenidas', 
                               'consumo_energia', 'vibraciones', 'costo_mantenimiento', 
                               'eficiencia_modelo', 'es_proyeccion']])

# Sección de acciones recomendadas
st.markdown("## Acciones Recomendadas")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    🔹 **Ampliar el uso de sensores IoT** para capturar datos en tiempo real en todas las líneas de producción restantes.
    
    🔹 **Ajustar modelos predictivos con datos más precisos** y entrenamiento continuo para mantener la efectividad.
    """)

with col2:
    st.info("""
    🔹 **Integrar dashboards de visualización con alertas automáticas** para mantenimiento proactivo basado en los patrones identificados.
    
    🔹 **Expandir el programa a otras plantas** utilizando la metodología ya probada.
    """)

# Información del dashboard
st.markdown("---")
st.caption("Dashboard de KPIs Arçelik - Sistema de Gemelos Digitales | Última actualización: Marzo 2025")
