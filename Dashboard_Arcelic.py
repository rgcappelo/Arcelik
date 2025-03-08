import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="Arçelik - Dashboard de Predicción de Fallas",
    page_icon="🏭",
    layout="wide"
)

# Función para cargar los datos
@st.cache_data
def load_data():
    try:
        # Intentar cargar el archivo CSV
        if os.path.exists("KPIs_normales_Arcelik.csv"):
            df = pd.read_csv("KPIs_normales_Arcelik.csv")
            
            # Mostrar información de columnas disponibles para depuración
            with st.sidebar.expander("Columnas disponibles"):
                st.write(df.columns.tolist())
            
            # Verificar y procesar columna de fecha
            date_columns = ['fecha', 'date', 'Fecha', 'Date', 'FECHA', 'DATE', 'fecha_medicion']
            date_col = next((col for col in date_columns if col in df.columns), None)
            
            if date_col:
                df['fecha'] = pd.to_datetime(df[date_col])
            else:
                st.warning("No se encontró columna de fecha. Creando fechas simuladas.")
                df['fecha'] = pd.date_range(start='2022-01-01', periods=len(df), freq='M')
            
            return df
        else:
            st.error(f"No se encontró el archivo 'KPIs_normales_Arcelik.csv' en el directorio actual.")
            st.info(f"Directorio actual: {os.getcwd()}")
            st.info(f"Archivos disponibles: {os.listdir()}")
            
            # Crear datos de muestra para evitar errores
            return create_sample_data()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return create_sample_data()

# Función para crear datos de muestra en caso de error
def create_sample_data():
    st.warning("Usando datos de muestra para demostración")
    
    # Fechas para los últimos 24 meses
    dates = pd.date_range(start='2022-01-01', periods=24, freq='M')
    
    # Crear datos de muestra para 3 máquinas
    machines = ['Máquina A', 'Máquina B', 'Máquina C']
    data = []
    
    for date in dates:
        for machine in machines:
            # Simular mejora en el tiempo (más fallas evitadas conforme pasa el tiempo)
            month_index = (date.year - 2022) * 12 + date.month - 1
            improvement_factor = min(0.8, month_index / 24)
            
            # Simular datos relevantes
            failures_occurred = max(1, 10 - int(8 * improvement_factor) + np.random.randint(-2, 3))
            failures_prevented = int(5 + 12 * improvement_factor + np.random.randint(-1, 2))
            
            # Costos de mantenimiento (decrecientes)
            maintenance_cost = max(1000, 5000 - 3000 * improvement_factor + np.random.normal(0, 500))
            
            # Precisión del modelo (creciente)
            model_accuracy = min(0.98, 0.75 + 0.2 * improvement_factor + np.random.normal(0, 0.02))
            
            # Valores de sensores
            temperature = 65 + np.random.normal(0, 5)
            vibration = 120 + np.random.normal(0, 10)
            energy = 450 + np.random.normal(0, 30)
            
            # Tiempos de respuesta (mejorando)
            response_time = max(5, 30 - 20 * improvement_factor + np.random.normal(0, 3))
            
            data.append({
                'fecha': date,
                'maquina': machine,
                'temperatura_equipo': round(temperature, 1),
                'vibraciones_anomalas': round(vibration, 2),
                'consumo_energia': round(energy, 2),
                'fallas_ocurridas': failures_occurred,
                'fallas_evitadas': failures_prevented,
                'costo_mantenimiento_correctivo': round(maintenance_cost, 2),
                'precisión_modelo': round(model_accuracy, 4),
                'tiempo_respuesta': round(response_time, 1)
            })
    
    return pd.DataFrame(data)

# Cargar datos
df = load_data()

# Sidebar para filtros
st.sidebar.title("Filtros")

# Visualizar las primeras filas para debug
with st.sidebar.expander("Vista previa de datos"):
    st.dataframe(df.head())

# Filtro de fecha
try:
    min_date = df['fecha'].min().to_pydatetime()
    max_date = df['fecha'].max().to_pydatetime()
    
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['fecha'] >= pd.Timestamp(start_date)) & 
                         (df['fecha'] <= pd.Timestamp(end_date))]
    else:
        filtered_df = df
except Exception as e:
    st.sidebar.error(f"Error en filtro de fechas: {e}")
    filtered_df = df

# Filtro de máquinas
try:
    machine_col = 'maquina'
    if machine_col in df.columns:
        machines = df[machine_col].unique()
        selected_machines = st.sidebar.multiselect(
            "Seleccionar máquinas",
            options=machines,
            default=machines
        )
        
        if selected_machines:
            filtered_df = filtered_df[filtered_df[machine_col].isin(selected_machines)]
    else:
        st.sidebar.warning(f"No se encontró la columna '{machine_col}' para filtrar máquinas")
except Exception as e:
    st.sidebar.error(f"Error en filtro de máquinas: {e}")

# Título principal del dashboard
st.title("🏭 Dashboard de Predicción de Fallas mediante Gemelos Digitales - Arçelik")

# Sección 1: Objetivo del OKR
st.header("1️⃣ Objetivo del OKR")
st.info("🎯 Predecir y reducir fallas en maquinaria en un 60% en dos años mediante simulaciones con gemelos digitales.")

# Sección 2: Key Results
st.header("2️⃣ Key Results (KR)")
kr_col1, kr_col2, kr_col3 = st.columns(3)

with kr_col1:
    st.metric(
        label="KR1",
        value="Modelos Predictivos",
        delta="100% líneas críticas"
    )
    st.caption("Implementar modelos predictivos en todas las líneas críticas en un año")

with kr_col2:
    st.metric(
        label="KR2",
        value="Reducción de Fallas",
        delta="70% en 24 meses"
    )
    st.caption("Lograr reducción del 70% en fallas no programadas")

with kr_col3:
    st.metric(
        label="KR3",
        value="Ahorro en Costos",
        delta="30% mantenimiento"
    )
    st.caption("Reducir en 30% los costos de mantenimiento correctivo")

# Sección 3: KPIs
st.header("3️⃣ Indicadores Clave (KPIs)")

# KPI 1: Fallas evitadas vs. ocurridas
st.subheader("📊 KPI 1: Número de fallas evitadas vs. fallas ocurridas")

# Cálculos para KPI 1
try:
    kpi1_data = filtered_df.groupby('fecha').agg({
        'fallas_ocurridas': 'sum',
        'fallas_evitadas': 'sum'
    }).reset_index()
    
    total_fallas_ocurridas = kpi1_data['fallas_ocurridas'].sum()
    total_fallas_evitadas = kpi1_data['fallas_evitadas'].sum()
    tasa_prevencion = (total_fallas_evitadas / (total_fallas_ocurridas + total_fallas_evitadas) * 100)
    
    # Cards para KPI 1
    kpi1_col1, kpi1_col2, kpi1_col3 = st.columns(3)
    
    with kpi1_col1:
        st.metric(
            label="Fallas Ocurridas",
            value=f"{total_fallas_ocurridas:,}"
        )
    
    with kpi1_col2:
        st.metric(
            label="Fallas Evitadas",
            value=f"{total_fallas_evitadas:,}"
        )
    
    with kpi1_col3:
        st.metric(
            label="Tasa de Prevención",
            value=f"{tasa_prevencion:.1f}%",
            delta=f"{tasa_prevencion - 50:.1f}%" if tasa_prevencion > 50 else None
        )
    
    with st.expander("📌 Detalles del KPI"):
        st.markdown("""
        **Construcción del KPI:** Este KPI compara la cantidad de fallas que ocurrieron con aquellas que fueron anticipadas y prevenidas por el modelo predictivo.
        
        **Importancia:** Permite evaluar la efectividad del mantenimiento predictivo.
        
        **Factores que afectan este KPI:** Precisión del modelo, tiempos de respuesta, calidad de los sensores IoT.
        """)
    
except Exception as e:
    st.error(f"Error al calcular KPI 1: {e}")

# KPI 2: Reducción en costos de mantenimiento
st.subheader("📊 KPI 2: Reducción en costos de mantenimiento correctivo")

try:
    # Determinar punto medio para comparar antes/después
    median_date = filtered_df['fecha'].min() + (filtered_df['fecha'].max() - filtered_df['fecha'].min()) / 2
    
    before_impl = filtered_df[filtered_df['fecha'] < median_date]
    after_impl = filtered_df[filtered_df['fecha'] >= median_date]
    
    avg_cost_before = before_impl['costo_mantenimiento_correctivo'].mean()
    avg_cost_after = after_impl['costo_mantenimiento_correctivo'].mean()
    cost_reduction_pct = ((avg_cost_before - avg_cost_after) / avg_cost_before * 100) if avg_cost_before > 0 else 0
    
    # Cards para KPI 2
    kpi2_col1, kpi2_col2, kpi2_col3 = st.columns(3)
    
    with kpi2_col1:
        st.metric(
            label="Costo Medio Antes",
            value=f"${avg_cost_before:,.2f}"
        )
    
    with kpi2_col2:
        st.metric(
            label="Costo Medio Después",
            value=f"${avg_cost_after:,.2f}",
            delta=f"-${avg_cost_before - avg_cost_after:,.2f}",
            delta_color="inverse"
        )
    
    with kpi2_col3:
        st.metric(
            label="Reducción de Costos",
            value=f"{cost_reduction_pct:.1f}%"
        )
    
    with st.expander("📌 Detalles del KPI"):
        st.markdown("""
        **Construcción del KPI:** Este indicador mide la diferencia en costos de mantenimiento correctivo antes y después de la implementación del sistema de predicción.
        
        **Importancia:** Muestra el impacto financiero del mantenimiento predictivo en la reducción de costos operativos.
        
        **Factores que afectan este KPI:** Frecuencia de fallas, costos de repuestos, eficiencia en la logística de mantenimiento.
        """)
    
except Exception as e:
    st.error(f"Error al calcular KPI 2: {e}")

# KPI 3: Precisión del modelo predictivo
st.subheader("📊 KPI 3: Precisión del modelo predictivo")

try:
    if 'precisión_modelo' in filtered_df.columns:
        avg_accuracy = filtered_df['precisión_modelo'].mean()
        min_accuracy = filtered_df['precisión_modelo'].min()
        max_accuracy = filtered_df['precisión_modelo'].max()
        
        # Cards para KPI 3
        kpi3_col1, kpi3_col2, kpi3_col3 = st.columns(3)
        
        with kpi3_col1:
            st.metric(
                label="Precisión Media",
                value=f"{avg_accuracy:.1%}"
            )
        
        with kpi3_col2:
            st.metric(
                label="Precisión Mínima",
                value=f"{min_accuracy:.1%}"
            )
        
        with kpi3_col3:
            st.metric(
                label="Precisión Máxima",
                value=f"{max_accuracy:.1%}"
            )
    else:
        # Calcular la precisión como porcentaje de fallas evitadas vs total de potenciales fallas
        kpi3_data = filtered_df.groupby('fecha').agg({
            'fallas_ocurridas': 'sum',
            'fallas_evitadas': 'sum'
        }).reset_index()
        
        kpi3_data['precisión_modelo'] = kpi3_data['fallas_evitadas'] / (kpi3_data['fallas_evitadas'] + kpi3_data['fallas_ocurridas'])
        
        avg_accuracy = kpi3_data['precisión_modelo'].mean()
        min_accuracy = kpi3_data['precisión_modelo'].min()
        max_accuracy = kpi3_data['precisión_modelo'].max()
        
        # Cards para KPI 3
        kpi3_col1, kpi3_col2, kpi3_col3 = st.columns(3)
        
        with kpi3_col1:
            st.metric(
                label="Precisión Media",
                value=f"{avg_accuracy:.1%}"
            )
        
        with kpi3_col2:
            st.metric(
                label="Precisión Mínima",
                value=f"{min_accuracy:.1%}"
            )
        
        with kpi3_col3:
            st.metric(
                label="Precisión Máxima",
                value=f"{max_accuracy:.1%}"
            )
    
    with st.expander("📌 Detalles del KPI"):
        st.markdown("""
        **Construcción del KPI:** Mide la capacidad del modelo de predicción para anticipar fallas correctamente. Se evalúa usando métricas como Accuracy, Sensitivity, Specificity y ROC-AUC.
        
        **Importancia:** Indica la confiabilidad del modelo en la toma de decisiones estratégicas.
        
        **Factores que afectan este KPI:** Calidad de los datos de entrenamiento, ajuste de hiperparámetros, variabilidad en los datos operativos.
        """)
    
except Exception as e:
    st.error(f"Error al calcular KPI 3: {e}")

# Sección 4: Gráficos del Dashboard
st.header("4️⃣ Gráficos del Dashboard")

# Gráfico 1: Evolución de fallas ocurridas y prevenidas
st.subheader("📊 Gráfico 1: Evolución de fallas ocurridas y prevenidas")

try:
    # Agregación por fecha para mostrar la evolución temporal
    g1_data = filtered_df.groupby('fecha').agg({
        'fallas_ocurridas': 'sum',
        'fallas_evitadas': 'sum'
    }).reset_index()
    
    # Gráfico de líneas con Plotly
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=g1_data['fecha'],
        y=g1_data['fallas_ocurridas'],
        mode='lines+markers',
        name='Fallas Ocurridas',
        line=dict(color='#FF5733', width=2),
        marker=dict(size=8)
    ))
    
    fig1.add_trace(go.Scatter(
        x=g1_data['fecha'],
        y=g1_data['fallas_evitadas'],
        mode='lines+markers',
        name='Fallas Evitadas',
        line=dict(color='#33A8FF', width=2),
        marker=dict(size=8)
    ))
    
    fig1.update_layout(
        title='¿Cómo ha evolucionado la cantidad de fallas ocurridas y prevenidas en el tiempo?',
        xaxis_title='Fecha',
        yaxis_title='Número de Fallas',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    with st.expander("💡 Explicación"):
        st.markdown("""
        Este gráfico muestra la evolución mensual de fallas en maquinaria y cuántas fueron anticipadas mediante el modelo predictivo.
        
        Podemos observar cómo ha ido cambiando la relación entre las fallas que ocurrieron realmente (línea roja) y las que el sistema pudo evitar (línea azul) a lo largo del tiempo, evidenciando la mejora en la efectividad del modelo predictivo.
        """)
    
except Exception as e:
    st.error(f"Error al generar Gráfico 1: {e}")

# Gráfico 2: Tendencia del costo de mantenimiento correctivo
st.subheader("📊 Gráfico 2: Tendencia del costo de mantenimiento correctivo")

try:
    # Agregación por fecha y máquina para los costos
    g2_data = filtered_df.groupby(['fecha', 'maquina']).agg({
        'costo_mantenimiento_correctivo': 'mean'
    }).reset_index()
    
    # Gráfico de barras con Plotly
    fig2 = px.bar(
        g2_data,
        x='fecha',
        y='costo_mantenimiento_correctivo',
        color='maquina',
        title='¿Cuál es la tendencia del costo de mantenimiento correctivo después de la implementación?',
        labels={
            'fecha': 'Fecha',
            'costo_mantenimiento_correctivo': 'Costo de Mantenimiento ($)',
            'maquina': 'Máquina'
        }
    )
    
    # Añadir línea de implementación (fecha media)
    median_date = filtered_df['fecha'].min() + (filtered_df['fecha'].max() - filtered_df['fecha'].min()) / 2
    
    fig2.add_vline(
        x=median_date,
        line_dash="dash",
        line_color="red",
        annotation_text="Implementación del Modelo",
        annotation_position="top right"
    )
    
    fig2.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Costo de Mantenimiento ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("💡 Explicación"):
        st.markdown("""
        Este gráfico visualiza la reducción de costos operativos, demostrando el impacto financiero de la estrategia predictiva.
        
        La línea roja punteada marca la fecha aproximada de implementación del modelo predictivo. Se puede observar cómo los costos tienden a reducirse después de esta implementación, evidenciando el retorno de inversión del proyecto.
        """)
    
except Exception as e:
    st.error(f"Error al generar Gráfico 2: {e}")

# Gráfico 3: Relación entre vibraciones y fallas
st.subheader("📊 Gráfico 3: Relación entre vibraciones y fallas")

try:
    # Scatter plot para vibración vs fallas
    fig3 = px.scatter(
        filtered_df,
        x='vibraciones_anomalas',
        y='fallas_ocurridas',
        color='maquina',
        size='temperatura_equipo',
        hover_data=['fecha', 'consumo_energia'],
        title='¿Cuál es la relación entre las vibraciones de la maquinaria y la cantidad de fallas ocurridas?',
        labels={
            'vibraciones_anomalas': 'Vibraciones Anómalas (Hz)',
            'fallas_ocurridas': 'Número de Fallas',
            'maquina': 'Máquina',
            'temperatura_equipo': 'Temperatura (°C)'
        }
    )
    
    fig3.update_layout(
        xaxis_title='Vibraciones Anómalas (Hz)',
        yaxis_title='Número de Fallas',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    with st.expander("💡 Explicación"):
        st.markdown("""
        Este gráfico muestra la relación entre las vibraciones anómalas y las fallas detectadas, permitiendo validar la influencia de este factor en la predicción.
        
        Cada punto representa una medición, donde el eje X muestra el nivel de vibraciones anómalas y el eje Y el número de fallas ocurridas. El tamaño del punto representa la temperatura del equipo, permitiendo visualizar la interacción entre estos tres factores críticos para la predicción de fallas.
        """)
    
except Exception as e:
    st.error(f"Error al generar Gráfico 3: {e}")

# Gráfico 4: Precisión del modelo a lo largo del tiempo
st.subheader("📊 Gráfico 4: Precisión del modelo de predicción a lo largo del tiempo")

try:
    # Verificar si existe la columna de precisión, si no, calcularla
    if 'precisión_modelo' in filtered_df.columns:
        # Agrupar por fecha para ver evolución temporal
        g4_data = filtered_df.groupby(['fecha', 'maquina']).agg({
            'precisión_modelo': 'mean'
        }).reset_index()
    else:
        # Calcular precisión como fallas evitadas / (evitadas + ocurridas)
        g4_data = filtered_df.groupby(['fecha', 'maquina']).apply(
            lambda x: pd.Series({
                'precisión_modelo': x['fallas_evitadas'].sum() / (x['fallas_evitadas'].sum() + x['fallas_ocurridas'].sum())
                if (x['fallas_evitadas'].sum() + x['fallas_ocurridas'].sum()) > 0 else 0
            })
        ).reset_index()
    
    # Crear gráfico de líneas
    fig4 = px.line(
        g4_data,
        x='fecha',
        y='precisión_modelo',
        color='maquina',
        title='¿Cómo varía la precisión del modelo de predicción de fallas a lo largo del tiempo?',
        labels={
            'fecha': 'Fecha',
            'precisión_modelo': 'Precisión del Modelo',
            'maquina': 'Máquina'
        }
    )
    
    # Añadir rangos de referencia
    fig4.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Precisión Objetivo Mínima")
    fig4.add_hline(y=0.9, line_dash="dash", line_color="green", annotation_text="Precisión Objetivo Óptima")
    
    fig4.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Precisión del Modelo (%)',
        yaxis=dict(
            tickformat='.0%'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    with st.expander("💡 Explicación"):
        st.markdown("""
        Este gráfico ilustra la precisión del modelo predictivo, evaluando su estabilidad y efectividad en distintos períodos.
        
        Podemos observar la evolución de la precisión del modelo para cada máquina a lo largo del tiempo. Las líneas de referencia naranja y verde indican los niveles objetivo de precisión (mínimo y óptimo, respectivamente) para considerar que el modelo es efectivo en la predicción de fallas.
        """)
    
except Exception as e:
    st.error(f"Error al generar Gráfico 4: {e}")

# Sección 5: Datos Necesarios y Fuentes
st.header("5️⃣ Datos Necesarios y Cómo Fueron Obtenidos")

with st.expander("Ver tabla de variables y fuentes de datos"):
    data_sources = pd.DataFrame({
        'Variable': ['Fecha de medición', 'Máquina monitoreada', 'Temperatura del equipo (°C)', 
                    'Vibraciones anómalas (Hz)', 'Consumo de energía (kWh)', 'Número de fallas detectadas',
                    'Número de fallas evitadas', 'Costo de mantenimiento correctivo ($)', 'Tiempo medio de respuesta (min)'],
        'Fuente de Datos': ['Sensores IoT y registros de mantenimiento', 'Inventario de equipos en la planta', 
                            'Sensores IoT', 'Sensores IoT (acelerómetros)', 'Sensores IoT', 'Reportes de mantenimiento', 
                            'Gemelo digital (simulaciones)', 'ERP, Finanzas', 'Sistema de monitoreo'],
        'Método de Obtención': ['Captura en tiempo real', 'Relación con datos IoT', 'Media diaria/mensual', 
                               'Análisis en tiempo real', 'Registro automático', 'Registro de fallas', 
                               'Comparación con histórico', 'Reportes financieros', 'Captura en tiempo real'],
        'Transformaciones Necesarias': ['Ninguna, solo formato de tiempo', 'Asociar con ID de máquina', 'Cálculo de desviaciones', 
                                       'Promedio mensual', 'Media y varianza mensual', 'Frecuencia de ocurrencia', 
                                       'Calcular % de reducción', 'Comparación pre/post intervención', 'Cálculo de tiempos promedio']
    })
    
    st.table(data_sources)

# Sección 6: Acciones Necesarias
st.header("6️⃣ Acciones Necesarias")

action_col1, action_col2 = st.columns(2)

with action_col1:
    st.info("🛠️ Integrar sensores IoT en toda la maquinaria crítica para mejorar la recolección de datos en tiempo real.")
    st.info("🛠️ Optimizar los modelos predictivos ajustando hiperparámetros y evaluando nuevas arquitecturas de machine learning.")
    st.info("🛠️ Capacitar a los operadores y equipos de mantenimiento para interpretar correctamente los datos y responder a alertas predictivas.")

with action_col2:
    st.info("🛠️ Revisar periódicamente la precisión del modelo y actualizarlo con datos más recientes.")
    st.info("🛠️ Implementar un sistema de alertas visuales y notificaciones dentro del dashboard para advertencias críticas.")

# Vista de datos y descarga
st.header("Vista de Datos")

with st.expander("Ver datos utilizados"):
    st.dataframe(filtered_df)

# Botón de descarga de datos
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar Datos en CSV",
    data=csv,
    file_name="arcelik_datos_gemelo_digital.csv",
    mime="text/csv",
)

# Pie de página
st.markdown("---")
st.markdown("*Dashboard desarrollado para la visualización de KPIs de predicción de fallas mediante gemelos digitales para Arçelik*")
