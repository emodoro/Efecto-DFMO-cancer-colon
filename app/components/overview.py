import streamlit as st
import plotly.express as px
from utils.plotting import create_summary_stats

def show_overview(expr_data, covariables):
    """
    Muestra la visión general del estudio y los datos.
    
    Args:
        expr_data (pd.DataFrame): Matriz de expresión génica
        covariables (pd.DataFrame): Matriz de covariables
    """
    st.header("Visión General del Estudio")
    
    # Información del estudio
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objetivo del Estudio")
        st.write("""
        Este estudio investiga el efecto del DFMO (difluorometilornitina) en:
        - Células de cáncer de colon (HT29)
        - Colonocitos normales (NCM460)
        """)
        
        st.subheader("🔬 Diseño Experimental")
        st.write("""
        - **Tecnología**: Microarrays Clariom D Human
        - **Condiciones**: Control vs DFMO
        - **Réplicas**: Múltiples por condición
        """)
    
    with col2:
        st.subheader("📊 Resumen de Datos")
        stats = create_summary_stats(expr_data, covariables)
        st.write(stats)
        
    # Distribución de muestras
    st.subheader("📈 Distribución de Muestras")
    fig = px.scatter(
        covariables,
        x="Linea",
        color="Tratamiento",
        title="Distribución de Muestras por Línea Celular y Tratamiento",
        category_orders={"Linea": ["NCM460", "HT29"]}
    )
    st.plotly_chart(fig, use_container_width=True)