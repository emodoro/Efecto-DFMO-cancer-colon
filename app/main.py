import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# App version - update to force Streamlit Cloud cache refresh
APP_VERSION = "1.0.1"

# Importar componentes
from components.sidebar import create_sidebar
from components.overview import show_overview
from components.data_explorer import show_data_explorer
from components.analysis import show_analysis
from components.conclusions import show_conclusions
from utils.data_loader import load_data
from utils.plotting import *

# Configuración de la página
st.set_page_config(
    page_title="Análisis DFMO en Cáncer de Colon",
    page_icon="🧬",
    layout="wide"
)

# Cargar CSS personalizado
def load_css():
    css_file = Path(__file__).parent / "styles" / "main.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Cargar CSS
    load_css()
    
    # Título principal
    st.title("🧬 Análisis del Efecto DFMO en Cáncer de Colon")
    
    # Cargar datos
    data_path = Path(__file__).parent.parent / "data"
    expr_data = load_data(data_path / "matriz_expr_symbol_median.csv")
    covariables = load_data(data_path / "matriz_covariables_ordenada.csv")
    
    # Crear sidebar con filtros básicos primero (sin n_top_genes aún)
    from utils.data_loader import filter_data
    
    # Crear filtros básicos en sidebar
    st.sidebar.title("Filtros y Opciones")
    st.sidebar.header("Filtros de Datos")
    
    # Filtro de línea celular
    cell_lines = st.sidebar.multiselect(
        "Línea Celular",
        ["HT29", "NCM460"],
        default=["HT29", "NCM460"]
    )
    
    # Filtro de tratamiento
    treatments = st.sidebar.multiselect(
        "Tratamiento",
        ["Control", "DFMO"],
        default=["Control", "DFMO"]
    )
    
    # Aplicar filtrado básico para calcular genes disponibles
    basic_filters = {
        "cell_lines": cell_lines,
        "treatments": treatments,
        "normalization": "Sin normalizar",
        "clustering_method": "Ward",
        "n_top_genes": 50
    }
    expr_filtered_temp, _ = filter_data(expr_data, covariables, basic_filters)
    n_genes_after_filter = len(expr_filtered_temp)
    
    # Ahora crear el resto de opciones del sidebar con el número correcto
    st.sidebar.header("Opciones de Visualización")
    
    # Tipo de normalización global
    normalization = st.sidebar.selectbox(
        "Normalización",
        ["Sin normalizar", "Z-score", "Min-Max"],
        help="Método de normalización aplicado a los datos"
    )
    
    # Sección específica para heatmap clusterizado
    st.sidebar.header("Parámetros de Heatmap Clusterizado")
    
    # Método de clustering
    clustering_method = st.sidebar.selectbox(
        "Método de Clustering",
        ["Ward", "Complete", "Average"],
        help="Algoritmo de clustering jerárquico para el heatmap"
    )
    
    # Número de genes top
    n_top_genes = st.sidebar.slider(
        "Número de genes",
        min_value=10,
        max_value=n_genes_after_filter,
        value=min(50, n_genes_after_filter),
        step=10,
        help=f"Genes con mayor varianza a mostrar en el heatmap clusterizado (de 10 a {n_genes_after_filter})"
    )
    
    # Crear diccionario de filtros completo
    filters = {
        "cell_lines": cell_lines,
        "treatments": treatments,
        "normalization": normalization,
        "clustering_method": clustering_method,
        "n_top_genes": n_top_genes
    }
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visión General",
        "🔍 Explorador de Datos",
        "📈 Análisis",
        "📑 Conclusiones"
    ])
    
    # Contenido de las pestañas
    with tab1:
        show_overview(expr_data, covariables)
    
    with tab2:
        show_data_explorer(expr_data, covariables, filters)
    
    with tab3:
        show_analysis(expr_data, covariables, filters)
    
    with tab4:
        # Pasar datos para conclusiones dinámicas
        show_conclusions(expr_data, covariables)

if __name__ == "__main__":
    main()