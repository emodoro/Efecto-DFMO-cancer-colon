import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

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
    
    # Crear sidebar y obtener filtros
    filters = create_sidebar()
    
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
        show_conclusions()

if __name__ == "__main__":
    main()