import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from utils.plotting import create_summary_stats
from utils.data_loader import non_specific_gene_filter

def show_overview(expr_data, covariables):
    """
    Muestra la visión general del estudio y los datos.
    
    Args:
        expr_data (pd.DataFrame): Matriz de expresión génica (ya agregada por mediana)
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
        
        # Cargar las matrices originales para mostrar las estadísticas
        data_path = Path(__file__).parent.parent.parent / "data"
        
        try:
            # Matriz original (sin filtrar)
            matriz_original = pd.read_csv(data_path / "matriz_expresion.csv", index_col=0)
            n_genes_original = len(matriz_original)
            
            # Matriz con gene symbols (agregada por mediana)
            n_genes_symbol = len(expr_data)
            
            # Matriz tras filtrado no específico
            expr_filtered = non_specific_gene_filter(expr_data)
            n_genes_filtered = len(expr_filtered)
            
            # Crear tabla de resumen
            stats_data = {
                "Métrica": [
                    "Genes iniciales (matriz sin filtrar)",
                    "Genes alineados con gene symbol",
                    "Genes tras filtrado no específico",
                    "Número de muestras"
                ],
                "Valor": [
                    f"{n_genes_original:,}",
                    f"{n_genes_symbol:,}",
                    f"{n_genes_filtered:,}",
                    len(expr_data.columns)
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al cargar estadísticas: {str(e)}")
            # Fallback al resumen simple
            stats = create_summary_stats(expr_data, covariables)
            st.write(stats)
        
    # Distribución de muestras
    st.subheader("📈 Distribución de Muestras")
    
    # Crear una columna combinada para el gráfico de barras
    cov_plot = covariables.copy()
    cov_plot['Grupo'] = cov_plot['Tratamiento'] + ' - ' + cov_plot['Linea']
    
    # Asegurar que Día sea tratado como categórico
    cov_plot['Dia'] = cov_plot['Dia'].astype(str)
    
    # Contar muestras por grupo y día
    count_data = cov_plot.groupby(['Grupo', 'Dia']).size().reset_index(name='Número de Muestras')
    
    fig = px.bar(
        count_data,
        x="Grupo",
        y="Número de Muestras",
        color="Dia",
        barmode="group",
        title="Distribución de Muestras por Línea Celular, Tratamiento y Día",
        text="Número de Muestras",
        category_orders={"Dia": sorted(cov_plot['Dia'].unique())}
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Tratamiento - Línea Celular",
        yaxis_title="Número de Muestras",
        showlegend=True,
        legend_title="Día"
    )
    
    st.plotly_chart(fig, use_container_width=True)