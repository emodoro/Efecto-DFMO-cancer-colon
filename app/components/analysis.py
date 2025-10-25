import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils.data_loader import filter_data, normalize_data
from utils.analysis_utils import (
    perform_pca_analysis,
    create_clustered_heatmap,
    calculate_differential_expression,
    calculate_gene_correlations
)

def show_analysis(expr_data, covariables, filters):
    """
    Muestra la sección de análisis avanzado.
    """
    st.header("📈 Análisis Avanzado")
    
    # Solo filtrar datos (la normalización se hará específicamente en cada análisis)
    expr_filtered, cov_filtered = filter_data(expr_data, covariables, filters)
    
    # Selector de tipo de análisis
    analysis_type = st.selectbox(
        "Tipo de Análisis",
        ["PCA", "Heatmap Clusterizado"]
    )
    
    if analysis_type == "PCA":
        show_pca_analysis(expr_filtered, cov_filtered, filters)
    else:
        show_clustered_heatmap(expr_filtered, cov_filtered)

def show_pca_analysis(expr_data, covariables, filters):
    """
    Muestra el análisis de PCA.
    """
    st.subheader("Análisis de Componentes Principales")
    
    # Convertir la normalización de la barra lateral al formato del PCA
    norm_method = {
        "Sin normalizar": "Ninguna",
        "Z-score": "Z-score",
        "Min-Max": "Min-Max"
    }[filters['normalization']]
    
    # Realizar PCA
    results = perform_pca_analysis(expr_data, covariables, norm_method=norm_method)
    var_exp = results['var_exp']
    scores = results['scores']
    
    # Crear DataFrame base con información de muestras
    df_pca = pd.DataFrame({
        'Linea': covariables["Linea"],
        'Trt': covariables["Tratamiento"],
        'TrtLinea': covariables["Tratamiento"] + " - " + covariables["Linea"],
        'Dia': covariables["Dia"]
    })
    
    # Selectores para las componentes principales
    st.subheader("Selección de Componentes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pc_x = st.selectbox(
            "Componente X",
            options=range(1, len(var_exp) + 1),
            format_func=lambda x: f"PC{x} ({(var_exp[x-1]*100).round(2)}%)",
            index=0
        )
    
    with col2:
        pc_y = st.selectbox(
            "Componente Y",
            options=range(1, len(var_exp) + 1),
            format_func=lambda x: f"PC{x} ({(var_exp[x-1]*100).round(2)}%)",
            index=1
        )
    
    with col3:
        pc_corr = st.selectbox(
            "PC para correlaciones",
            options=range(1, len(var_exp) + 1),
            format_func=lambda x: f"PC{x} ({(var_exp[x-1]*100).round(2)}%)",
            index=0,
            help="Componente principal para calcular correlaciones con genes"
        )
    
    # Obtener los scores de las PC seleccionadas
    df_plot = df_pca.copy()
    df_plot[f'PC{pc_x}'] = scores[:, pc_x-1] * (-1 if pc_x == 1 else 1)  # Cambiar signo solo para PC1
    df_plot[f'PC{pc_y}'] = scores[:, pc_y-1]
    
    # Calcular correlaciones con la PC seleccionada
    pc_scores = scores[:, pc_corr-1] * (-1 if pc_corr == 1 else 1)  # Cambiar signo solo para PC1
    gene_correlations = calculate_gene_correlations(expr_data, pc_scores)
    
    # Crear visualización de PCA
    fig = px.scatter(
        df_plot,
        x=f'PC{pc_x}',
        y=f'PC{pc_y}',
        color="Linea",
        symbol="Trt",
        hover_data=df_plot,
        title="PCA",
        labels={
            f'PC{pc_x}': f"PC{pc_x} ({(var_exp[pc_x-1]*100).round(2)}%)",
            f'PC{pc_y}': f"PC{pc_y} ({(var_exp[pc_y-1]*100).round(2)}%)"
        }
    )
    
    # Calcular los rangos para cada eje independientemente
    rango_x = abs(df_plot[f'PC{pc_x}']).max()
    rango_y = abs(df_plot[f'PC{pc_y}']).max()
    
    # Añadir un margen del 10% para evitar que los puntos toquen los bordes
    margin_x = rango_x * 0.1
    margin_y = rango_y * 0.1
    
    # Actualizar los ejes con rangos específicos para cada uno
    fig.update_xaxes(
        range=[-rango_x-margin_x, rango_x+margin_x],
        zeroline=True,
        zerolinecolor="gray",
        showgrid=True,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        range=[-rango_y-margin_y, rango_y+margin_y],
        zeroline=True,
        zerolinecolor="gray",
        showgrid=True,
        gridcolor='lightgray',
        scaleanchor="x",
        scaleratio=1  # Mantener proporción 1:1
    )
    
    # Ajustar marcadores y diseño
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',  # Fondo blanco
        paper_bgcolor='white',  # Fondo del papel blanco
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig)
    
    # Mostrar correlaciones con genes
    st.subheader(f"Correlaciones con PC{pc_corr}")
    st.write(f"""
    Esta tabla muestra los genes más correlacionados con PC{pc_corr}. 
    Los valores positivos y negativos indican la dirección de la correlación con esta componente principal.
    """)
    
    # Añadir enlaces a GeneCards
    gene_correlations['enlace'] = [
        f'<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}" target="_blank">{gene}</a>'
        for gene in gene_correlations.index
    ]
    
    # Mostrar top 20 genes positivos y negativos
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Top 20 genes con correlación positiva con PC{pc_corr}:")
        st.write(gene_correlations.head(20).style.format({
            'correlacion': '{:.3f}',
            't': '{:.3f}',
            'p_valor': '{:.2e}',
            'p_ajustado': '{:.2e}'
        }).to_html(escape=False), unsafe_allow_html=True)
    
    with col2:
        st.write(f"Top 20 genes con correlación negativa con PC{pc_corr}:")
        neg_correlations = gene_correlations[gene_correlations['correlacion'] < 0].sort_values('correlacion', ascending=True)
        st.write(neg_correlations.head(20).style.format({
            'correlacion': '{:.3f}',
            't': '{:.3f}',
            'p_valor': '{:.2e}',
            'p_ajustado': '{:.2e}'
        }).to_html(escape=False), unsafe_allow_html=True)

def show_differential_expression(expr_data, covariables):
    """
    Muestra el análisis de expresión diferencial.
    """
    st.subheader("Análisis de Expresión Diferencial")
    
    # Selector de comparación
    st.write("### Seleccionar Comparación")
    col1, col2 = st.columns(2)
    
    with col1:
        base_group = st.selectbox(
            "Grupo Base",
            covariables['Linea'].unique()
        )
    
    with col2:
        comp_group = st.selectbox(
            "Grupo Comparación",
            [x for x in covariables['Linea'].unique() if x != base_group]
        )
    
    # Realizar análisis
    if st.button("Realizar Análisis"):
        # Filtrar datos por grupos
        mask_base = covariables['Linea'] == base_group
        mask_comp = covariables['Linea'] == comp_group
        
        # Calcular expresión diferencial
        results = calculate_differential_expression(
            expr_data,
            mask_base,
            mask_comp
        )
        
        # Mostrar volcano plot
        fig = px.scatter(
            results,
            x='log2FoldChange',
            y='-log10(padj)',
            color='significant',
            hover_data=['gene_symbol'],
            title=f"Volcano Plot: {base_group} vs {comp_group}"
        )
        
        fig.update_layout(
            xaxis_title="log2 Fold Change",
            yaxis_title="-log10(p-value adj)"
        )
        
        st.plotly_chart(fig)
        
        # Mostrar tabla de resultados
        st.write("### Genes Diferencialmente Expresados")
        st.dataframe(results)

def show_clustered_heatmap(expr_data, covariables):
    """
    Muestra el heatmap clusterizado de expresión génica.
    """
    st.subheader("Heatmap Clusterizado de Expresión Génica")
    
    # Crear heatmap clusterizado
    expr_clustered = create_clustered_heatmap(expr_data)
    
    # Crear figura
    fig = go.Figure(data=go.Heatmap(
        z=expr_clustered.values,
        x=expr_clustered.columns,
        y=expr_clustered.index,
        colorscale='RdBu_r'
    ))
    
    fig.update_layout(
        title="Heatmap Clusterizado de Expresión Génica",
        xaxis_title="Muestras",
        yaxis_title="Genes",
        height=800
    )
    
    st.plotly_chart(fig, key="analysis_heatmap")