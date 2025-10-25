import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from utils.data_loader import filter_data, normalize_data
from utils.plotting import create_heatmap, create_boxplot
from utils.analysis_utils import create_clustered_heatmap

def show_data_explorer(expr_data, covariables, filters):
    """
    Muestra el explorador interactivo de datos.
    
    Args:
        expr_data (pd.DataFrame): Matriz de expresión génica
        covariables (pd.DataFrame): Matriz de covariables
        filters (dict): Filtros seleccionados en el sidebar
    """
    st.header("🔍 Explorador de Datos")
    
    # Filtrar datos según selección
    expr_filtered, cov_filtered = filter_data(expr_data, covariables, filters)
    
    # Aplicar normalización desde la barra lateral
    if filters['normalization'] != "Sin normalizar":
        method = 'zscore' if filters['normalization'] == "Z-score" else 'minmax'
        expr_filtered = normalize_data(expr_filtered, method)
    
    # Mostrar dimensiones de los datos
    st.subheader("📊 Dimensiones de los Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Número de Genes", len(expr_filtered.index))
    with col2:
        st.metric("Número de Muestras", len(expr_filtered.columns))
    
    # Selector de visualización
    view_type = st.radio(
        "Tipo de Visualización",
        ["Vista General", "Genes Específicos", "Distribuciones"]
    )
    
    if view_type == "Vista General":
        show_general_view(expr_filtered, cov_filtered)
    elif view_type == "Genes Específicos":
        show_specific_genes(expr_filtered, cov_filtered)
    else:
        show_distributions(expr_filtered, cov_filtered)

def show_general_view(expr_data, covariables):
    """
    Muestra una vista general de los datos usando un heatmap clusterizado.
    """
    st.subheader("Vista General de la Expresión Génica")
    
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
    
    st.plotly_chart(fig, use_container_width=True, key="explorer_heatmap")

def show_specific_genes(expr_data, covariables):
    """
    Permite explorar genes específicos.
    """
    # Selector de genes
    selected_genes = st.multiselect(
        "Seleccionar Genes",
        options=expr_data.index.tolist(),
        max_selections=5
    )
    
    if selected_genes:
        # Crear visualización para cada gen seleccionado
        for gene in selected_genes:
            st.subheader(f"Gen: {gene}")
            
            # Preparar datos para el boxplot
            gene_data = expr_data.loc[gene]
            plot_data = gene_data.reset_index()
            plot_data.columns = ['muestra', 'expresion']
            plot_data = plot_data.merge(covariables, left_on='muestra', right_index=True)
            
            # Crear boxplot
            fig = create_boxplot(
                plot_data,
                x='Linea',
                y='expresion',
                color='Tratamiento',
                title=f"Expresión de {gene} por Línea Celular y Tratamiento"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_distributions(expr_data, covariables):
    """
    Muestra distribuciones generales de la expresión génica.
    """
    st.subheader("Distribuciones de Expresión Génica")
    
    # Definir el layout de los subplots
    n_cols = 3  # Número de columnas
    n_rows = int(np.ceil(len(expr_data.columns) / n_cols))  # Número de filas necesarias
    
    # Crear figura con subplots
    fig = go.Figure()
    
    # Crear los subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=expr_data.columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Para cada muestra
    for idx, columna in enumerate(expr_data.columns):
        # Calcular posición en la matriz de subplots
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Obtener datos
        datos = expr_data[columna]
        
        # Calcular estadísticas
        media = round(datos.mean(), 3)
        mediana = round(datos.median(), 3)
        desv = round(datos.std(), 3)
        mad = round((datos - mediana).abs().median(), 3)
        curtosis = round(datos.kurtosis(), 3)
        asimetria = round(datos.skew(), 3)
        
        # Calcular densidad kernel
        kernel = stats.gaussian_kde(datos)
        x_range = np.linspace(datos.min(), datos.max(), 200)
        
        # Añadir histograma
        fig.add_trace(
            go.Histogram(
                x=datos,
                histnorm='density',
                name=columna,
                opacity=0.5,
                showlegend=False,
                hovertemplate="Valor: %{x}<br>Densidad: %{y:.3f}<extra></extra>"
            ),
            row=row,
            col=col
        )
        
        # Añadir curva de densidad
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kernel(x_range),
                mode='lines',
                name=columna,
                showlegend=False,
                line=dict(color='red'),
                hovertemplate=f"Media: {media}<br>Mediana: {mediana}<br>Desviación: {desv}<br>MAD: {mad}<br>Asimetría: {asimetria}<br>Curtosis: {curtosis}<extra></extra>"
            ),
            row=row,
            col=col
        )
    
    # Ajustar el diseño
    height = 300 * n_rows  # Altura proporcional al número de filas
    
    fig.update_layout(
        title="Distribución de la Expresión Génica por Muestra",
        showlegend=False,
        height=height,
        width=900,
        template='plotly_white'
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Expresión")
    fig.update_yaxes(title_text="Densidad")
    
    st.plotly_chart(fig)