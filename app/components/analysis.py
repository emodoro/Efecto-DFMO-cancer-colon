import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage
from utils.data_loader import filter_data, normalize_data
from utils.analysis_utils import (
    perform_pca_analysis,
    perform_tsne_analysis,
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
        ["PCA", "t-SNE", "Heatmap Clusterizado"]
    )
    
    if analysis_type == "PCA":
        show_pca_analysis(expr_filtered, cov_filtered, filters)
    elif analysis_type == "t-SNE":
        show_tsne_analysis(expr_filtered, cov_filtered, filters)
    else:
        show_clustered_heatmap(expr_filtered, cov_filtered, filters)

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
        },
        color_discrete_sequence=px.colors.qualitative.Set1  # Usar colores específicos y claros
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
    
    # Botón para descargar la figura
    import plotly.io as pio
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        # Descargar como HTML interactivo
        html_buffer = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar gráfico PCA (HTML)",
            data=html_buffer,
            file_name=f"pca_PC{pc_x}_vs_PC{pc_y}.html",
            mime="text/html"
        )
    with col_download2:
        # Descargar como imagen PNG
        try:
            # Asegurar que la figura tenga todos los colores y leyendas visibles
            fig_export = fig
            img_bytes = pio.to_image(fig_export, format='png', width=1400, height=1400, scale=2)
            st.download_button(
                label="📥 Descargar gráfico PCA (PNG)",
                data=img_bytes,
                file_name=f"pca_PC{pc_x}_vs_PC{pc_y}.png",
                mime="image/png"
            )
        except Exception as e:
            st.warning("⚠️ Exportación PNG no disponible. Instala kaleido: pip install kaleido")
    
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
    
    # Botón para descargar tabla completa de correlaciones
    st.write("---")
    st.write("### Descargar tabla completa de correlaciones")
    
    # Preparar tabla para descarga (todos los genes)
    gene_correlations_download = gene_correlations.copy()
    gene_correlations_download = gene_correlations_download.drop(columns=['enlace'], errors='ignore')
    gene_correlations_download.index.name = 'gene_symbol'
    gene_correlations_download = gene_correlations_download.reset_index()
    
    csv_buffer = gene_correlations_download.to_csv(index=False)
    st.download_button(
        label=f"📥 Descargar todas las correlaciones PC{pc_corr} (CSV)",
        data=csv_buffer,
        file_name=f"correlaciones_PC{pc_corr}_completo.csv",
        mime="text/csv"
    )

def show_tsne_analysis(expr_data, covariables, filters):
    """
    Muestra el análisis de t-SNE.
    """
    st.subheader("Análisis t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    
    # Convertir la normalización de la barra lateral al formato del t-SNE
    norm_method = {
        "Sin normalizar": "Ninguna",
        "Z-score": "Z-score",
        "Min-Max": "Min-Max"
    }[filters['normalization']]
    
    # Parámetros de t-SNE
    n_samples = len(expr_data.columns)
    max_perplexity = min(50, n_samples - 1)  # La perplejidad debe ser menor que n_samples
    
    col1, col2 = st.columns(2)
    with col1:
        perplexity = st.slider("Perplejidad", min_value=2, max_value=max_perplexity, 
                               value=min(5, max_perplexity), step=1,
                               help=f"Equilibra aspectos locales vs globales. Debe ser menor que el número de muestras ({n_samples})")
    with col2:
        n_iter = st.slider("Iteraciones", min_value=250, max_value=2000, value=1000, step=250,
                          help="Número de iteraciones para la optimización")
    
    # Realizar t-SNE
    with st.spinner("Calculando t-SNE..."):
        # Calcular 3 componentes (máximo permitido por el algoritmo barnes_hut)
        results = perform_tsne_analysis(expr_data, covariables, norm_method=norm_method, 
                                       perplexity=perplexity, n_iter=n_iter, n_components=3)
        tsne_results = results['tsne_results']
        df_tsne = results['tsne_df']
        n_components = results['n_components']
    
    # Selectores de componentes para cada eje
    st.subheader("Seleccionar componentes t-SNE")
    col1, col2 = st.columns(2)
    with col1:
        comp_x = st.selectbox(
            "Componente eje X",
            options=list(range(1, n_components + 1)),
            index=0,
            format_func=lambda x: f"t-SNE {x}"
        )
    with col2:
        comp_y = st.selectbox(
            "Componente eje Y",
            options=list(range(1, n_components + 1)),
            index=1,
            format_func=lambda x: f"t-SNE {x}"
        )
    
    # Crear DataFrame para plotly
    df_plot = df_tsne.copy()
    df_plot['tSNE_X'] = tsne_results[:, comp_x - 1]
    df_plot['tSNE_Y'] = tsne_results[:, comp_y - 1]
    
    # Crear gráfico de t-SNE
    st.subheader("Visualización t-SNE")
    
    fig = px.scatter(
        df_plot,
        x='tSNE_X',
        y='tSNE_Y',
        color="Linea",
        symbol="Trt",
        hover_data=df_plot,
        title=f"t-SNE: Componente {comp_x} vs Componente {comp_y}",
        labels={
            'tSNE_X': f"t-SNE {comp_x}",
            'tSNE_Y': f"t-SNE {comp_y}"
        },
        color_discrete_sequence=px.colors.qualitative.Set1  # Usar colores específicos y claros
    )
    
    # Calcular los rangos
    rango_x = abs(df_plot['tSNE_X']).max()
    rango_y = abs(df_plot['tSNE_Y']).max()
    
    # Añadir un margen del 10%
    margin_x = rango_x * 0.1
    margin_y = rango_y * 0.1
    
    # Actualizar los ejes
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
        scaleratio=1
    )
    
    # Ajustar marcadores y diseño
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Botón para descargar la figura t-SNE
    import plotly.io as pio
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        # Descargar como HTML interactivo
        html_buffer = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar gráfico t-SNE (HTML)",
            data=html_buffer,
            file_name=f"tsne_comp{comp_x}_vs_comp{comp_y}.html",
            mime="text/html"
        )
    with col_download2:
        # Descargar como imagen PNG
        try:
            img_bytes = pio.to_image(fig, format='png', width=1200, height=1200, scale=2)
            st.download_button(
                label="📥 Descargar gráfico t-SNE (PNG)",
                data=img_bytes,
                file_name=f"tsne_comp{comp_x}_vs_comp{comp_y}.png",
                mime="image/png"
            )
        except Exception as e:
            st.warning("⚠️ Exportación PNG no disponible. Instala kaleido: pip install kaleido")
    
    # Análisis de correlaciones con los ejes t-SNE
    st.subheader("Correlación de Genes con los Ejes t-SNE")
    
    # Selector de eje para correlación
    tsne_axis = st.selectbox(
        "Seleccionar eje t-SNE para análisis de correlación",
        options=list(range(1, n_components + 1)),
        format_func=lambda x: f"t-SNE {x}"
    )
    
    # Calcular correlaciones
    tsne_scores = tsne_results[:, tsne_axis - 1]
    gene_correlations = calculate_gene_correlations(expr_data, tsne_scores)
    
    # Mostrar tablas de correlación
    st.write("### Genes correlacionados con el eje t-SNE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Top 20 genes con correlación positiva con t-SNE {tsne_axis}:")
        st.write(gene_correlations.head(20).style.format({
            'correlacion': '{:.3f}',
            't': '{:.3f}',
            'p_valor': '{:.2e}',
            'p_ajustado': '{:.2e}'
        }).to_html(escape=False), unsafe_allow_html=True)
    
    with col2:
        st.write(f"Top 20 genes con correlación negativa con t-SNE {tsne_axis}:")
        neg_correlations = gene_correlations[gene_correlations['correlacion'] < 0].sort_values('correlacion', ascending=True)
        st.write(neg_correlations.head(20).style.format({
            'correlacion': '{:.3f}',
            't': '{:.3f}',
            'p_valor': '{:.2e}',
            'p_ajustado': '{:.2e}'
        }).to_html(escape=False), unsafe_allow_html=True)
    
    # Botón para descargar tabla completa de correlaciones
    st.write("---")
    st.write("### Descargar tabla completa de correlaciones")
    
    # Preparar tabla para descarga (todos los genes)
    gene_correlations_download = gene_correlations.copy()
    gene_correlations_download.index.name = 'gene_symbol'
    gene_correlations_download = gene_correlations_download.reset_index()
    
    csv_buffer = gene_correlations_download.to_csv(index=False)
    st.download_button(
        label=f"📥 Descargar todas las correlaciones t-SNE{tsne_axis} (CSV)",
        data=csv_buffer,
        file_name=f"correlaciones_tSNE{tsne_axis}_completo.csv",
        mime="text/csv"
    )

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

def show_clustered_heatmap(expr_data, covariables, filters):
    """
    Muestra el heatmap clusterizado de expresión génica con dendrogramas y anotaciones de covariables.
    """
    st.subheader("Heatmap Clusterizado de Expresión Génica")
    
    # Obtener parámetros del sidebar
    clustering_method = filters.get('clustering_method', 'Ward').lower()
    n_top_genes = filters.get('n_top_genes', 50)
    
    # Ajustar n_top_genes si es mayor que el total de genes disponibles
    n_genes_available = len(expr_data)
    n_genes_to_show = min(n_top_genes, n_genes_available)
    
    # Seleccionar los top genes por varianza
    gene_variance = expr_data.var(axis=1)
    top_genes = gene_variance.nlargest(n_genes_to_show).index
    expr_top = expr_data.loc[top_genes]
    
    st.info(f"Mostrando {n_genes_to_show} genes (de {n_genes_available} disponibles) con mayor varianza usando clustering {clustering_method.capitalize()}")
    
    # Normalizar datos (Z-score por genes)
    expr_norm = (expr_top.T - expr_top.mean(axis=1)) / expr_top.std(axis=1)
    expr_norm = expr_norm.T
    
    # Calcular linkage para filas (genes) y columnas (muestras)
    row_linkage = linkage(expr_norm.values, method=clustering_method, metric='euclidean')
    col_linkage = linkage(expr_norm.T.values, method=clustering_method, metric='euclidean')
    
    # Crear figura con dendrogramas usando figure_factory
    fig = ff.create_dendrogram(
        expr_norm.T.values,
        orientation='bottom',
        labels=expr_norm.columns.tolist(),
        linkagefun=lambda x: linkage(x.T, method=clustering_method, metric='euclidean')
    )
    
    # Obtener el orden de las columnas del dendrograma
    col_order = fig['layout']['xaxis']['ticktext']
    
    # Crear dendrograma para las filas (genes)
    dendro_side = ff.create_dendrogram(
        expr_norm.values,
        orientation='right',
        labels=expr_norm.index.tolist(),
        linkagefun=lambda x: linkage(x, method=clustering_method, metric='euclidean')
    )
    
    # Obtener el orden de las filas del dendrograma
    row_order = dendro_side['layout']['yaxis']['ticktext']
    
    # Reordenar la matriz según los dendrogramas
    expr_reordered = expr_norm.loc[row_order, col_order]
    
    # Reordenar las covariables según el orden de las columnas
    cov_reordered = covariables.loc[col_order]
    
    # Definir paletas de colores para cada covariable
    color_maps = {
        'Tratamiento': {'Control': '#66c2a5', 'DFMO': '#fc8d62'},
        'Linea': {'HT29': '#8da0cb', 'NCM460': '#e78ac3'},
        'Dia': {}
    }
    
    # Crear colores para los días únicos
    dias_unicos = sorted(covariables['Dia'].unique())
    dia_colors = ['#a6d854', '#ffd92f', '#e5c494']
    for i, dia in enumerate(dias_unicos):
        color_maps['Dia'][dia] = dia_colors[i % len(dia_colors)]
    
    # Crear los subplots con espacio para las anotaciones de covariables
    fig = make_subplots(
        rows=5, cols=2,
        row_heights=[0.15, 0.02, 0.02, 0.02, 0.79],
        column_widths=[0.85, 0.15],
        specs=[[{'type': 'scatter'}, None],
               [{'type': 'heatmap'}, None],
               [{'type': 'heatmap'}, None],
               [{'type': 'heatmap'}, None],
               [{'type': 'heatmap'}, {'type': 'scatter'}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.005
    )
    
    # Dendrograma superior (muestras)
    dendro_top = ff.create_dendrogram(
        expr_norm.T.values,
        orientation='bottom',
        linkagefun=lambda x: linkage(x.T, method=clustering_method, metric='euclidean')
    )
    for trace in dendro_top['data']:
        fig.add_trace(trace, row=1, col=1)
    
    # Añadir anotaciones de covariables (3 líneas)
    # Fila 2: Tratamiento
    trt_values = [cov_reordered.loc[sample, 'Tratamiento'] for sample in col_order]
    trt_colors_numeric = [0 if val == 'Control' else 1 for val in trt_values]
    fig.add_trace(go.Heatmap(
        z=[trt_colors_numeric],
        x=col_order,
        y=['Tratamiento'],
        colorscale=[[0, '#66c2a5'], [1, '#fc8d62']],
        showscale=False,
        hovertemplate='Muestra: %{x}<br>Tratamiento: %{customdata}<extra></extra>',
        customdata=[trt_values]
    ), row=2, col=1)
    
    # Fila 3: Línea
    linea_values = [cov_reordered.loc[sample, 'Linea'] for sample in col_order]
    linea_colors_numeric = [0 if val == 'HT29' else 1 for val in linea_values]
    fig.add_trace(go.Heatmap(
        z=[linea_colors_numeric],
        x=col_order,
        y=['Línea'],
        colorscale=[[0, '#8da0cb'], [1, '#e78ac3']],
        showscale=False,
        hovertemplate='Muestra: %{x}<br>Línea: %{customdata}<extra></extra>',
        customdata=[linea_values]
    ), row=3, col=1)
    
    # Fila 4: Día
    dia_values = [cov_reordered.loc[sample, 'Dia'] for sample in col_order]
    # Crear mapeo numérico para los días
    dias_unicos_sorted = sorted(set(dia_values))
    dia_to_num = {dia: i for i, dia in enumerate(dias_unicos_sorted)}
    dia_colors_numeric = [dia_to_num[val] for val in dia_values]
    
    # Crear escala de colores normalizada para los días
    n_dias = len(dias_unicos_sorted)
    if n_dias == 1:
        dia_colorscale = [[0, '#a6d854'], [1, '#a6d854']]
    elif n_dias == 2:
        dia_colorscale = [[0, '#a6d854'], [1, '#ffd92f']]
    else:
        dia_colorscale = [[0, '#a6d854'], [0.5, '#ffd92f'], [1, '#e5c494']]
    
    fig.add_trace(go.Heatmap(
        z=[dia_colors_numeric],
        x=col_order,
        y=['Día'],
        colorscale=dia_colorscale,
        showscale=False,
        hovertemplate='Muestra: %{x}<br>Día: %{customdata}<extra></extra>',
        customdata=[dia_values]
    ), row=4, col=1)
    
    # Dendrograma lateral (genes)
    dendro_side = ff.create_dendrogram(
        expr_norm.values,
        orientation='left',
        linkagefun=lambda x: linkage(x, method=clustering_method, metric='euclidean')
    )
    for trace in dendro_side['data']:
        fig.add_trace(trace, row=5, col=2)
    
    # Heatmap principal
    heatmap = go.Heatmap(
        z=expr_reordered.values,
        x=col_order,
        y=row_order,
        colorscale='RdBu_r',
        colorbar=dict(
            title="Z-score",
            x=1.15,
            len=0.79,
            y=0.395
        )
    )
    fig.add_trace(heatmap, row=5, col=1)
    
    # Crear leyendas manualmente como anotaciones
    legend_items = []
    y_pos = 0.98
    
    # Leyenda Tratamiento
    legend_items.append(dict(text="<b>Tratamiento:</b>", x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=10)))
    y_pos -= 0.02
    for key, color in color_maps['Tratamiento'].items():
        legend_items.append(dict(
            text=f'<span style="color:{color}">■</span> {key}',
            x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=9)
        ))
        y_pos -= 0.015
    
    # Leyenda Línea
    y_pos -= 0.01
    legend_items.append(dict(text="<b>Línea:</b>", x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=10)))
    y_pos -= 0.02
    for key, color in color_maps['Linea'].items():
        legend_items.append(dict(
            text=f'<span style="color:{color}">■</span> {key}',
            x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=9)
        ))
        y_pos -= 0.015
    
    # Leyenda Día
    y_pos -= 0.01
    legend_items.append(dict(text="<b>Día:</b>", x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=10)))
    y_pos -= 0.02
    for key, color in color_maps['Dia'].items():
        legend_items.append(dict(
            text=f'<span style="color:{color}">■</span> {key}',
            x=1.02, y=y_pos, showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=9)
        ))
        y_pos -= 0.015
    
    # Actualizar layout
    fig.update_layout(
        title="Heatmap Clusterizado de Expresión Génica",
        showlegend=False,
        height=950,
        width=1100,
        annotations=legend_items
    )
    
    # Ocultar ejes de los dendrogramas
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=5, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=5, col=2)
    
    # Ocultar ejes de las anotaciones de covariables
    for row in [2, 3, 4]:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=1)
        fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=False, tickfont=dict(size=9), row=row, col=1)
    
    # Mostrar etiquetas solo en el heatmap
    fig.update_xaxes(title="Muestras", side='bottom', row=5, col=1)
    fig.update_yaxes(title="Genes", side='left', row=5, col=1)
    
    st.plotly_chart(fig, key="analysis_heatmap")
    
    # Botones para descargar el heatmap
    import plotly.io as pio
    col_download1, col_download2, col_download3 = st.columns(3)
    with col_download1:
        # Descargar como HTML interactivo
        html_buffer = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar heatmap (HTML)",
            data=html_buffer,
            file_name=f"heatmap_clusterizado_{clustering_method}_{n_genes_to_show}genes.html",
            mime="text/html"
        )
    with col_download2:
        # Descargar como imagen PNG
        try:
            img_bytes = pio.to_image(fig, format='png', width=1400, height=1200, scale=2)
            st.download_button(
                label="📥 Descargar heatmap (PNG)",
                data=img_bytes,
                file_name=f"heatmap_clusterizado_{clustering_method}_{n_genes_to_show}genes.png",
                mime="image/png"
            )
        except Exception as e:
            st.warning("⚠️ Exportación PNG no disponible. Instala kaleido: pip install kaleido")
    with col_download3:
        # Descargar datos del heatmap como CSV
        csv_buffer = expr_reordered.to_csv()
        st.download_button(
            label="📥 Descargar datos (CSV)",
            data=csv_buffer,
            file_name=f"datos_heatmap_{n_genes_to_show}genes.csv",
            mime="text/csv"
        )