import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage
from utils.data_loader import filter_data, normalize_data
from utils.plotting import create_heatmap, create_boxplot, is_streamlit_cloud
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
        show_general_view(expr_filtered, cov_filtered, filters)
    elif view_type == "Genes Específicos":
        show_specific_genes(expr_filtered, cov_filtered)
    else:
        show_distributions(expr_filtered, cov_filtered)

def show_general_view(expr_data, covariables, filters):
    """
    Muestra una vista general de los datos usando un heatmap clusterizado con dendrogramas y anotaciones de covariables.
    """
    st.subheader("Vista General de la Expresión Génica")
    
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
    
    st.plotly_chart(fig, use_container_width=True, key="explorer_heatmap")
    
    # Botones para descargar el heatmap
    import plotly.io as pio
    col_download1, col_download2, col_download3 = st.columns(3)
    with col_download1:
        # Descargar como HTML interactivo
        html_buffer = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar heatmap (HTML)",
            data=html_buffer,
            file_name=f"heatmap_exploracion_{clustering_method}_{n_genes_to_show}genes.html",
            mime="text/html",
            key="download_heatmap_html_explorer"
        )
    with col_download2:
        # Descargar como imagen PNG
        try:
            img_bytes = pio.to_image(fig, format='png', width=1400, height=1200, scale=2)
            st.download_button(
                label="📥 Descargar heatmap (PNG)",
                data=img_bytes,
                file_name=f"heatmap_exploracion_{clustering_method}_{n_genes_to_show}genes.png",
                mime="image/png",
                key="download_heatmap_png_explorer"
            )
        except Exception as e:
            if not is_streamlit_cloud():
                st.warning("⚠️ Exportación PNG no disponible. Instala kaleido: pip install kaleido")
    with col_download3:
        # Descargar datos del heatmap como CSV
        csv_buffer = expr_reordered.to_csv()
        st.download_button(
            label="📥 Descargar datos (CSV)",
            data=csv_buffer,
            file_name=f"datos_heatmap_exploracion_{n_genes_to_show}genes.csv",
            mime="text/csv",
            key="download_heatmap_csv_explorer"
        )

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
        vertical_spacing=0.15,
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
        media = datos.mean()
        mediana = datos.median()
        sd = datos.std()
        q1 = datos.quantile(0.25)
        q3 = datos.quantile(0.75)
        iqr = q3 - q1
        curtosis = datos.kurtosis()
        asimetria = datos.skew()
        
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
                hovertemplate="Valor: %{x:.2f}<br>Densidad: %{y:.3f}<extra></extra>"
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
                line=dict(color='red', width=2),
                hovertemplate=f"<b>Estadísticas</b><br>" +
                             f"Media: {media:.2f}<br>" +
                             f"Mediana: {mediana:.2f}<br>" +
                             f"SD: {sd:.2f}<br>" +
                             f"IQR: {iqr:.2f}<br>" +
                             f"Asimetría: {asimetria:.2f}<br>" +
                             f"Curtosis: {curtosis:.2f}<extra></extra>"
            ),
            row=row,
            col=col
        )
        
        # Añadir anotación con estadísticas en el subplot
        stats_text = (f"μ={media:.1f}, M={mediana:.1f}<br>" +
                     f"SD={sd:.1f}, IQR={iqr:.1f}<br>" +
                     f"Asim={asimetria:.2f}, Kurt={curtosis:.2f}")
        
        # Determinar referencias de ejes para este subplot
        xref = f"x{idx+1} domain" if idx > 0 else "x domain"
        yref = f"y{idx+1} domain" if idx > 0 else "y domain"
        
        fig.add_annotation(
            text=stats_text,
            xref=xref, 
            yref=yref,
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=9, color='black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=3
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
    
    # Botones para descargar la figura de distribuciones
    import plotly.io as pio
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        # Descargar como HTML interactivo
        html_buffer = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar distribuciones (HTML)",
            data=html_buffer,
            file_name="distribuciones_muestras.html",
            mime="text/html",
            key="download_distributions_html"
        )
    with col_download2:
        # Descargar como imagen PNG
        try:
            img_bytes = pio.to_image(fig, format='png', width=1400, height=height, scale=2)
            st.download_button(
                label="📥 Descargar distribuciones (PNG)",
                data=img_bytes,
                file_name="distribuciones_muestras.png",
                mime="image/png",
                key="download_distributions_png"
            )
        except Exception as e:
            if not is_streamlit_cloud():
                st.warning("⚠️ Exportación PNG no disponible. Instala kaleido: pip install kaleido")