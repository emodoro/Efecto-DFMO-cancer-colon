import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_summary_stats(expr_data, covariables):
    """
    Crea un diccionario con estadísticas resumidas del dataset.
    """
    stats = {
        "Número total de genes": len(expr_data.index),
        "Número total de muestras": len(expr_data.columns),
        "Líneas celulares": len(covariables['Linea'].unique()),
        "Condiciones": len(covariables['Tratamiento'].unique())
    }
    return pd.DataFrame(list(stats.items()), columns=['Métrica', 'Valor'])

def create_heatmap(data, title="Heatmap de Expresión Génica"):
    """
    Crea un heatmap interactivo usando plotly.
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu_r'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Muestras",
        yaxis_title="Genes",
        height=800
    )
    
    return fig

def create_pca_plot(pca_results, metadata, color_by="Tratamiento"):
    """
    Crea un gráfico de PCA usando plotly.
    """
    fig = px.scatter(
        pca_results,
        x='PC1',
        y='PC2',
        color=metadata[color_by],
        symbol=metadata['Linea'],
        title='Análisis de Componentes Principales (PCA)',
        labels={
            'PC1': f'PC1 ({pca_results["var_explained"][0]:.1f}% varianza)',
            'PC2': f'PC2 ({pca_results["var_explained"][1]:.1f}% varianza)'
        }
    )
    return fig

def create_volcano_plot(results_df, title="Volcano Plot"):
    """
    Crea un volcano plot para visualizar genes diferencialmente expresados.
    """
    fig = px.scatter(
        results_df,
        x='log2FoldChange',
        y='-log10(pvalue)',
        color='significant',
        hover_data=['gene_symbol'],
        title=title
    )
    
    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)"
    )
    
    return fig

def create_boxplot(data, x, y, color, title):
    """
    Crea un boxplot para comparar distribuciones.
    """
    fig = px.box(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        points="all"
    )
    return fig