import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests
import streamlit as st

def perform_pca_analysis(expr_data, covariables, norm_method="Z-score"):
    """
    Realiza el análisis de PCA siguiendo el notebook original.
    
    Args:
        expr_data: DataFrame con los datos de expresión
        covariables: DataFrame con las covariables
        norm_method: str, método de normalización ('Ninguna', 'Z-score', o 'Min-Max')
    """
    # Preparar los datos base
    X = expr_data.T.values
    
    if norm_method == "Z-score":
        # Normalización Z-score por gen (columnas)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = (X - means) / stds
        
    elif norm_method == "Min-Max":
        # Normalización Min-Max por gen (columnas)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X = (X - min_vals) / (max_vals - min_vals)
        
    else:  # "Ninguna"
        # No hacer nada con los datos originales
        pass
    
    # Centrar los datos solo para PCA
    X_centered = X - np.mean(X, axis=0)
    
    # Realizar PCA sobre los datos centrados
    pca = PCA(svd_solver='full')
    scores = pca.fit_transform(X_centered)  # Usar los datos centrados para PCA
    var_exp_p = pca.explained_variance_ratio_
    
    # Crear DataFrame básico para PCA
    df_pca = pd.DataFrame({
        'Linea': covariables["Linea"],
        'Trt': covariables["Tratamiento"],
        'TrtLinea': covariables["Tratamiento"] + " - " + covariables["Linea"],
        'Dia': covariables["Dia"]
    })
    
    return {
        'pca_df': df_pca,
        'var_exp': var_exp_p,
        'scores': scores
    }

def perform_tsne_analysis(expr_data, covariables, norm_method="Z-score", perplexity=30, n_iter=1000, n_components=3):
    """
    Realiza el análisis de t-SNE.
    
    Args:
        expr_data: DataFrame con los datos de expresión
        covariables: DataFrame con las covariables
        norm_method: str, método de normalización ('Ninguna', 'Z-score', o 'Min-Max')
        perplexity: int, parámetro de perplejidad para t-SNE (default: 30)
        n_iter: int, número de iteraciones (default: 1000)
        n_components: int, número de componentes t-SNE a calcular (default: 3)
    """
    # Preparar los datos base
    X = expr_data.T.values
    
    if norm_method == "Z-score":
        # Normalización Z-score por gen (columnas)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = (X - means) / stds
        
    elif norm_method == "Min-Max":
        # Normalización Min-Max por gen (columnas)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X = (X - min_vals) / (max_vals - min_vals)
        
    else:  # "Ninguna"
        # No hacer nada con los datos originales
        pass
    
    # Aplicar t-SNE (usar max_iter en lugar de n_iter)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(X)
    
    # Crear DataFrame básico para t-SNE con todas las componentes
    df_tsne = pd.DataFrame({
        'Linea': covariables["Linea"],
        'Trt': covariables["Tratamiento"],
        'TrtLinea': covariables["Tratamiento"] + " - " + covariables["Linea"],
        'Dia': covariables["Dia"]
    })
    
    # Agregar todas las componentes t-SNE como columnas adicionales
    for i in range(n_components):
        df_tsne[f't-SNE{i+1}'] = tsne_results[:, i]
    
    return {
        'tsne_df': df_tsne,
        'tsne_results': tsne_results,
        'n_components': n_components
    }

def calculate_gene_correlations(expr_data, pc_scores):
    """
    Calcula las correlaciones entre los genes y una componente principal.
    
    Args:
        expr_data: DataFrame con los datos de expresión
        pc_scores: Array con los scores de la componente principal
    """
    # Calcular correlaciones con la PC
    correlaciones = []
    for gen in expr_data.index:
        corr, p_val = stats.spearmanr(expr_data.loc[gen], pc_scores)
        correlaciones.append({
            'gen': gen,
            'correlacion': corr,
            'p_valor': p_val
        })
    
    # Crear DataFrame con resultados
    r_PC = pd.DataFrame(correlaciones)
    r_PC.set_index('gen', inplace=True)
    
    # Ajustar p valores por FDR
    _, p_adj, _, _ = multipletests(r_PC['p_valor'], method='fdr_bh')
    r_PC['p_ajustado'] = p_adj
    
    # Calcular estadístico t
    n_muestras = len(expr_data.columns)
    r_PC['t'] = r_PC['correlacion'] * np.sqrt((n_muestras - 2) / (1 - r_PC['correlacion']**2))
    
    # Añadir enlaces a GeneCards
    r_PC['enlace'] = [
        f'<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}" target="_blank">{gene}</a>'
        for gene in r_PC.index
    ]
    
    # Ordenar por correlación
    return r_PC.sort_values('correlacion', ascending=False)

def create_clustered_heatmap(expr_data):
    """
    Crea un heatmap clusterizado de la expresión génica.
    """
    # Calcular el linkage para filas y columnas
    row_linkage = linkage(expr_data, method='ward', metric='euclidean')
    col_linkage = linkage(expr_data.T, method='ward', metric='euclidean')
    
    # Obtener el orden de las filas y columnas
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    
    # Reordenar la matriz
    expr_clustered = expr_data.iloc[row_order, col_order]
    
    return expr_clustered

def calculate_differential_expression(expr_data, mask_base, mask_comp):
    """
    Calcula la expresión diferencial entre dos grupos con ajuste FDR.
    """
    # Calcular medias por grupo
    mean_base = expr_data.loc[:, mask_base].mean(axis=1)
    mean_comp = expr_data.loc[:, mask_comp].mean(axis=1)
    
    # Calcular fold change
    log2fc = np.log2(mean_comp / mean_base)
    
    # Calcular p-valores (t-test)
    pvalues = []
    for gene in expr_data.index:
        t, p = stats.ttest_ind(
            expr_data.loc[gene, mask_base],
            expr_data.loc[gene, mask_comp]
        )
        pvalues.append(p)
    
    # Ajustar p-valores por FDR
    _, p_adj, _, _ = multipletests(pvalues, method='fdr_bh')
    
    # Crear DataFrame de resultados
    results = pd.DataFrame({
        'gene_symbol': expr_data.index,
        'log2FoldChange': log2fc,
        'pvalue': pvalues,
        'padj': p_adj
    })
    
    # Marcar genes significativos (|log2FC| > 1 y padj < 0.05)
    results['significant'] = (results['padj'] < 0.05) & (abs(results['log2FoldChange']) > 1)
    
    return results.sort_values('padj')