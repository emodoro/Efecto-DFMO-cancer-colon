import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Carga y preprocesa los datos desde un archivo CSV.
    
    Args:
        file_path (Path): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame procesado
    """
    try:
        # Cargar datos
        data = pd.read_csv(file_path, index_col=0)
        
        # Verificar si hay valores faltantes
        if data.isnull().any().any():
            print(f"Advertencia: Se encontraron {data.isnull().sum().sum()} valores faltantes")
        
        return data
        
    except Exception as e:
        raise Exception(f"Error al cargar {file_path}: {str(e)}")

def normalize_data(data, method='zscore'):
    """
    Normaliza los datos exactamente como en el notebook EDA.
    
    Args:
        data (pd.DataFrame): Datos a normalizar
        method (str): Método de normalización ('zscore' o 'minmax')
        
    Returns:
        pd.DataFrame: Datos normalizados
    """
    if method == 'zscore':
        # Escalar por filas (genes) como en el notebook
        normalized = (data.T - data.mean(axis=1)) / data.std(axis=1)
        return normalized.T
    elif method == 'minmax':
        normalized = (data - data.min()) / (data.max() - data.min())
        return normalized
    else:
        return data

def non_specific_gene_filter(expr_data):
    """
    Realiza un filtrado no específico de genes basado en mediana e IQR.
    
    Args:
        expr_data (pd.DataFrame): Matriz de expresión
        
    Returns:
        pd.DataFrame: Matriz de expresión filtrada
    """
    # Calcular medianas de expresión de cada gen
    med_exp = np.median(expr_data, axis=1)
    q1_med_exp = np.quantile(med_exp, q=0.25)
    
    # Calcular IQR de expresión de cada gen
    q3_exp = np.quantile(expr_data, q=0.75, axis=1)
    q1_exp = np.quantile(expr_data, q=0.25, axis=1)
    iqr_exp = q3_exp - q1_exp
    q1_iqr_exp = np.quantile(iqr_exp, q=0.25)
    
    # Aplicar filtro
    filtro = (iqr_exp >= q1_iqr_exp) & (med_exp >= q1_med_exp)
    return expr_data.loc[filtro, :]

def filter_data(expr_data, covariables, filters):
    """
    Filtra los datos según los criterios especificados.
    
    Args:
        expr_data (pd.DataFrame): Matriz de expresión
        covariables (pd.DataFrame): Matriz de covariables
        filters (dict): Diccionario con los filtros a aplicar
        
    Returns:
        tuple: (expr_data_filtered, covariables_filtered)
    """
    # Aplicar filtros de línea celular
    mask_cell = covariables['Linea'].isin(filters['cell_lines'])
    
    # Aplicar filtros de tratamiento
    mask_treat = covariables['Tratamiento'].isin(filters['treatments'])
    
    # Combinar máscaras
    mask_final = mask_cell & mask_treat
    
    # Filtrar datos
    covariables_filtered = covariables[mask_final]
    expr_data_filtered = expr_data[covariables_filtered.index]
    
    # Aplicar filtrado no específico de genes
    expr_data_filtered = non_specific_gene_filter(expr_data_filtered)
    
    return expr_data_filtered, covariables_filtered