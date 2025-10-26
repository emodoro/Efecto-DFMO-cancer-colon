import streamlit as st

def create_sidebar(n_total_genes=5000):
    """
    Crea y gestiona el sidebar de la aplicación.
    Args:
        n_total_genes: Número total de genes en el dataset
    Returns:
        dict: Diccionario con los filtros seleccionados
    """
    st.sidebar.title("Filtros y Opciones")
    
    # Sección de filtros
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
    
    # Opciones de visualización
    st.sidebar.header("Opciones de Visualización")
    
    # Tipo de normalización global
    normalization = st.sidebar.selectbox(
        "Normalización",
        ["Sin normalizar", "Z-score", "Min-Max"],
        help="Método de normalización aplicado a los datos"
    )
    
    # Método de clustering
    clustering_method = st.sidebar.selectbox(
        "Método de Clustering",
        ["Ward", "Complete", "Average"]
    )
    
    # Número de genes top
    n_top_genes = st.sidebar.slider(
        "Número de genes",
        min_value=10,
        max_value=n_total_genes,
        value=min(50, n_total_genes),
        step=10,
        help=f"Selecciona cuántos genes mostrar en el heatmap (de 10 a {n_total_genes})"
    )
    
    return {
        "cell_lines": cell_lines,
        "treatments": treatments,
        "normalization": normalization,
        "clustering_method": clustering_method,
        "n_top_genes": n_top_genes
    }