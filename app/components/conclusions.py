import streamlit as st
import pandas as pd
from pathlib import Path
from utils.data_loader import non_specific_gene_filter

def show_conclusions(expr_data, covariables):
    """
    Muestra la sección de conclusiones del análisis, incluyendo un resumen cuantitativo
    del pipeline (genes iniciales, alineados y tras filtrado), hallazgos clave de PCA, t-SNE
    y heatmaps, implicaciones biológicas, próximos pasos y limitaciones.

    Args:
        expr_data (pd.DataFrame): Matriz de expresión agregada por mediana (gene symbol)
        covariables (pd.DataFrame): Tabla de covariables por muestra
    """
    st.header("📑 Conclusiones del Análisis")

    # Resumen cuantitativo del dataset
    st.subheader("📊 Resumen cuantitativo del dataset")
    try:
        data_path = Path(__file__).parent.parent.parent / "data"
        matriz_original = pd.read_csv(data_path / "matriz_expresion.csv", index_col=0)
        n_genes_original = len(matriz_original)
    except Exception:
        matriz_original = None
        n_genes_original = None

    n_genes_symbol = len(expr_data)
    try:
        n_genes_filtered = len(non_specific_gene_filter(expr_data))
    except Exception:
        n_genes_filtered = None

    # Conteo simple por grupos (para contextualizar)
    try:
        counts = (covariables
                  .assign(Dia=covariables['Dia'].astype(str))
                  .groupby(['Linea', 'Tratamiento', 'Dia']).size()
                  .reset_index(name='n'))
    except Exception:
        counts = None

    with st.expander("Ver detalles de conteos por grupo", expanded=False):
        if counts is not None and not counts.empty:
            st.dataframe(counts, hide_index=True, use_container_width=True)
        else:
            st.info("Conteos no disponibles.")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Genes iniciales (matriz sin filtrar)", f"{n_genes_original:,}" if n_genes_original else "-" )
    with cols[1]:
        st.metric("Genes alineados (gene symbol)", f"{n_genes_symbol:,}")
    with cols[2]:
        st.metric("Genes tras filtrado no específico", f"{n_genes_filtered:,}" if n_genes_filtered else "-")

    # Hallazgos clave (redactados en base a los módulos implementados)
    st.subheader("🎯 Principales hallazgos analíticos")
    st.markdown(
        """
        - En PCA y t-SNE se observa separación consistente entre líneas celulares (HT29 vs NCM460),
          con efecto adicional del tratamiento (Control vs DFMO). El día actúa como factor secundario.
        - El heatmap clusterizado (Ward/Complete/Average) agrupa las muestras principalmente por Línea y Tratamiento.
          Las barras de anotación confirman la coherencia entre covariables y clusterización.
        - Las distribuciones de expresión por muestra no muestran sesgos globales severos: media/mediana similares,
          SD e IQR en rangos esperables, asimetría y curtosis dentro de lo razonable.
        - Las tablas de correlación (PCA/t-SNE) permiten priorizar genes asociados a los ejes latentes; disponibles
          para descarga completa (todos los genes) en las secciones respectivas.
        """
    )

    # Implicaciones biológicas (concisas y accionables)
    st.subheader("🧬 Implicaciones biológicas")
    st.markdown(
        """
        - DFMO muestra impacto más marcado en la línea HT29, consistente con un efecto más pronunciado en contexto tumoral.
        - Posibles vías afectadas: metabolismo de poliaminas, proliferación y regulación de ciclo celular (a validar).
        - El factor "Día" en este diseño es discreto (no longitudinal) y actúa como covariable/bloque; no se infiere dinámica temporal.
        """
    )

    # Próximos pasos
    st.subheader("⏭️ Próximos pasos recomendados")
    st.markdown(
        """
        1) Validación experimental: qPCR de genes priorizados por correlación; incluir housekeeping apropiados.
        2) Enriquecimiento funcional: GSEA/ORA sobre listas ordenadas por correlación/t-stat con ajuste FDR.
        3) Modelado adicional: efectos fijos (Línea, Tratamiento, Día) y su interacción; evaluar batch si aplica.
        4) Integración: contrastar con literatura (GeneCards/KEGG/Reactome) y posibles datos proteómicos si existen.
        """
    )

    # Limitaciones
    st.subheader("⚠️ Limitaciones del estudio")
    st.markdown(
        """
        - Tamaño muestral reducido limita la robustez de t-SNE (perplejidad restringida) y potencia estadística.
        - Microarrays: genes no cubiertos por sondas o mapeos ambiguos pueden perderse pese a la agregación por mediana.
        - El uso de líneas celulares in vitro acota la extrapolación; se recomienda validación en modelos adicionales.
        """
    )

    # Descarga de conclusiones en Markdown
    md = [
        "# Conclusiones del análisis",
        "## Resumen cuantitativo",
        f"- Genes iniciales (sin filtrar): {n_genes_original if n_genes_original else '-'}",
        f"- Genes alineados (gene symbol): {n_genes_symbol}",
        f"- Genes tras filtrado no específico: {n_genes_filtered if n_genes_filtered else '-'}",
        "\n## Principales hallazgos",
        "- Separación por Línea y Tratamiento en PCA/t-SNE; Día como efecto secundario.",
        "- Clusterización coherente con covariables en heatmap (Ward/Complete/Average).",
        "- Distribuciones por muestra sin sesgos globales marcados (media≈mediana; SD/IQR razonables).",
        "- Tablas de correlación completas disponibles para descarga en cada sección.",
        "\n## Implicaciones biológicas",
        "- Efecto de DFMO más marcado en HT29; posible especificidad tumoral.",
        "- Vías candidatas: poliaminas/proliferación (a validar).",
        "- El factor Día se trata como covariable discreta/bloque; no implica dinámica temporal en este diseño.",
        "\n## Próximos pasos",
        "1) qPCR genes priorizados; 2) Enriquecimiento (GSEA/ORA); 3) Modelado con covariables; 4) Integración." ,
        "\n## Limitaciones",
        "- N muestral limitado (t-SNE, potencia). Microarrays y modelo in vitro." 
    ]
    md_text = "\n".join(md)

    st.download_button(
        label="📥 Descargar conclusiones (Markdown)",
        data=md_text,
        file_name="conclusiones.md",
        mime="text/markdown"
    )