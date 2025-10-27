# 🧬 Efecto de DFMO en Cáncer de Colon — Panel interactivo y análisis reproducible

Este repositorio reúne un pipeline reproducible de preprocesamiento, EDA y un panel interactivo (Streamlit) para explorar el efecto del DFMO (difluorometilornitina) en células de cáncer de colon (HT29) frente a colonocitos normales (NCM460) usando datos de microarrays (Clariom D Human).

Incluye:
- Datos procesados y anotados (CSV) listos para análisis.
- Documentación narrativa y técnica del preprocesamiento y EDA.
- App Streamlit con exploración, análisis dimensional (PCA/t‑SNE) y heatmaps clusterizados con anotaciones de covariables.
- Descargas de figuras y tablas para trazabilidad y comunicación de resultados.


## Contenidos principales

- app/: aplicación Streamlit con componentes modulares
- data/: matrices de expresión y covariables procesadas
- docs/: guías narrativas de preprocesamiento y EDA
- notebooks/: cuadernos originales de trabajo (prep.ipynb, eda.ipynb)
- resultados/: carpeta para tablas y figuras exportadas


## Arquitectura de la solución

La app está diseñada por componentes, con utilidades para cargar/filtrar/normalizar datos y funciones de análisis. Flujo principal:

1) Carga de datos (utils/data_loader.py)
	- matriz_expr_symbol_median.csv: expresión por gene symbol (agregada por mediana)
	- matriz_covariables_ordenada.csv: covariables por muestra (Línea, Tratamiento, Día)

2) Pre‑filtrado no específico (utils/data_loader.non_specific_gene_filter)
	- Mantiene genes con IQR y mediana en cuartiles superiores para reducir ruido.

3) Exploración y análisis (app/components/)
	- overview.py: resumen del estudio y métricas de dataset
	- data_explorer.py: heatmap clusterizado con dendrogramas y anotaciones (Tratamiento/Línea/Día), exploración de genes específicos y distribuciones
	- analysis.py: PCA, t‑SNE y heatmap clusterizado avanzado; tablas de correlación gen‑componente con FDR
	- conclusions.py: conclusiones cuantitativas y cualitativas con exportación Markdown

4) Visualización y descargas (utils/plotting.py, Plotly)
	- Exportación a HTML interactivo y PNG (opcional con kaleido)

5) Orquestación (app/main.py)
	- Sidebar con filtros: Línea, Tratamiento, Normalización, Método de clustering, Nº de genes
	- Pestañas: Visión General, Explorador, Análisis, Conclusiones


## Datos y origen

Carpeta data/ (ya versionada para reproducibilidad):
- matriz_expresion.csv: matriz original a nivel de probeset (sin filtrar)
- probe2symbol.csv: mapeo probeset → gene symbol
- matriz_expr_symbol_median.csv: expresión a nivel de gene symbol (agregada por mediana entre probesets)
- matriz_covariables.csv: covariables originales
- matriz_covariables_ordenada.csv: covariables curadas/ordenadas

Notas de proceso (ver docs/prep.md):
- Normalización RMA (Affymetrix) y control de calidad inicial en R (oligo)
- Mapeo a gene symbol y agregación por mediana para reducir redundancia
- Orden lógico de muestras (por Línea, Tratamiento y Día)


## App Streamlit: funcionalidades clave

Sidebar (filtros):
- Línea Celular: HT29, NCM460
- Tratamiento: Control, DFMO
- Normalización: Sin normalizar, Z‑score, Min‑Max
- Clustering: Ward, Complete, Average
- Nº de genes: selección por mayor varianza

Pestañas:
- Visión General: contexto del estudio, métricas (genes iniciales, alineados, filtrados, nº de muestras), distribución de muestras por línea/tratamiento/día
- Explorador de Datos: heatmap clusterizado con dendrogramas y anotaciones de covariables; exploración de genes (boxplots); distribuciones por muestra con estadísticas (μ, mediana, SD, IQR, asimetría, curtosis)
- Análisis: PCA y t‑SNE con selección de componentes, rangos ajustados y colores legibles; tablas de correlación gene‑ejes (Spearman, t, p, FDR) y descarga CSV; heatmap clusterizado avanzado
- Conclusiones: resumen cuantitativo (conteos de genes y muestras), hallazgos clave, implicaciones biológicas, próximos pasos y limitaciones; descarga de conclusiones en Markdown

Exportaciones:
- HTML interactivo de todos los gráficos (siempre disponible)
- PNG de alta resolución (requiere kaleido)
- CSV con datos (por ejemplo, matriz del heatmap reordenada y tablas de correlación)


## Cómo ejecutar (Windows PowerShell)

Requisitos:
- Python 3.10+ recomendado
- Pip actualizado

Pasos:
1) Crear entorno virtual e instalar dependencias
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
   - pip install -r requirements.txt

2) (Opcional para exportar PNG en local) Instalar kaleido
   - pip install kaleido

3) Iniciar la app
   - streamlit run app/main.py

### Deploy en Streamlit Cloud

**Nota importante:** La exportación a PNG no está disponible en Streamlit Cloud (requiere librerías del sistema). 
Los usuarios pueden descargar:
- ✅ **HTML interactivo** (siempre disponible — mejor opción en Cloud)
- ✅ **CSV** con datos para análisis externo
- ✅ **PNG** (solo en desarrollo local con kaleido)

Solución de problemas comunes:
- Error al cargar CSS: asegúrate de que app/styles/main.css existe (viene incluido) y que ejecutas desde la raíz del repo.
- Error al cargar datos: verifica que los CSV de data/ están presentes y con permisos de lectura.
- Exportación PNG falla en local: instala kaleido y reinicia la app.
- En Streamlit Cloud: no verás warnings de PNG (se ocultan automáticamente).
## Reproducibilidad y notebooks

- notebooks/prep.ipynb: preprocesamiento y anotación; documentado en docs/prep.md
- notebooks/eda.ipynb: análisis exploratorio; narrado en docs/eda.md

Las guías en docs/ ofrecen una versión narrativa y ejecutiva del trabajo para comunicación con stakeholders no técnicos y para orientar desarrollos futuros del panel.


## Metodología analítica (resumen)

- Filtrado no específico por IQR y mediana (utils/data_loader.non_specific_gene_filter)
- Normalización por gen (Z‑score/Min‑Max) aplicada según necesidad de cada análisis
- PCA (sklearn) sobre datos centrados; reporte de varianza explicada y scores
- t‑SNE (sklearn) con control de perplejidad e iteraciones; soporte hasta 3 componentes
- Correlaciones gene‑eje (Spearman) con ajuste FDR (statsmodels); enlaces a GeneCards por cada gen
- Heatmaps con dendrogramas de genes y muestras; anotaciones de Tratamiento/Línea/Día y leyendas manuales


## Estructura del proyecto

```
Efecto-DFMO-cancer-colon/
├─ app/
│  ├─ main.py
│  ├─ components/
│  │  ├─ overview.py
│  │  ├─ data_explorer.py
│  │  ├─ analysis.py
│  │  ├─ conclusions.py
│  │  └─ sidebar.py
│  ├─ styles/
│  │  └─ main.css
│  └─ utils/
│     ├─ data_loader.py
│     ├─ analysis_utils.py
│     └─ plotting.py
├─ data/
│  ├─ matriz_expresion.csv
│  ├─ matriz_expr_symbol_median.csv
│  ├─ matriz_covariables.csv
│  ├─ matriz_covariables_ordenada.csv
│  └─ probe2symbol.csv
├─ docs/
│  ├─ prep.md
│  ├─ eda.md
│  └─ prompts.md
├─ notebooks/
│  ├─ prep.ipynb
│  └─ eda.ipynb
├─ resultados/
│  ├─ img/
│  └─ tablas/
├─ requirements.txt
├─ LICENSE
└─ README.md
```


## Resultados y hallazgos (highlights)

- Separación consistente HT29 vs NCM460 (PCA/t‑SNE); tratamiento (DFMO) como factor secundario pero visible
- Clusterización coherente con covariables; heatmaps muestran patrones por Línea y Tratamiento
- Distribuciones por muestra sin sesgos globales severos; métricas descriptivas dentro de rangos esperables
- Listas priorizadas de genes por correlación con ejes latentes y posibilidad de descarga completa para downstream (GSEA/ORA)


## Contribución y roadmap

Ideas de mejora:
- Integrar análisis de enriquecimiento (GSEA/ORA) sobre listas ordenadas por correlación/t‑stat
- Añadir modelado con efectos fijos (Línea, Tratamiento, Día) y comparaciones específicas
- Añadir pruebas automatizadas mínimas para utilidades (I/O y cálculos de estadísticos)

Contribuciones son bienvenidas vía PR. Mantén estilo PEP8 y docstrings en nuevas funciones.


## Licencia

Este proyecto se distribuye bajo la licencia indicada en LICENSE.

