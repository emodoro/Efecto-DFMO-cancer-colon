# Preprocesamiento de Datos Transcriptómicos del Estudio DFMO en Cáncer de Colon

## 📋 Resumen Ejecutivo

Este documento detalla el proceso de preparación de datos transcriptómicos obtenidos de un estudio que investiga el efecto del DFMO (difluorometilornitina) en células de cáncer de colon y colonocitos normales. El preprocesamiento es un paso crucial para garantizar la calidad y fiabilidad de los análisis posteriores.

## 🎯 Objetivo del Preprocesamiento

El objetivo principal es transformar los datos crudos de microarrays en una matriz de expresión génica limpia y estructurada que permita realizar análisis estadísticos robustos y obtener conclusiones biológicamente significativas.

## 🔬 Contexto Experimental

- **Tecnología**: Microarrays Clariom D Human de Affymetrix
- **Diseño Experimental**:
  - Dos tipos de líneas celulares:
    - Células de cáncer de colon (HT29).
    - Colonocitos normales (NCM460).
  - Dos condiciones por línea:
    - Tratadas con DFMO.
    - Sin tratar (control).
- **Tratamiento**: DFMO, un inhibidor suicida de la ODC (ornitina descarboxilasa).

## 🛠 Proceso de Preprocesamiento

### 1. Preprocesamiento Inicial en R
> *Nota: Este paso fue realizado previamente*
1. Lectura de datos crudos (.CEL files) utilizando el paquete `oligo` [enlace a documentación](https://www.bioconductor.org/packages/release/bioc/vignettes/oligo/inst/doc/oug.pdf).
2. Control de calidad inicial mediante la evaluación de los gráficos MA, de NUSE, RLE ([enlace1](https://github.com/slowkow/arrayqc), [enlace2](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://pmc.ncbi.nlm.nih.gov/articles/PMC3582178/pdf/nihms138776.pdf)) y PCA para detectar posibles outliers.
3. Método: RMA (Robust Multi-array Average), que incluye:
    - Corrección de fondo.
    - Normalización entre arrays por cuantiles.
    - Resumen a nivel de probeset.

### 2. Preprocesamiento en Python

#### 2.1 Lectura y Verificación de Datos
1. Carga de matrices de expresión y de covariables.
2. Verificación de valores faltantes:
  2.1 No se encontraron valores faltantes en ninguna matriz.
  2.2 Esto indica una alta calidad en la captura de datos.

#### 2.2 Reorganización de Datos
1. **Renombrado de Muestras**
Con el fin de mejorar la claridad y comprensión de los datos, las columnas se renombraron usando identificadores más informativos basados en la línea celular y el tratamiento.

2. **Reordenamiento Estratégico**
    Se estableció un nuevo orden para mejor interpretación:
    - Colonocitos normales (NC460) sin tratar.
    - Colonocitos normales (NC460) con DFMO.
    - Células cancerosas (HT29) sin tratar.
    - Células cancerosas (HT29) con DFMO.

#### 2.3 Mapeo de Probesets a Genes
1. **Proceso de Mapeo**
   - Integración con archivo de anotación `probe2symbol.csv`.
   - Eliminación de probesets sin símbolo de gen asociado.
   - Resultado: Matriz más interpretable biológicamente.

2. **Manejo de Duplicados**
   - En ocasiones pueden existir múltiples probesets por gen. Para abordar este desafío, se implementó la siguiente solución: Agregación por mediana. Esto trae consigo una serie de ventajas:
     - Permite consolidar las mediciones en un solo valor representativo por gen. Es decir, se obtiene una única expresión génica por cada gen, facilitando su análisis posterior.
     - Proporciona robustez frente a valores extremos.
     - Se logra una primera reducción de dimensionalidad manteniendo información biológica.

## 📊 Resultados del Preprocesamiento

### Transformación de Datos
- **Matriz Original (`matriz_expresion.scv`)**: 
  - Filas: Probesets individuales.
  - Columnas: Muestras experimentales.
- **Matriz Final (`matriz_expr_symbol_median.csv`)**: 
  - Filas: Genes únicos (símbolos).
  - Columnas: Muestras experimentales ordenadas por condición.

### Impacto en la Dimensionalidad
- Reducción significativa de dimensiones.
- Eliminación de redundancia.
- Mantenimiento de la integridad biológica de los datos.

## 🎯 Conclusiones

1. **Calidad de Datos**
   - No se detectaron valores faltantes.
   - Proceso de normalización efectivo.
   - Mapeo exitoso de probesets a genes.

2. **Optimización de Estructura**
   - Organización lógica de muestras.
   - Eliminación de redundancia en mediciones.
   - Base sólida para análisis posteriores.

3. **Consideraciones Biológicas**
   - Preservación de la información relevante.
   - Reducción de ruido técnico.
   - Facilitación de interpretación biológica.

## 📝 Notas Técnicas

- Los archivos procesados se guardan en el directorio `data/`
- Nombres de archivos finales:
  - `matriz_expr_symbol_median.csv`.
  - `matriz_covariables_ordenada.csv`.

## 🔄 Próximos Pasos

El conjunto de datos está ahora preparado para el análisis exploratorio de datos, reducción de la dimensionalidad, identificación de patrones y descubrimiento de biomarcadores relevantes en el contexto del cáncer de colon y del efecto del DFMO tanto en líneas celulares de colonocitos normales como de cáncer de colon.
