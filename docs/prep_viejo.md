# Expresión transcriptómica de líneas celulares de cáncer de colon

El presente proyecto parte de una serie de datos de expresión a nivel transcriptímico a los cuales, previamente en el entorno R, se les ha corregido el fondo, han sido normalizados y resumidos mediante el método *RMA (Robust Multi-Array Average)*. 

A partir de estos datos (ver apartado [Datos de partida tras el preprocesado 1](#datos-de-partida-tras-el-preprocesado-1)).

## Preprocesado 1

El presente proyecto parte de una serie de datos de expresión a nivel transcriptímico ya a los que previamente se les ha corregido el fondo, han sido normalizados y resumidos mediante el método *RMA (Robust Multi-Array Average)*. Su objetivo es eliminar variaciones técnicas y obtener una medida fiable de la expresión génica a partir de las intensidades crudas de las sondas. El proceso consta de tres pasos principales: (1) corrección de fondo, que ajusta las intensidades para reducir el ruido no biológico; (2) normalización cuantílica, que iguala las distribuciones de intensidades entre arrays para hacerlos comparables; y (3) resumen, que combina las señales de las múltiples sondas asociadas a cada gen mediante un modelo lineal robusto, produciendo finalmente valores log₂ de expresión génica comparables entre muestras.

Esta primera parte del proceso ha sido realizada, previamente, en el entorno R. Como resultado, se obtienen los datos que serán analizados en el presente proyecto.

## Datos de partida tras el preprocesado 1

Los datos , disponibles en la carpeta `data` de partida consisten en:
-  `matriz_expresion.csv`: matriz de expresión transcriptómica de dimensiones *genesXmuestra*. En los datos procedentes de experimentos transcriptómicos se suele seguir el siguiente convenio:
    - Cada fila (variable) es un gen.
    - Cada columna es una muestra.
Esto puede resultar confuso para personas acostumbradas al análisis de datos en otros campos donde cada registro (fila) suele ser la muestra y cada campo (columna) una variable.
- `matriz_covariables`: matriz con información adicional sobre cada muestra, de dimensiones *muestraXcovariables*.
- `probe2symbol.csv`: Existen diferentes anotaciones para los nombres de los genes: **ENSEMBL, SYMBOL, ENTREZ**... Además, en los experimentos transcriptómicos realizados con microarrays, se emplea, de partida, la anotación de las sondas con las que se hibrida cada transcrito. Esta anotación, **PROBE**, depende del microarray y son "nombres propios" creados por la empresa que desarrolla el microarray empleado (como puede ser Affymetrix).
    Por ello, es necesario una tabla de correspondencias entre el nombre de la sonda dado por la empresa, **PROBE**, y otra anotación de interés (**SYMBOL** en este caso).

## Preprocesado 2

Tras el preprocesado 1, realizado en el entorno de R, se genera una matriz de expresión normalizada, resumida y con el ruido de fondo ya corregido, `matriz_expresion.csv`. Esta matriz será la que se utilice en el presente proyecto, junto con información adicional de cada muestra, `matriz_covariables.csv`. Además, con el fin de identificar fácilmente los genes analizados, los identificadores de estos, que originalmente están en anotación **PROBE** (el nombre de la sonda que proporciona la empresa, nada intuitivo), se mapean con sus correspondientes símbolos, **SYMBOL**.

A continuación, se explica el preprocesamiento realizado en el entorno de python, al que, para diferenciarlo del realizado previamente en R, se le ha denominado **preprocesado 2**.

### Distribución de la expresión

Aunque es esperable que existan genes diferencialmente expresados (DEGs) entre las distintas condiciones biológicas, la distribución global de los niveles de expresión debería ser similar entre todas las muestras. En este contexto, se considera la expresión génica como una variable aleatoria y a cada gen como una observación o individuo de dicha distribución. La similitud u homogeneidad en la forma general de las distribuciones (centralización, dispersión, curtosis, asimetría...) garantiza que las diferencias detectadas entre condiciones reflejen verdaderos cambios biológicos y no sesgos técnicos o deficiencias en la normalización.  


### Filtrado de genes

Por norma general, no todos los genes se expresan a la vez en todas las células. Por ello, el primer paso tras la normalización y resumen de los chips, es filtrar los genes y eliminar aquellos "no informativos". Este paso es crítico, por lo que los resultados finales del análisis de los datos transcriptómicos dependerá, en gran parte, del filtrado que se haya llevado a cabo. Existen diferentes metodologías para ello; entre las más comunes están:

- Filtrado no específico. Aquí se trata de eliminar del data set los genes no informativos sin tener en cuenta el diseño experimental. Para ello, se filtran: 
    - Sondas problemáticas, sin mapeo fiable.
    - Controles (sondas que empiezan por AFFX...).
    - Sondas con baja expresión: aquellas cuya intensidad es cercana al nivel de fondo en la mayoría de las muestras. Para ello:
        1. Se estima la mediana de cada gen *i* (a lo largo de las muestras): $med_i$.
        2. Se fija un percentil de orden $p_1$ de las $med_i$: $q_{p1}(med)$.
        3. Se mantienen los genes tal que $med_i>q_{p1}(med)$.
    - Sondas con poca variabilidad: la expresión de las sondas varía poco a lo largo de las muestras, IQR o MAD bajos.
        1. Se estima el IQR o el MAD de cada gen *i* (a lo largo de las muestras): $IQR_i$ o $MAD_i$.
        2. Se fija un percentil de orden $p_2$ de los $IQR_i$ (o de $MAD_i$): $q_{p2}(IQR)$ (o $q_{p2}(MAD)$). ¡¡¡Ojo!!! el orden $p_1$ no tiene por qué coincidir con el orden $p_2$.
        3. Se mantienen los genes tal que $IQR_i>q_{2p}(IQR)$ (o $MAD_i>q_{p2}(MAD)$).
- K sobre A: Variante del filtrado de baja expresión, en la que se exige que al menos k muestras presenten una expresión y/o una variabilidad superior a un umbral "a" (diferente para la expresión que para la variabilidad). Este criterio sigue siendo no específico siempre que no se utilice información sobre los grupos experimentales, y permite conservar genes que están expresados en una fracción mínima de las muestras.
En la práctica, suele considerarse como valor de k aproximadamente la mitad del tamaño muestral del grupo más pequeño, de modo que el gen se conserve si está expresado en una proporción representativa de las muestras. 

Para más información, recomiendo el material [Bioinformática Estadística. Estadística de datos ómicos](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.uv.es/ayala/docencia/tami/tami13.pdf), de Guillermo Ayala.

