import streamlit as st

def show_conclusions():
    """
    Muestra la sección de conclusiones del análisis.
    """
    st.header("📑 Conclusiones del Análisis")
    
    # Principales hallazgos
    st.subheader("🎯 Principales Hallazgos")
    
    st.write("""
    1. **Validación del Diseño Experimental**
    - Clara separación entre condiciones experimentales
    - Efectos detectables del tratamiento con DFMO
    - Control técnico adecuado de la variabilidad
    
    2. **Respuesta Diferencial al DFMO**
    - Efecto más pronunciado en células cancerosas (HT29)
    - Respuesta heterogénea en células normales (NCM460)
    - Sugestivo de especificidad terapéutica
    
    3. **Calidad de los Datos**
    - Normalización efectiva
    - Variación técnica controlada
    - Base sólida para análisis posteriores
    """)
    
    # Implicaciones biológicas
    st.subheader("🧬 Implicaciones Biológicas")
    
    st.write("""
    1. **Especificidad del DFMO**
    - Mayor efecto en células cancerosas
    - Potencial terapéutico selectivo
    - Menor impacto en células normales
    
    2. **Mecanismos Moleculares**
    - Alteración de vías metabólicas específicas
    - Cambios en genes relacionados con proliferación
    - Modulación de la expresión génica diferencial
    
    3. **Relevancia Clínica**
    - Posible uso como terapia dirigida
    - Necesidad de validación adicional
    - Potencial biomarcador de respuesta
    """)
    
    # Próximos pasos
    st.subheader("⏭️ Próximos Pasos")
    
    st.write("""
    1. **Validación Experimental**
    - Confirmación por qPCR de genes clave
    - Ensayos funcionales adicionales
    - Estudios de dosis-respuesta
    
    2. **Análisis Adicionales**
    - Enriquecimiento de vías metabólicas
    - Análisis de redes de interacción
    - Integración con datos proteómicos
    
    3. **Desarrollo Terapéutico**
    - Optimización de dosis
    - Estudios de combinación
    - Biomarcadores de respuesta
    """)
    
    # Limitaciones del estudio
    st.subheader("⚠️ Limitaciones del Estudio")
    
    st.write("""
    1. **Técnicas**
    - Limitaciones inherentes a microarrays
    - Necesidad de validación por otras técnicas
    - Tamaño muestral limitado
    
    2. **Biológicas**
    - Modelo in vitro
    - Líneas celulares específicas
    - Tiempo de tratamiento único
    
    3. **Analíticas**
    - Necesidad de validación adicional
    - Posibles sesgos técnicos
    - Limitaciones en la interpretación
    """)