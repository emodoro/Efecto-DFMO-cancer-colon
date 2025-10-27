# Fix para Deploy en Streamlit Cloud

## Problema identificado

El error **"Error installing requirements"** fue causado por:

1. **`pywin32==311`** — Dependencia específica de Windows que no existe en Linux (donde corre Streamlit Cloud)
2. **Todas las dependencias de Jupyter, notebooks y desarrollo** — No son necesarias en producción y añaden complejidad innecesaria

## Solución aplicada

### ✅ requirements.txt optimizado

Se ha reemplazado el archivo `requirements.txt` con versiones **mínimas y necesarias**:

```
streamlit==1.50.0
pandas==2.3.3
numpy==2.3.3
plotly==6.3.1
scikit-learn==1.7.2
scipy==1.16.2
statsmodels==0.14.5
```

**Beneficios:**
- ✅ Compatible con Linux (Streamlit Cloud)
- ✅ Tiempo de instalación más rápido (~3-5 min vs 10-15 min)
- ✅ Tamaño de app más pequeño
- ✅ Menos dependencias de conflicto

### ✅ Configuración de Streamlit

Se creó `.streamlit/config.toml` con:
- Tema personalizado (colores y fuentes)
- Toolbar en modo minimal para mejor UX
- Logging configurado

### 📦 Respaldo

El archivo `requirements_full.txt` contiene todas las dependencias originales (para desarrollo local)

## Próximos pasos

1. **Hace push al repo:**
   ```bash
   git add requirements.txt .streamlit/config.toml
   git commit -m "fix: optimize requirements for Streamlit Cloud deployment"
   git push
   ```

2. **En Streamlit Cloud:**
   - Ve a "Manage App"
   - Click en "Reboot app"
   - Los logs deberían mostrar instalación exitosa

3. **Verificación:**
   - La app debería cargar en ~2-3 minutos
   - Abre la URL y prueba todos los tabs y filtros

## Si necesitas PNG export en Cloud

Streamlit Cloud no soporta `kaleido` de forma nativa (requiere librerías del sistema). 
**Alternativa:** Los usuarios pueden exportar a HTML (que es interactivo y funciona mejor).

Para desarrollo local, puedes seguir usando:
```bash
pip install kaleido
```

---

**Contacto:** Si vuelve a fallar, revisa los logs en Streamlit Cloud → Manage App → Logs
