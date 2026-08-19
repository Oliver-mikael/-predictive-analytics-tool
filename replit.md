# SalesPredict AI — Dashboard predictivo de ventas

## Descripción
Aplicación de forecasting de ventas desarrollada en Python y Streamlit. El objetivo principal es producir pronósticos precisos y honestos mediante validación temporal. La precisión del modelo tiene prioridad sobre agregar funcionalidades nuevas.

## Stack
- **Lenguaje:** Python 3.12
- **Framework UI:** Streamlit
- **Modelos:** Baseline, Baseline estacional, Facebook Prophet, SARIMA/AutoARIMA (statsmodels + pmdarima), XGBoost y LightGBM cuando está disponible
- **Gráficos:** Plotly
- **Datos:** pandas, numpy

## Cómo ejecutar
```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

El workflow **"Start application"** está configurado y arranca automáticamente.

## Estructura de archivos
```
app.py                  — Interfaz Streamlit y coordinación general
src/data_loader.py      — Carga, limpieza inicial, detección de columnas y fechas
src/features.py         — Construcción de variables predictivas
src/models.py           — Implementación de modelos de forecasting
src/evaluation.py       — Métricas, folds temporales y selección de modelos
src/forecast.py         — Orquestación del forecast y resultados
data/                   — Datasets de prueba no sensibles
tests/                  — Pruebas automáticas, cuando se agreguen
requirements.txt        — Dependencias Python
.streamlit/config.toml  — Configuración de Streamlit
```

Aunque conceptualmente existen módulos de preprocessing y metrics, en la implementación actual esas responsabilidades están cubiertas por `src/data_loader.py`, `src/features.py` y `src/evaluation.py`. No crear módulos duplicados sin una razón clara.

## Reglas críticas de forecasting

1. No introducir leakage.
2. Nunca utilizar información futura para construir features.
3. Los lags y rolling features deben ser causales.
4. Los parámetros derivados de datos deben calcularse usando solo el entrenamiento de cada fold.
5. La validación debe ser temporal y representar el escenario real de predicción.
6. No utilizar datos del período de validación para seleccionar hiperparámetros.
7. No optimizar artificialmente el MAPE.
8. No cambiar el período de validación solo para obtener mejores métricas.
9. No aceptar una mejora de MAPE si introduce leakage u overfitting.
10. Comparar siempre contra un baseline.
11. Mantener MAPE, WAPE, sMAPE, MAE y RMSE.
12. Un modelo debe completar los folds necesarios para competir justamente.
13. El modelo seleccionado durante la validación debe ser el mismo utilizado en el forecast final.

## Estado y referencia actual

En `data/ventas_ejemplo.csv`:

- 95 días de historial.
- MAPE anterior: 25.97%.
- MAPE actual: 17.45%.
- WAPE actual: 14.15%.
- sMAPE actual: 15.48%.
- Prophet es el modelo ganador actual.
- Existe una fuerte estacionalidad semanal: lag 7 ≈ 0.802, lag 14 ≈ 0.801 y lag 21 ≈ 0.798.
- El domingo presenta el mayor error.

Estos valores son una referencia del dataset de ejemplo, no una garantía para otros datasets.

## Filosofía de experimentación

Antes de implementar cualquier mejora de forecasting:

1. Formular una hipótesis basada en los datos.
2. Ejecutar un benchmark sin modificar la arquitectura.
3. Comparar contra el baseline actual.
4. Medir MAPE, WAPE, sMAPE, MAE y RMSE.
5. Comprobar explícitamente que no existe leakage.
6. Evaluar si la mejora puede generalizar a otros datasets.
7. Integrar permanentemente solo si la evidencia lo justifica.

No asumir que un modelo más complejo será mejor. No asumir que Prophet + XGBoost residual necesariamente mejorará el resultado.

## Prioridad actual

Investigar por qué Prophet obtiene aproximadamente 17.45% de MAPE y determinar si puede reducirse de forma reproducible, con especial atención al comportamiento del domingo.

La interfaz debe permanecer estable durante los experimentos. No agregar funcionalidades de UI durante la optimización salvo que sean necesarias para diagnosticar resultados.

## Verificación antes de finalizar cambios

Ejecutar, cuando corresponda:

- Pruebas automáticas.
- Compilación de los módulos modificados.
- Benchmark con el mismo período de validación.
- Comprobaciones de leakage.
- Confirmación de que Streamlit continúa arrancando.

Informar siempre:

- Archivos modificados.
- Cambios realizados.
- Métricas antes y después.
- Riesgos y limitaciones.
- Siguiente experimento recomendado.

## Formato esperado del CSV
- Una columna con **fechas** (DD/MM/YYYY o MM/DD/YYYY — se auto-detecta)
- Una columna con **montos de venta** (número positivo por transacción o por día)
- Mínimo 30 días de historial; se recomienda 90+ días para mayor precisión
- Columnas opcionales detectadas automáticamente: sucursal, ciudad, producto, hora, cliente, género, método de pago

## Tabs de la aplicación
1. **Proyección** — Gráfico principal + tres escenarios + tabla día a día + descarga CSV
2. **Recomendaciones** — Acciones concretas basadas en el análisis
3. **Análisis** — Gráficos por sucursal, ciudad, producto, hora, día de semana
4. **Técnico** — Comparación de modelos, métricas de error, puntaje de confiabilidad

## User preferences
- Idioma de la UI: Español
- Botones primarios: azul (#2563eb)
- Sin lenguaje técnico en la UI principal (MAPE, WAPE, etc. solo en tab Técnico)
- Sin código Python visible en pantalla
