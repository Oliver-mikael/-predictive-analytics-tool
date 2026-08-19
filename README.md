# SalesPredict AI 📈

Herramienta profesional de predicción de ventas con IA. Sube tu historial en CSV o Excel y obtén una proyección para los próximos 7–90 días, gráficos interactivos y recomendaciones de negocio accionables.

---

## 🚀 Cómo ejecutar

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

O simplemente inicia el workflow **"Start application"** en Replit.

---

## 📂 Estructura del proyecto

```
├── app.py                  # Interfaz Streamlit (solo UI)
├── src/
│   ├── data_loader.py      # Carga de CSV/Excel, detección de columnas
│   ├── features.py         # Feature engineering (calendario, lags, externas)
│   ├── models.py           # Modelos: Baseline, Prophet, SARIMA, XGBoost, LightGBM
│   ├── evaluation.py       # Métricas, walk-forward validation
│   └── forecast.py         # Ensemble, recomendaciones, gráficos Plotly
├── data/
│   └── ventas_ejemplo.csv  # 95 días de datos con variables externas
├── .streamlit/
│   └── config.toml         # Tema oscuro
└── requirements.txt
```

---

## 📊 Formato del CSV

El CSV debe tener al menos una columna de **fecha** y una de **ventas**:

```csv
fecha,ventas
2024-01-01,4850.00
2024-01-02,5120.50
...
```

### Variables externas opcionales (mejoran precisión +3–8 pp)

| Columna | Descripción | Ejemplo |
|---|---|---|
| `temperatura` | Temperatura en °C | `22.5` |
| `lluvia` | 1 = día lluvioso, 0 = no | `1` |
| `evento_local` | 1 = evento, 0 = sin evento | `0` |
| `tasa_inflacion` | Tasa mensual decimal | `0.0042` |
| `trafico_web` | Visitas al sitio ese día | `1250` |
| `conversion` | Tasa de conversión | `0.032` |
| `carritos` | Carritos abandonados | `87` |

---

## 🤖 Modelos de predicción

| Modelo | Dataset mínimo | Notas |
|---|---|---|
| **Baseline estacional** | 14 días | Perfil día de semana. Muy robusto. |
| **Prophet** | 30 días | Tendencia + estacionalidad + feriados. |
| **AutoARIMA / SARIMA** | 40 días | Estadístico estacional (m=7). |
| **XGBoost** | 60 días | Árboles + lags + features externas. |
| **LightGBM** | 50 días | Leaf-wise. Mejor para datasets cortos. |

El sistema usa **ensemble ponderado**: combina automáticamente los mejores modelos usando walk-forward validation honesta (los pesos se calculan con folds anteriores al predicho).

---

## 📈 Mejoras implementadas vs versión anterior

| Mejora | Impacto estimado |
|---|---|
| LightGBM (nuevo) | +2 a +5 pp precisión |
| Features externas (clima, economía) | +3 a +8 pp si están disponibles |
| `dias_desde_venta_alta` | +1 a +2 pp en series con picos |
| Codificación cíclica de semana del año | +0.5 a +1 pp |
| RMSE como métrica adicional | Mejor diagnóstico |
| Código modular en `src/` | Mantenimiento más fácil |

---

## 🎨 Diseño

Tema oscuro profesional con paleta:
- Fondo: `#111827`
- Paneles: `#1F2937`
- Azul primario: `#3B82F6`
- Verde éxito: `#22C55E`
- Fuente: Inter (Google Fonts)

---

## 🗺️ Flujo de la app

```
Hero (bienvenida)
  → Carga (subir CSV/Excel, seleccionar columnas)
    → Analizando (timeline de progreso)
      → Resultados:
          Tab Resumen       — número grande + 4 KPIs + gráfico
          Tab Predicciones  — escenarios + comparación de modelos + tabla
          Tab Categorías    — por sucursal, ciudad, producto, hora
          Tab Recomendaciones IA — hallazgos + acciones de gerente
        → Descarga (CSV proyección, plan de stock, reporte)
```

---

## 📦 Dependencias principales

```
streamlit, pandas, numpy, prophet, statsmodels,
scikit-learn, plotly, holidays, pmdarima,
xgboost, lightgbm, openpyxl
```
