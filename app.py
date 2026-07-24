import streamlit as st
import pandas as pd
import numpy as np
import warnings
import holidays
warnings.filterwarnings('ignore')

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

# ============================================
# DETECCION DE LIBRERIAS OPCIONALES
# ============================================

HAS_PMDARIMA = False
HAS_XGBOOST = False

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    pass

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

# ============================================
# FEATURE ENGINEERING OPTIMIZADO (8 features)
# ============================================

def crear_features_optimizado(df):
    """
    Crea solo 8 features de alto impacto para evitar overfitting.
    Con datos limitados (< 500 dias), menos features = mejor generalizacion.
    """
    df = df.copy()
    df = df.sort_values('ds').reset_index(drop=True)

    # 1. Patron semanal (lag 7)
    df['lag_7'] = df['y'].shift(7)

    # 2. Patron mensual (lag 30)
    df['lag_30'] = df['y'].shift(30)

    # 3. Tendencia corta (media movil 7 dias)
    df['ma_7'] = df['y'].shift(1).rolling(window=7).mean()

    # 4. Tendencia larga (media movil 30 dias)
    df['ma_30'] = df['y'].shift(1).rolling(window=30).mean()

    # 5. Dia de la semana (0=Lunes, 6=Domingo)
    df['dia_semana'] = df['ds'].dt.dayofweek

    # 6. Fin de semana
    df['es_finde'] = (df['ds'].dt.dayofweek >= 5).astype(int)

    # 7. Mes
    df['mes'] = df['ds'].dt.month

    # 8. Ratio venta vs tendencia corta
    df['ratio_ma7'] = df['y'].shift(1) / (df['ma_7'] + 1)

    # Eliminar filas con NaN (las primeras 30 filas no tienen lag_30)
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================
# FUNCIONES DE LIMPIEZA Y UTILIDAD
# ============================================

def limpiar_datos(df, col_fecha, col_ventas):
    """
    Limpia datos y retorna (df_limpio, diccionario_validacion)
    """
    df_limpio = pd.DataFrame()
    df_limpio['ds'] = pd.to_datetime(
        df[col_fecha], dayfirst=True, errors='coerce'
    )
    df_limpio['y'] = pd.to_numeric(
        df[col_ventas], errors='coerce'
    )
    df_limpio = df_limpio.dropna()
    df_limpio = df_limpio[df_limpio['y'] >= 0]
    df_limpio = df_limpio.sort_values('ds')
    df_limpio = df_limpio.groupby('ds', as_index=False)['y'].sum()

    # Rellenar dias faltantes con cero
    rango = pd.DataFrame({
        'ds': pd.date_range(
            start=df_limpio['ds'].min(),
            end=df_limpio['ds'].max(),
            freq='D'
        )
    })
    df_limpio = rango.merge(df_limpio, on='ds', how='left')
    df_limpio['y'] = df_limpio['y'].fillna(0)

    # Outliers: clip al Q3 + 3*IQR
    Q1 = df_limpio['y'].quantile(0.25)
    Q3 = df_limpio['y'].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 3 * IQR
    df_limpio['y'] = df_limpio['y'].clip(upper=limite_superior)

    # Validaciones
    dias_disponibles = (df_limpio['ds'].max() - df_limpio['ds'].min()).days
    cantidad_registros = len(df_limpio)
    pct_zeros = (df_limpio['y'] == 0).sum() / len(df_limpio) * 100

    if dias_disponibles < 30:
        estado = "ERROR"
        mensaje = f"Pocos datos: Solo {dias_disponibles} dias. Minimo 30 requerido."
    elif pct_zeros > 40:
        estado = "ERROR"
        mensaje = f"Datos anomalos: {pct_zeros:.1f}% son cero. Verifica el CSV."
    elif pct_zeros > 20:
        estado = "WARNING"
        mensaje = f"{pct_zeros:.1f}% de datos son cero. Precision puede ser menor."
    elif dias_disponibles < 60:
        estado = "WARNING"
        mensaje = f"Solo {dias_disponibles} dias. Se recomienda 60+ para mejor precision."
    else:
        estado = "OK"
        mensaje = "Datos validos para analisis."

    validacion = {
        'dias': dias_disponibles,
        'registros': cantidad_registros,
        'pct_zeros': round(pct_zeros, 2),
        'estado': estado,
        'mensaje': mensaje
    }

    return df_limpio, validacion


def obtener_feriados(pais, anos):
    paises_map = {
        'Bolivia': holidays.Bolivia,
        'Mexico': holidays.Mexico,
        'Argentina': holidays.Argentina,
        'Colombia': holidays.Colombia,
        'Peru': holidays.Peru,
        'Chile': holidays.Chile,
        'Espana': holidays.Spain,
        'USA': holidays.US,
        'Brasil': holidays.Brazil,
        'Ecuador': holidays.Ecuador,
        'Venezuela': holidays.Venezuela,
        'Paraguay': holidays.Paraguay
    }
    try:
        clase_feriados = paises_map.get(pais)
        if clase_feriados:
            feriados_lista = []
            for ano in anos:
                f = clase_feriados(years=ano)
                for fecha, nombre in f.items():
                    feriados_lista.append({
                        'holiday': nombre,
                        'ds': pd.Timestamp(fecha)
                    })
            return pd.DataFrame(feriados_lista)
    except:
        pass
    return None

# ============================================
# MODELOS DE PREDICCION
# ============================================

def calcular_mape(real, pred):
    """Calcula MAPE de forma robusta"""
    mask = real > (real.mean() * 0.1)
    if mask.sum() > 0:
        mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100
    else:
        mape = np.mean(np.abs((real - pred) / (real + 1))) * 100
    return mape


def correr_prophet(df_train, df_test, feriados=None):
    """Prophet con configuracion adaptativa"""
    try:
        modelo = Prophet(
            weekly_seasonality=len(df_train) > 60,
            yearly_seasonality=len(df_train) > 365,
            daily_seasonality=False,
            interval_width=0.95,
            holidays=feriados,
            changepoint_prior_scale=0.05
        )
        modelo.fit(df_train)

        futuro = modelo.make_future_dataframe(
            periods=len(df_test), freq='D'
        )
        pred = modelo.predict(futuro)
        pred_test = pred['yhat'].tail(len(df_test)).values
        real_test = df_test['y'].values

        mape = calcular_mape(real_test, pred_test)
        mae = mean_absolute_error(real_test, pred_test)

        return {'nombre': 'Prophet', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': modelo}
    except Exception as e:
        return {'nombre': 'Prophet', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_arima(df_train, df_test):
    """ARIMA(1,1,1) como fallback siempre disponible"""
    try:
        modelo = ARIMA(df_train['y'], order=(1, 1, 1))
        resultado = modelo.fit()
        pred = resultado.forecast(steps=len(df_test))
        real_test = df_test['y'].values
        pred_values = pred.values

        mape = calcular_mape(real_test, pred_values)
        mae = mean_absolute_error(real_test, pred_values)

        return {'nombre': 'ARIMA', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': resultado}
    except Exception as e:
        return {'nombre': 'ARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_autoarima(df_train, df_test):
    """AutoARIMA: busca automaticamente el mejor (p,d,q)"""
    if not HAS_PMDARIMA:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999,
                'error': 'pmdarima no instalado. Instala con: pip install pmdarima'}

    try:
        modelo = auto_arima(
            df_train['y'],
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            max_p=5,
            max_d=2,
            max_q=5,
            n_jobs=1,
            trace=False
        )

        pred = modelo.predict(n_periods=len(df_test))
        real_test = df_test['y'].values
        pred_values = pred.values

        mape = calcular_mape(real_test, pred_values)
        mae = mean_absolute_error(real_test, pred_values)

        return {'nombre': 'AutoARIMA', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': modelo}
    except Exception as e:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_xgboost_optimizado(df_train, df_test):
    """XGBoost con 8 features optimizadas"""
    if not HAS_XGBOOST:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999,
                'error': 'xgboost no instalado. Instala con: pip install xgboost'}

    try:
        # Crear features
        df_train_f = crear_features_optimizado(df_train)
        df_combined = pd.concat([df_train, df_test]).reset_index(drop=True)
        df_test_f = crear_features_optimizado(df_combined)
        df_test_f = df_test_f.tail(len(df_test)).reset_index(drop=True)

        feature_cols = [c for c in df_train_f.columns if c not in ['ds', 'y']]

        X_train = df_train_f[feature_cols]
        y_train = df_train_f['y']
        X_test = df_test_f[feature_cols]
        y_test = df_test_f['y']

        modelo = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,           # Mas bajo = menos overfitting
            learning_rate=0.08,    # Mas lento = mejor generalizacion
            subsample=0.7,         # Menos datos por arbol
            colsample_bytree=0.7,  # Menos features por arbol
            reg_alpha=0.1,         # Regularizacion L1
            reg_lambda=1.0,        # Regularizacion L2
            random_state=42
        )
        modelo.fit(X_train, y_train)

        pred = modelo.predict(X_test)

        mape = calcular_mape(y_test.values, pred)
        mae = mean_absolute_error(y_test, pred)

        return {
            'nombre': 'XGBoost',
            'mape': round(mape, 2),
            'mae': round(mae, 2),
            'modelo': modelo,
            'feature_cols': feature_cols
        }
    except Exception as e:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999, 'error': str(e)}

# ============================================
# VALIDACION Y ENSEMBLE
# ============================================

def walk_forward_validation(df, pais, window_test=30):
    """
    Valida modelos usando los ULTIMOS window_test dias como test.
    Esto es mas realista que un split aleatorio.
    """
    fecha_corte = df['ds'].max() - pd.Timedelta(days=window_test)

    if (df['ds'].max() - fecha_corte).days < 14:
        # Si hay muy pocos datos, usar 80/20 split
        split = int(len(df) * 0.8)
        df_train = df[:split]
        df_test = df[split:]
    else:
        df_train = df[df['ds'] <= fecha_corte].copy()
        df_test = df[df['ds'] > fecha_corte].copy()

    if len(df_train) < 30:
        return None, "Pocos datos para validacion"

    anos = df['ds'].dt.year.unique().tolist()
    anos += [max(anos) + 1]
    feriados = obtener_feriados(pais, anos)

    # Ejecutar todos los modelos
    res_prophet = correr_prophet(df_train, df_test, feriados)
    res_arima = correr_arima(df_train, df_test)
    res_autoarima = correr_autoarima(df_train, df_test)
    res_xgb = correr_xgboost_optimizado(df_train, df_test)

    resultados = [res_prophet, res_arima, res_autoarima, res_xgb]
    resultados_validos = [r for r in resultados if r['mape'] < 100]

    if not resultados_validos:
        return None, "Ningun modelo valido"

    # Ordenar por MAPE
    resultados_validos.sort(key=lambda x: x['mape'])

    return resultados_validos, None


def ensemble_ponderado(resultados):
    """
    Combina los 2-3 mejores modelos con pesos inversamente proporcionales al MAPE.
    Si solo hay 1 modelo bueno, devuelve ese.
    """
    # Filtrar solo los buenos (MAPE < 35)
    buenos = [r for r in resultados if r['mape'] < 35]

    if len(buenos) == 0:
        # Si ninguno es bueno, tomar el mejor disponible
        buenos = [resultados[0]]

    if len(buenos) == 1:
        return buenos[0]

    # Tomar maximo 3 modelos para el ensemble
    buenos = buenos[:3]

    # Pesos inversamente proporcionales al MAPE
    pesos = [1 / max(r['mape'], 1) for r in buenos]
    suma_pesos = sum(pesos)
    pesos = [p / suma_pesos for p in pesos]

    # MAPE ponderado
    mape_ensemble = sum(r['mape'] * p for r, p in zip(buenos, pesos))
    mae_ensemble = sum(r['mae'] * p for r, p in zip(buenos, pesos))

    nombres = '+'.join([r['nombre'] for r in buenos])

    return {
        'nombre': f'Ensemble ({nombres})',
        'mape': round(mape_ensemble, 2),
        'mae': round(mae_ensemble, 2),
        'modelos': buenos,
        'pesos': pesos
    }


def analizar(df, pais, dias_futuro):
    """
    Pipeline completo: Walk Forward -> Ensemble -> Prediccion final
    """
    # 1. Walk Forward Validation
    resultados_wf, error = walk_forward_validation(df, pais, window_test=30)

    if error:
        st.error(f"Error en validacion: {error}")
        return None, None

    # 2. Ensemble de los mejores modelos
    ganador = ensemble_ponderado(resultados_wf)

    # 3. Entrenar modelo final con TODOS los datos para prediccion futura
    anos = df['ds'].dt.year.unique().tolist()
    anos += [max(anos) + 1]
    feriados = obtener_feriados(pais, anos)

    modelo_final = Prophet(
        weekly_seasonality=len(df) > 60,
        yearly_seasonality=len(df) > 365,
        daily_seasonality=False,
        interval_width=0.95,
        holidays=feriados,
        changepoint_prior_scale=0.05
    )
    modelo_final.fit(df)
    futuro = modelo_final.make_future_dataframe(periods=dias_futuro, freq='D')
    prediccion = modelo_final.predict(futuro)

    # Recopilar metricas de todos los modelos
    metricas = {
        'modelo_ganador': ganador['nombre'],
        'MAPE': ganador['mape'],
        'MAE': ganador['mae'],
        'Precision': max(0, round(100 - ganador['mape'], 2)),
        'prophet_mape': next((r['mape'] for r in resultados_wf if r['nombre'] == 'Prophet'), None),
        'arima_mape': next((r['mape'] for r in resultados_wf if r['nombre'] == 'ARIMA'), None),
        'autoarima_mape': next((r['mape'] for r in resultados_wf if r['nombre'] == 'AutoARIMA'), None),
        'xgboost_mape': next((r['mape'] for r in resultados_wf if r['nombre'] == 'XGBoost'), None),
    }

    return prediccion, metricas


# ============================================
# FUNCIONES DE ANALISIS Y RECOMENDACIONES
# ============================================

def obtener_mejor_dia(df):
    """Detecta el dia de la semana con mayores ventas promedio"""
    dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves',
                    'Viernes', 'Sabado', 'Domingo']
    df_temp = df.copy()
    df_temp['dia'] = df_temp['ds'].dt.dayofweek
    ventas_por_dia = df_temp.groupby('dia')['y'].mean()

    if ventas_por_dia.sum() == 0:
        return "No detectado", None

    mejor_dia_idx = ventas_por_dia.idxmax()
    mejor_dia_nombre = dias_nombres[mejor_dia_idx]
    ventas_promedio = ventas_por_dia[mejor_dia_idx]

    return mejor_dia_nombre, ventas_promedio


def detectar_cambio_tendencia(df, window=14):
    """
    Detecta si hay un cambio reciente en la tendencia.
    Util para alertar al usuario.
    """
    if len(df) < window * 2:
        return {'hay_cambio': False}

    ultimos = df['y'].tail(window)
    anteriores = df['y'].tail(window * 2).head(window)

    if anteriores.mean() == 0:
        return {'hay_cambio': False}

    cambio_pct = (ultimos.mean() - anteriores.mean()) / anteriores.mean() * 100

    if abs(cambio_pct) > 15:
        return {
            'hay_cambio': True,
            'tipo': 'subida' if cambio_pct > 0 else 'bajada',
            'magnitud': abs(cambio_pct)
        }
    return {'hay_cambio': False}


def generar_recomendaciones(df, prediccion, metricas):
    """
    Genera recomendaciones dinamicas basadas en los datos.
    """
    recomendaciones = []

    # 1. Tendencia general (primera mitad vs segunda mitad)
    primera_mitad = df['y'][:len(df)//2].mean()
    segunda_mitad = df['y'][len(df)//2:].mean()

    if primera_mitad > 0:
        if segunda_mitad > primera_mitad * 1.1:
            cambio = (segunda_mitad / primera_mitad - 1) * 100
            recomendaciones.append({
                'tipo': 'positivo',
                'texto': f'Tu negocio crecio {cambio:.1f}% en el ultimo periodo. Considera aumentar tu inventario.'
            })
        elif segunda_mitad < primera_mitad * 0.9:
            cambio = (1 - segunda_mitad / primera_mitad) * 100
            recomendaciones.append({
                'tipo': 'alerta',
                'texto': f'Tus ventas bajaron {cambio:.1f}%. Revisa que cambio en este periodo.'
            })

    # 2. Mejor dia de la semana
    mejor_dia, ventas_dia = obtener_mejor_dia(df)
    if mejor_dia != "No detectado":
        recomendaciones.append({
            'tipo': 'info',
            'texto': f'{mejor_dia} es tu mejor dia (promedio ${ventas_dia:.0f}). Considera promociones otros dias para equilibrar.'
        })

    # 3. Precision del modelo
    if metricas['Precision'] > 85:
        recomendaciones.append({
            'tipo': 'positivo',
            'texto': f'Prediccion muy confiable ({metricas["Precision"]}%). Puedes planificar compras con seguridad.'
        })
    elif metricas['Precision'] > 70:
        recomendaciones.append({
            'tipo': 'info',
            'texto': f'Prediccion confiable ({metricas["Precision"]}%). Usa como guia, manten margen de error.'
        })
    else:
        recomendaciones.append({
            'tipo': 'alerta',
            'texto': f'Prediccion con {metricas["Precision"]}% de confianza. Tus datos son irregulares, usa con precaucion.'
        })

    # 4. Prediccion proxima semana vs ultima semana
    proxima_semana = prediccion[prediccion['ds'] > df['ds'].max()]['yhat'].head(7).sum()
    ultima_semana = df['y'].tail(7).sum()

    if ultima_semana > 0:
        cambio_semana = (proxima_semana - ultima_semana) / ultima_semana * 100
        if cambio_semana > 5:
            recomendaciones.append({
                'tipo': 'positivo',
                'texto': f'Se espera un aumento de {cambio_semana:.1f}% en ventas la proxima semana. Prepara inventario adicional.'
            })
        elif cambio_semana < -5:
            recomendaciones.append({
                'tipo': 'alerta',
                'texto': f'Se espera una baja de {abs(cambio_semana):.1f}% en ventas la proxima semana. Considera promociones.'
            })

    # 5. Cambio de tendencia reciente
    cambio = detectar_cambio_tendencia(df)
    if cambio['hay_cambio']:
        if cambio['tipo'] == 'subida':
            recomendaciones.append({
                'tipo': 'positivo',
                'texto': f'Tendencia fuerte al alza detectada (+{cambio["magnitud"]:.1f}% en 14 dias). Verifica que tu stock pueda cubrir la demanda.'
            })
        else:
            recomendaciones.append({
                'tipo': 'alerta',
                'texto': f'Tendencia a la baja detectada (-{cambio["magnitud"]:.1f}% en 14 dias). Investiga causas: competencia, clima, feriados.'
            })

    # 6. Stock recomendado
    pred_total = prediccion[prediccion['ds'] > df['ds'].max()]['yhat'].sum()
    stock_rec = pred_total * 1.2  # 20% de margen
    recomendaciones.append({
        'tipo': 'info',
        'texto': f'Stock recomendado para proximos {len(prediccion[prediccion["ds"] > df["ds"].max()])} dias: ${stock_rec:,.0f} (incluye 20% margen de seguridad).'
    })

    return recomendaciones


def evaluar_confiabilidad(df, mape):
    """Calcula puntuacion 0-100 de confiabilidad del modelo"""
    dias = (df['ds'].max() - df['ds'].min()).days
    pct_zeros = (df['y'] == 0).sum() / len(df) * 100
    varianza = df['y'].std() / (df['y'].mean() if df['y'].mean() != 0 else 1)

    confianza = 0
    detalles = []

    # Cantidad de datos
    if dias >= 365:
        confianza += 35
        detalles.append("✅ Datos de 1+ ano")
    elif dias >= 90:
        confianza += 25
        detalles.append("✅ Datos de 3+ meses")
    else:
        confianza += 10
        detalles.append("⚠️ Menos de 3 meses de datos")

    # Ceros
    if pct_zeros < 5:
        confianza += 25
        detalles.append("✅ Pocas ventas en cero")
    elif pct_zeros < 20:
        confianza += 12
        detalles.append("⚠️ Algunas ventas en cero")
    else:
        confianza += 0
        detalles.append("🔴 Muchas ventas en cero")

    # Variabilidad
    if 0.3 < varianza < 2:
        confianza += 20
        detalles.append("✅ Variabilidad normal")
    elif varianza == 0:
        confianza += 0
        detalles.append("🔴 Ventas casi iguales")
    else:
        confianza += 10
        detalles.append("⚠️ Variabilidad muy alta")

    # MAPE
    if mape < 10:
        confianza += 20
        detalles.append("✅ Modelo muy preciso")
    elif mape < 20:
        confianza += 15
        detalles.append("✅ Modelo preciso")
    else:
        confianza += 10
        detalles.append("⚠️ Modelo moderado")

    # Nivel
    if confianza >= 85:
        nivel = "🟢 ALTA"
    elif confianza >= 60:
        nivel = "🟡 MEDIA"
    else:
        nivel = "🔴 BAJA"

    return {"score": confianza, "nivel": nivel, "detalles": detalles}

# ============================================
# INTERFAZ STREAMLIT - SALESPREDICT V2
# ============================================

st.set_page_config(
    page_title="SalesPredict v2 - Prediccion Inteligente",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .kpi-card {
        border-radius: 14px;
        padding: 1.3rem 1rem;
        color: white;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
    }
    .kpi-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .kpi-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .kpi-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .kpi-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
    }
    .kpi-label {
        font-size: 0.8rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-delta {
        font-size: 0.75rem;
        margin-top: 0.4rem;
        opacity: 0.85;
    }
    .escenario-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e9ecef;
        border-top: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .escenario-optimista { border-top-color: #28a745; }
    .escenario-esperado { border-top-color: #007bff; }
    .escenario-conservador { border-top-color: #dc3545; }
    .recomendacion-box {
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        font-size: 0.95rem;
    }
    .rec-positivo { background: #e8f5e9; border-color: #28a745; }
    .rec-alerta { background: #ffebee; border-color: #dc3545; }
    .rec-info { background: #e3f2fd; border-color: #2196f3; }
    .confianza-alta { color: #28a745; font-weight: 700; }
    .confianza-media { color: #ff9800; font-weight: 700; }
    .confianza-baja { color: #f44336; font-weight: 700; }
    .modelo-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ganador { background: #d4edda; color: #155724; }
    .badge-normal { background: #f8f9fa; color: #6c757d; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .info-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #dee2e6;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header">📊 SalesPredict v2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediccion inteligente de ventas para tu negocio</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuracion")
    st.divider()

    st.markdown("**🏪 Tu Negocio**")
    nombre_negocio = st.text_input("Nombre:", placeholder="Ej: Mi Tienda", label_visibility="collapsed")

    st.markdown("**🌍 Pais**")
    pais = st.selectbox(
        "Pais:",
        ["Bolivia", "Mexico", "Argentina", "Colombia", "Peru",
         "Chile", "Espana", "USA", "Brasil", "Ecuador",
         "Venezuela", "Paraguay"],
        label_visibility="collapsed"
    )

    st.markdown("**📅 Dias a predecir**")
    dias_futuro = st.slider("Dias:", min_value=7, max_value=90, value=30, step=7, label_visibility="collapsed")

    st.divider()

    st.markdown("**📁 Datos**")
    archivo = st.file_uploader("Sube tu CSV:", type=['csv'], label_visibility="collapsed")

    st.divider()

    # Info de librerias
    st.markdown("**🔧 Estado del sistema**")
    col_lib1, col_lib2 = st.columns(2)
    with col_lib1:
        status_pmd = "🟢" if HAS_PMDARIMA else "🔴"
        st.caption(f"{status_pmd} pmdarima")
    with col_lib2:
        status_xgb = "🟢" if HAS_XGBOOST else "🔴"
        st.caption(f"{status_xgb} xgboost")

    if not HAS_PMDARIMA or not HAS_XGBOOST:
        st.info("""
        💡 **Para mejores resultados instala:**
        ```
        pip install pmdarima xgboost
        ```
        """)

    st.divider()
    st.info("""
    📌 **Requisitos:**
    - Minimo 30 dias de datos
    - Formato fecha: DD/MM/YYYY
    - Columna de ventas numerica
    """)

# CUERPO PRINCIPAL
df_raw = None
col_fecha = None
col_ventas = None

if archivo is not None:
    df_raw = pd.read_csv(archivo, encoding='latin1')
    st.success(f"✅ Archivo cargado: **{len(df_raw)} filas**")

    with st.expander("👁️ Vista previa del CSV"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    st.write("**Selecciona las columnas correctas:**")
    col_a, col_b = st.columns(2)
    with col_a:
        col_fecha = st.selectbox("📅 Fecha:", df_raw.columns.tolist(), key="fecha_select")
    with col_b:
        col_ventas = st.selectbox("💰 Ventas:", df_raw.columns.tolist(), key="ventas_select")
else:
    st.info("👈 **Usa el panel izquierdo para cargar tu CSV y comenzar el analisis**")

st.divider()

# BOTON ANALIZAR
if st.button("🚀 ANALIZAR CON IA", type="primary", use_container_width=True):
    if archivo is None:
        st.error("❌ Primero sube tu archivo CSV")
    elif not nombre_negocio:
        st.error("❌ Escribe el nombre de tu negocio")
    else:
        # STEP 1: LIMPIAR
        with st.spinner("🔄 Limpiando y validando datos..."):
            df_limpio, info_validacion = limpiar_datos(df_raw, col_fecha, col_ventas)

        if info_validacion['estado'] == "ERROR":
            st.error(info_validacion['mensaje'])
            st.stop()
        elif info_validacion['estado'] == "WARNING":
            st.warning(info_validacion['mensaje'])

        st.success(f"✅ Datos limpios: **{len(df_limpio)} registros**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Dias", info_validacion['dias'])
        with col2:
            st.metric("📊 Registros", info_validacion['registros'])
        with col3:
            st.metric("⚠️ Sin ventas", f"{info_validacion['pct_zeros']}%")

        st.divider()

        # STEP 2: ENTRENAR MODELOS
        with st.spinner("🔄 Entrenando modelos (Prophet, ARIMA, AutoARIMA, XGBoost)..."):
            prediccion, metricas = analizar(df_limpio, pais, dias_futuro)

        if prediccion is None:
            st.error("No se pudo entrenar ningun modelo. Verifica tus datos.")
            st.stop()

        recomendaciones = generar_recomendaciones(df_limpio, prediccion, metricas)
        st.success("✅ Modelos entrenados y validados correctamente")

        st.divider()

        # ============================================
        # SECCION: RESULTADOS
        # ============================================

        # CONFIABILIDAD
        confianza = evaluar_confiabilidad(df_limpio, metricas['MAPE'])

        st.markdown('<div class="section-title">📊 Confiabilidad del Analisis</div>', unsafe_allow_html=True)

        conf_col1, conf_col2 = st.columns([3, 1])
        with conf_col1:
            nivel_class = "confianza-alta" if "ALTA" in confianza['nivel'] else ("confianza-media" if "MEDIA" in confianza['nivel'] else "confianza-baja")
            st.markdown(f"**Nivel:** <span class='{nivel_class}'>{confianza['nivel']}</span>", unsafe_allow_html=True)
            for detalle in confianza['detalles']:
                st.caption(detalle)
        with conf_col2:
            st.markdown(f"<div style='text-align:center; font-size:2.5rem; font-weight:800; color:#667eea;'>{confianza['score']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; color:#888; font-size:0.8rem;'>de 100</div>", unsafe_allow_html=True)

        st.divider()

        # KPIs
        st.markdown('<div class="section-title">📈 Indicadores Clave</div>', unsafe_allow_html=True)

        ventas_totales = df_limpio['y'].sum()
        pred_futuras_df = prediccion[prediccion['ds'] > df_limpio['ds'].max()]
        pred_total = pred_futuras_df['yhat'].sum()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                '<div class="kpi-card kpi-green">'
                '<div class="kpi-label">🎯 Precision del Modelo</div>'
                '<div class="kpi-value">' + str(metricas["Precision"]) + '%</div>'
                '<div class="kpi-delta">' + metricas["modelo_ganador"] + '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with kpi2:
            st.markdown(
                '<div class="kpi-card kpi-orange">'
                '<div class="kpi-label">📉 Error (MAPE)</div>'
                '<div class="kpi-value">' + str(metricas["MAPE"]) + '%</div>'
                '<div class="kpi-delta">Walk Forward Validation</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with kpi3:
            st.markdown(
                '<div class="kpi-card kpi-blue">'
                '<div class="kpi-label">💰 Ventas Historicas</div>'
                '<div class="kpi-value">$' + f"{ventas_totales:,.0f}" + '</div>'
                '<div class="kpi-delta">Periodo completo</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                '<div class="kpi-card kpi-purple">'
                '<div class="kpi-label">🔮 Prediccion Total</div>'
                '<div class="kpi-value">$' + f"{pred_total:,.0f}" + '</div>'
                '<div class="kpi-delta">Proximos ' + str(dias_futuro) + ' dias</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ESCENARIOS
        st.markdown('<div class="section-title">🎯 Escenarios de Prediccion</div>', unsafe_allow_html=True)

        esc_optimista = pred_futuras_df['yhat_upper'].sum()
        esc_esperado = pred_futuras_df['yhat'].sum()
        esc_conservador = pred_futuras_df['yhat_lower'].sum()

        esc1, esc2, esc3 = st.columns(3)
        with esc1:
            st.markdown(
                '<div class="escenario-card escenario-optimista">'
                '<div style="font-size: 0.85rem; color: #28a745; font-weight:600;">🟢 OPTIMISTA</div>'
                '<div style="font-size: 1.6rem; font-weight: 800; color: #1a1a2e; margin:0.5rem 0;">$' + f"{esc_optimista:,.0f}" + '</div>'
                '<div style="font-size: 0.8rem; color: #888;">Si todo va bien</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with esc2:
            st.markdown(
                '<div class="escenario-card escenario-esperado">'
                '<div style="font-size: 0.85rem; color: #007bff; font-weight:600;">🔵 ESPERADO</div>'
                '<div style="font-size: 1.6rem; font-weight: 800; color: #1a1a2e; margin:0.5rem 0;">$' + f"{esc_esperado:,.0f}" + '</div>'
                '<div style="font-size: 0.8rem; color: #888;">Escenario mas probable</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with esc3:
            st.markdown(
                '<div class="escenario-card escenario-conservador">'
                '<div style="font-size: 0.85rem; color: #dc3545; font-weight:600;">🔴 CONSERVADOR</div>'
                '<div style="font-size: 1.6rem; font-weight: 800; color: #1a1a2e; margin:0.5rem 0;">$' + f"{esc_conservador:,.0f}" + '</div>'
                '<div style="font-size: 0.8rem; color: #888;">Si algo sale mal</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # COMPARACION DE MODELOS
        st.markdown('<div class="section-title">🤖 Comparacion de Modelos (Walk Forward)</div>', unsafe_allow_html=True)

        comp_cols = st.columns(4)
        modelos_info = [
            ('Prophet', metricas.get('prophet_mape')),
            ('ARIMA', metricas.get('arima_mape')),
            ('AutoARIMA', metricas.get('autoarima_mape')),
            ('XGBoost', metricas.get('xgboost_mape'))
        ]

        for i, (nombre, mape) in enumerate(modelos_info):
            with comp_cols[i]:
                if mape is not None:
                    es_ganador = nombre in metricas['modelo_ganador']
                    badge = '<span class="modelo-badge badge-ganador">GANADOR</span>' if es_ganador else '<span class="modelo-badge badge-normal">-</span>'
                    st.markdown(f"**{nombre}** {badge}", unsafe_allow_html=True)
                    st.metric("MAPE", f"{mape}%")
                else:
                    st.markdown(f"**{nombre}**")
                    st.caption("No disponible")

        # Explicacion MAPE
        with st.expander("ℹ️ Que significa MAPE y Precision?"):
            st.write(f"""
            **MAPE = {metricas['MAPE']}%** (Error Porcentual Medio Absoluto)

            En promedio, el modelo se equivoca **{metricas['MAPE']:.1f}%** en sus predicciones.

            **Escala de calidad:**
            - **< 10%** → 🟢 Excelente
            - **10-20%** → 🟢 Bueno
            - **20-30%** → 🟡 Aceptable
            - **> 30%** → 🔴 Bajo

            **Tu precision:** {metricas['Precision']}%
            """)

        st.divider()

        # RECOMENDACIONES
        st.markdown('<div class="section-title">💡 Recomendaciones para tu Negocio</div>', unsafe_allow_html=True)

        for rec in recomendaciones:
            css_class = f"rec-{rec['tipo']}"
            icono = {'positivo': '✅', 'alerta': '⚠️', 'info': '💡'}.get(rec['tipo'], '💡')
            st.markdown(
                f'<div class="recomendacion-box {css_class}">'
                f'<span style="font-size: 1.3rem; margin-right: 0.5rem;">{icono}</span>'
                f'<strong>{rec["texto"]}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # GRAFICO
        st.markdown('<div class="section-title">📈 Prediccion de Ventas</div>', unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_limpio['ds'], y=df_limpio['y'],
            name='Ventas reales',
            line=dict(color='#2196F3', width=2),
            hovertemplate='Fecha: %{x}<br>Ventas: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=prediccion['ds'], y=prediccion['yhat'],
            name='Prediccion',
            line=dict(color='#FF5722', width=2, dash='dash'),
            hovertemplate='Fecha: %{x}<br>Prediccion: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([prediccion['ds'], prediccion['ds'][::-1]]),
            y=pd.concat([prediccion['yhat_upper'], prediccion['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,87,34,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo 95%',
            hoverinfo='skip'
        ))

        fecha_hoy = df_limpio['ds'].max()
        fig.add_vline(x=fecha_hoy, line=dict(color='green', width=2, dash='dot'))
        fig.add_annotation(x=fecha_hoy, y=1.05, yref='paper', text='Hoy',
                          showarrow=False, font=dict(color='green', size=13))

        fig.update_layout(
            template='plotly_white',
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Fecha",
            yaxis_title="Ventas ($)",
            margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # TABLA DE PROXIMOS DIAS
        st.markdown('<div class="section-title">📅 Proximos Dias Detallados</div>', unsafe_allow_html=True)

        pred_tabla = prediccion[prediccion['ds'] > df_limpio['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(dias_futuro)
        pred_tabla.columns = ['Fecha', 'Prediccion', 'Minimo', 'Maximo']
        pred_tabla['Fecha'] = pred_tabla['Fecha'].dt.strftime('%d/%m/%Y')
        pred_tabla = pred_tabla.round(2)
        pred_tabla['Cambio %'] = pred_tabla['Prediccion'].pct_change() * 100
        pred_tabla['Cambio %'] = pred_tabla['Cambio %'].round(1)

        st.dataframe(pred_tabla, use_container_width=True, hide_index=True)

        # DESCARGAR CSV
        csv = pred_tabla.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar predicciones (CSV)",
            data=csv,
            file_name=f"prediccion_{nombre_negocio.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # FOOTER
        st.markdown(
            f'<div class="footer">'
            f'✅ Analisis completado para <strong>{nombre_negocio}</strong> | '
            f'SalesPredict v2.0 | '
            f'Modelo: {metricas["modelo_ganador"]} | '
            f'Precision: {metricas["Precision"]}%'
            f'</div>',
            unsafe_allow_html=True
        )
