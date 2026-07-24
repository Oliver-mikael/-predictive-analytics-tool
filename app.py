import streamlit as st
import pandas as pd
import numpy as np
import warnings
import holidays
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

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
# CONSTANTES Y CONFIGURACION
# ============================================

MAX_FILAS_CACHE = 500_000  # Limite para datasets grandes
CHUNK_SIZE = 50_000       # Procesar en chunks si excede

# ============================================
# FUNCIONES DE UTILIDAD (CACHE + PERFORMANCE)
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_csv_seguro(archivo_bytes, encoding='latin1'):
    """
    Carga CSV de forma segura y eficiente.
    Para archivos grandes, usa sampling estratificado.
    """
    try:
        # Primero leer solo headers para detectar columnas
        headers = pd.read_csv(archivo_bytes, nrows=0, encoding=encoding).columns.tolist()
        archivo_bytes.seek(0)

        # Contar filas aproximadas
        total_filas = sum(1 for _ in archivo_bytes) - 1
        archivo_bytes.seek(0)

        if total_filas > MAX_FILAS_CACHE:
            st.warning(f"Archivo muy grande ({total_filas:,} filas). Se procesara una muestra representativa.")
            # Cargar con skiprows para muestreo (mantener distribucion temporal)
            skip = max(1, int(total_filas / MAX_FILAS_CACHE))
            df = pd.read_csv(
                archivo_bytes,
                encoding=encoding,
                skiprows=lambda x: x > 0 and x % skip != 0
            )
        else:
            df = pd.read_csv(archivo_bytes, encoding=encoding)

        return df, total_filas
    except Exception as e:
        return None, str(e)


def detectar_columnas_clave(df):
    """
    Detecta automaticamente las columnas relevantes del CSV.
    Soporta multiples formatos de nombres.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Fecha
    candidatos_fecha = ['date', 'fecha', 'ds', 'fecha_venta', 'fecha_transaccion',
                        'fecha venta', 'fecha transaccion', 'transaction_date',
                        'order_date', 'sale_date', 'fecha_pedido']
    col_fecha = None
    for c in candidatos_fecha:
        if c in cols_lower:
            col_fecha = cols_lower[c]
            break

    # Ventas
    candidatos_ventas = ['sales', 'ventas', 'y', 'total', 'monto', 'amount',
                         'total_venta', 'sale_amount', 'precio_total', 'grand_total',
                         'total amount', 'monto total']
    col_ventas = None
    for c in candidatos_ventas:
        if c in cols_lower:
            col_ventas = cols_lower[c]
            break

    # Dimensiones opcionales
    candidatos_branch = ['branch', 'rama', 'sucursal', 'store', 'tienda', 'location']
    col_branch = next((cols_lower[c] for c in candidatos_branch if c in cols_lower), None)

    candidatos_ciudad = ['city', 'ciudad', 'town', 'municipio']
    col_ciudad = next((cols_lower[c] for c in candidatos_ciudad if c in cols_lower), None)

    candidatos_producto = ['product line', 'product_line', 'productline', 'categoria',
                           'category', 'producto', 'product', 'linea', 'line']
    col_producto = next((cols_lower[c] for c in candidatos_producto if c in cols_lower), None)

    candidatos_hora = ['time', 'hora', 'hour', 'transaction_time', 'hora_transaccion']
    col_hora = next((cols_lower[c] for c in candidatos_hora if c in cols_lower), None)

    candidatos_cliente = ['customer type', 'customer_type', 'customertype',
                          'tipo_cliente', 'cliente', 'member', 'membership']
    col_cliente = next((cols_lower[c] for c in candidatos_cliente if c in cols_lower), None)

    candidatos_genero = ['gender', 'genero', 'sex', 'sexo']
    col_genero = next((cols_lower[c] for c in candidatos_genero if c in cols_lower), None)

    candidatos_pago = ['payment', 'metodo_pago', 'metodo pago', 'payment_method',
                       'forma_pago', 'forma pago']
    col_pago = next((cols_lower[c] for c in candidatos_pago if c in cols_lower), None)

    return {
        'fecha': col_fecha,
        'ventas': col_ventas,
        'branch': col_branch,
        'ciudad': col_ciudad,
        'producto': col_producto,
        'hora': col_hora,
        'cliente': col_cliente,
        'genero': col_genero,
        'pago': col_pago
    }

# ============================================
# LIMPIEZA Y FEATURE ENGINEERING MEJORADO
# ============================================

def limpiar_datos_v3(df, col_fecha, col_ventas):
    """
    Limpia datos de forma robusta. Maneja grandes volumenes.
    Retorna: (df_limpio, info_validacion, df_original_con_features)
    """
    df_proc = pd.DataFrame()
    df_proc['ds'] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
    df_proc['y'] = pd.to_numeric(df[col_ventas], errors='coerce')

    # Eliminar nulos y negativos
    df_proc = df_proc.dropna()
    df_proc = df_proc[df_proc['y'] >= 0]
    df_proc = df_proc.sort_values('ds')

    # AGREGAR por fecha (suma diaria)
    df_diario = df_proc.groupby('ds', as_index=False)['y'].sum()

    # Rellenar dias faltantes
    rango = pd.DataFrame({
        'ds': pd.date_range(start=df_diario['ds'].min(),
                            end=df_diario['ds'].max(), freq='D')
    })
    df_diario = rango.merge(df_diario, on='ds', how='left')
    df_diario['y'] = df_diario['y'].fillna(0)

    # Outliers: clip al percentil 99.5 (mas permisivo que IQR*3)
    p99 = df_diario['y'].quantile(0.995)
    df_diario['y'] = df_diario['y'].clip(upper=p99)

    # Validaciones
    dias = (df_diario['ds'].max() - df_diario['ds'].min()).days
    registros = len(df_diario)
    pct_zeros = (df_diario['y'] == 0).sum() / len(df_diario) * 100

    if dias < 30:
        estado, mensaje = "ERROR", f"Solo {dias} dias. Minimo 30."
    elif pct_zeros > 50:
        estado, mensaje = "ERROR", f"{pct_zeros:.1f}% ceros. Datos muy fragmentados."
    elif pct_zeros > 25:
        estado, mensaje = "WARNING", f"{pct_zeros:.1f}% ceros. Precision afectada."
    elif dias < 90:
        estado, mensaje = "WARNING", f"Solo {dias} dias. Recomendado 90+ para precision."
    else:
        estado, mensaje = "OK", "Datos validos."

    info = {
        'dias': dias,
        'registros': registros,
        'pct_zeros': round(pct_zeros, 2),
        'estado': estado,
        'mensaje': mensaje,
        'fecha_min': df_diario['ds'].min(),
        'fecha_max': df_diario['ds'].max(),
        'venta_promedio': df_diario['y'].mean(),
        'venta_std': df_diario['y'].std()
    }

    return df_diario, info


def crear_features_v3(df, usar_log=False):
    """
    Feature engineering optimizado para supermercados.
    Incluye quincena, fin de mes, y patrones de pago (critico en LATAM).
    """
    df = df.copy()
    df = df.sort_values('ds').reset_index(drop=True)

    # Transformacion logaritmica opcional (para alta varianza)
    if usar_log:
        df['y_raw'] = df['y'].copy()
        df['y'] = np.log1p(df['y'])

    # === CALENDARIO ===
    df['dia_semana'] = df['ds'].dt.dayofweek          # 0=Lunes
    df['es_finde'] = (df['ds'].dt.dayofweek >= 5).astype(int)
    df['dia_mes'] = df['ds'].dt.day
    df['mes'] = df['ds'].dt.month
    df['trimestre'] = df['ds'].dt.quarter
    df['semana_ano'] = df['ds'].dt.isocalendar().week.astype(int)

    # === QUINCENA / DIA DE PAGO (critico en LATAM) ===
    # Quincena: dias 14-16 y 29-31 (ajustado por mes)
    df['es_quincena'] = ((df['dia_mes'] >= 14) & (df['dia_mes'] <= 16)).astype(int)
    df['es_fin_mes'] = df['dia_mes'].isin([28, 29, 30, 31]).astype(int)
    df['es_inicio_mes'] = (df['dia_mes'] <= 3).astype(int)

    # === LAGS (patrones ciclicos) ===
    for lag in [1, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # === MEDIAS MOVILES ===
    df['ma_7'] = df['y'].shift(1).rolling(7, min_periods=1).mean()
    df['ma_14'] = df['y'].shift(1).rolling(14, min_periods=1).mean()
    df['ma_30'] = df['y'].shift(1).rolling(30, min_periods=1).mean()

    # === TENDENCIA ===
    df['diff_7'] = df['y'].shift(1).diff(7)
    df['diff_14'] = df['y'].shift(1).diff(14)

    # === RATIOS ===
    df['ratio_ma7'] = df['y'].shift(1) / (df['ma_7'] + 1)
    df['ratio_ma30'] = df['y'].shift(1) / (df['ma_30'] + 1)

    # === FEATURES CICLICAS (seno/coseno para dia semana y mes) ===
    # Esto ayuda a que el modelo entienda que Domingo(6) esta cerca de Lunes(0)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Eliminar NaN (primeras 28 filas no tienen lag_28)
    df = df.dropna().reset_index(drop=True)

    return df


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
        clase = paises_map.get(pais)
        if clase:
            lista = []
            for a in anos:
                f = clase(years=a)
                for fecha, nombre in f.items():
                    lista.append({'holiday': nombre, 'ds': pd.Timestamp(fecha)})
            return pd.DataFrame(lista)
    except:
        pass
    return None

# ============================================
# MODELOS DE PREDICCION MEJORADOS
# ============================================

def calcular_mape(real, pred):
    """MAPE robusto"""
    mask = real > (real.mean() * 0.05)
    if mask.sum() > 0:
        return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100
    return np.mean(np.abs((real - pred) / (real + 1))) * 100


def correr_prophet_mejorado(df_train, df_test, feriados=None, usar_regressors=True):
    """
    Prophet mejorado con:
    - changepoint_prior_scale alto (datos irregulares)
    - seasonality_mode multiplicativo
    - Regressors: fin de semana, quincena, fin de mes
    """
    try:
        modelo = Prophet(
            weekly_seasonality=len(df_train) > 30,
            yearly_seasonality=len(df_train) > 365,
            daily_seasonality=False,
            interval_width=0.95,
            holidays=feriados,
            changepoint_prior_scale=0.5,      # MAS flexible para cambios bruscos
            seasonality_mode='multiplicative', # Mejor para datos con crecimiento
            changepoint_range=0.95             # Detectar cambios hasta el 95% de datos
        )

        # Agregar regressors si hay suficientes datos
        if usar_regressors and len(df_train) > 60:
            # Crear features en train para usar como regressors
            df_train_f = crear_features_v3(df_train.copy(), usar_log=False)

            # Solo usar features que existen en train y se pueden proyectar a test
            regressors = ['es_finde', 'es_quincena', 'es_fin_mes']
            for r in regressors:
                if r in df_train_f.columns:
                    modelo.add_regressor(r)

            # Merge regressors al train original
            df_train_reg = df_train.merge(
                df_train_f[['ds'] + regressors], on='ds', how='left'
            )
            df_train_reg[regressors] = df_train_reg[regressors].fillna(0)
        else:
            df_train_reg = df_train.copy()
            regressors = []

        modelo.fit(df_train_reg)

        # Preparar futuro con regressors para test
        futuro = modelo.make_future_dataframe(periods=len(df_test), freq='D')

        if regressors:
            df_test_f = crear_features_v3(
                pd.concat([df_train, df_test]).reset_index(drop=True),
                usar_log=False
            )
            df_test_f = df_test_f[['ds'] + regressors]
            futuro = futuro.merge(df_test_f, on='ds', how='left')
            futuro[regressors] = futuro[regressors].fillna(0)

        pred = modelo.predict(futuro)
        pred_test = pred['yhat'].tail(len(df_test)).values
        real_test = df_test['y'].values

        mape = calcular_mape(real_test, pred_test)
        mae = mean_absolute_error(real_test, pred_test)

        return {'nombre': 'Prophet+', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': modelo}
    except Exception as e:
        return {'nombre': 'Prophet+', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_arima(df_train, df_test):
    """ARIMA(1,1,1) fallback siempre disponible"""
    try:
        modelo = ARIMA(df_train['y'], order=(1, 1, 1))
        res = modelo.fit()
        pred = res.forecast(steps=len(df_test))
        mape = calcular_mape(df_test['y'].values, pred.values)
        mae = mean_absolute_error(df_test['y'], pred)
        return {'nombre': 'ARIMA', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': res}
    except Exception as e:
        return {'nombre': 'ARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_autoarima(df_train, df_test):
    if not HAS_PMDARIMA:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999,
                'error': 'Instala: pip install pmdarima'}
    try:
        modelo = auto_arima(
            df_train['y'],
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            max_p=5, max_d=2, max_q=5,
            n_jobs=1, trace=False
        )
        pred = modelo.predict(n_periods=len(df_test))
        mape = calcular_mape(df_test['y'].values, pred)
        mae = mean_absolute_error(df_test['y'], pred)
        return {'nombre': 'AutoARIMA', 'mape': round(mape, 2),
                'mae': round(mae, 2), 'modelo': modelo}
    except Exception as e:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_xgboost_v3(df_train, df_test):
    """XGBoost con features v3 (incluye quincena, ciclicas, etc.)"""
    if not HAS_XGBOOST:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999,
                'error': 'Instala: pip install xgboost'}
    try:
        df_train_f = crear_features_v3(df_train.copy())
        df_combined = pd.concat([df_train, df_test]).reset_index(drop=True)
        df_test_f = crear_features_v3(df_combined)
        df_test_f = df_test_f.tail(len(df_test)).reset_index(drop=True)

        feature_cols = [c for c in df_train_f.columns
                        if c not in ['ds', 'y', 'y_raw']]

        X_train = df_train_f[feature_cols]
        y_train = df_train_f['y']
        X_test = df_test_f[feature_cols]
        y_test = df_test_f['y']

        modelo = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='mae'
        )
        modelo.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        pred = modelo.predict(X_test)
        mape = calcular_mape(y_test.values, pred)
        mae = mean_absolute_error(y_test, pred)

        # Feature importance
        importance = dict(zip(feature_cols, modelo.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'nombre': 'XGBoost',
            'mape': round(mape, 2),
            'mae': round(mae, 2),
            'modelo': modelo,
            'feature_cols': feature_cols,
            'top_features': top_features
        }
    except Exception as e:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999, 'error': str(e)}

# ============================================
# VALIDACION WALK FORWARD + ENSEMBLE
# ============================================

def walk_forward_validation(df, pais, window_test=30):
    """Valida en los ultimos N dias (mas realista que split aleatorio)"""
    fecha_corte = df['ds'].max() - pd.Timedelta(days=window_test)

    if (df['ds'].max() - fecha_corte).days < 14:
        split = int(len(df) * 0.8)
        df_train = df[:split]
        df_test = df[split:]
    else:
        df_train = df[df['ds'] <= fecha_corte].copy()
        df_test = df[df['ds'] > fecha_corte].copy()

    if len(df_train) < 30:
        return None, "Pocos datos para entrenar (< 30 dias)"

    anos = df['ds'].dt.year.unique().tolist()
    anos += [max(anos) + 1]
    feriados = obtener_feriados(pais, anos)

    # Ejecutar modelos
    res_prophet = correr_prophet_mejorado(df_train, df_test, feriados)
    res_arima = correr_arima(df_train, df_test)
    res_autoarima = correr_autoarima(df_train, df_test)
    res_xgb = correr_xgboost_v3(df_train, df_test)

    resultados = [res_prophet, res_arima, res_autoarima, res_xgb]
    validos = [r for r in resultados if r['mape'] < 100]

    if not validos:
        return None, "Ningun modelo convergio"

    validos.sort(key=lambda x: x['mape'])
    return validos, None


def ensemble_ponderado(resultados):
    """Combina 2-3 mejores modelos con pesos inversos al MAPE"""
    buenos = [r for r in resultados if r['mape'] < 40]
    if len(buenos) == 0:
        buenos = [resultados[0]]
    if len(buenos) == 1:
        return buenos[0]

    buenos = buenos[:3]
    pesos = [1 / max(r['mape'], 0.5) for r in buenos]
    total = sum(pesos)
    pesos = [p / total for p in pesos]

    mape_e = sum(r['mape'] * p for r, p in zip(buenos, pesos))
    mae_e = sum(r['mae'] * p for r, p in zip(buenos, pesos))
    nombres = '+'.join([r['nombre'] for r in buenos])

    return {
        'nombre': f'Ensemble ({nombres})',
        'mape': round(mape_e, 2),
        'mae': round(mae_e, 2),
        'modelos': buenos,
        'pesos': pesos
    }


def analizar_v3(df, pais, dias_futuro):
    """Pipeline completo"""
    resultados, error = walk_forward_validation(df, pais, window_test=30)
    if error:
        return None, None, error

    ganador = ensemble_ponderado(resultados)

    # Modelo final: Prophet entrenado con TODOS los datos
    anos = df['ds'].dt.year.unique().tolist()
    anos += [max(anos) + 1]
    feriados = obtener_feriados(pais, anos)

    modelo_final = Prophet(
        weekly_seasonality=len(df) > 30,
        yearly_seasonality=len(df) > 365,
        daily_seasonality=False,
        interval_width=0.95,
        holidays=feriados,
        changepoint_prior_scale=0.5,
        seasonality_mode='multiplicative',
        changepoint_range=0.95
    )

    # Agregar regressors si hay suficientes datos
    if len(df) > 60:
        df_f = crear_features_v3(df.copy(), usar_log=False)
        regressors = ['es_finde', 'es_quincena', 'es_fin_mes']
        for r in regressors:
            if r in df_f.columns:
                modelo_final.add_regressor(r)
        df_reg = df.merge(df_f[['ds'] + regressors], on='ds', how='left')
        df_reg[regressors] = df_reg[regressors].fillna(0)
    else:
        df_reg = df.copy()

    modelo_final.fit(df_reg)

    futuro = modelo_final.make_future_dataframe(periods=dias_futuro, freq='D')
    if len(df) > 60:
        # Proyectar regressors al futuro
        futuro['dia_semana'] = futuro['ds'].dt.dayofweek
        futuro['es_finde'] = (futuro['dia_semana'] >= 5).astype(int)
        futuro['dia_mes'] = futuro['ds'].dt.day
        futuro['es_quincena'] = ((futuro['dia_mes'] >= 14) & (futuro['dia_mes'] <= 16)).astype(int)
        futuro['es_fin_mes'] = futuro['dia_mes'].isin([28, 29, 30, 31]).astype(int)
        futuro = futuro.drop(columns=['dia_semana', 'dia_mes'])

    prediccion = modelo_final.predict(futuro)

    metricas = {
        'modelo_ganador': ganador['nombre'],
        'MAPE': ganador['mape'],
        'MAE': ganador['mae'],
        'Precision': max(0, round(100 - ganador['mape'], 2)),
        'prophet_mape': next((r['mape'] for r in resultados if 'Prophet' in r['nombre']), None),
        'arima_mape': next((r['mape'] for r in resultados if r['nombre'] == 'ARIMA'), None),
        'autoarima_mape': next((r['mape'] for r in resultados if r['nombre'] == 'AutoARIMA'), None),
        'xgboost_mape': next((r['mape'] for r in resultados if r['nombre'] == 'XGBoost'), None),
    }

    return prediccion, metricas, None


# ============================================
# ANALISIS Y RECOMENDACIONES
# ============================================

def obtener_mejor_dia(df):
    dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves',
                    'Viernes', 'Sabado', 'Domingo']
    df_temp = df.copy()
    df_temp['dia'] = df_temp['ds'].dt.dayofweek
    ventas = df_temp.groupby('dia')['y'].mean()
    if ventas.sum() == 0:
        return "No detectado", 0
    idx = ventas.idxmax()
    return dias_nombres[idx], ventas[idx]


def detectar_cambio_tendencia(df, window=14):
    if len(df) < window * 2:
        return {'hay_cambio': False}
    ultimos = df['y'].tail(window)
    anteriores = df['y'].tail(window * 2).head(window)
    if anteriores.mean() == 0:
        return {'hay_cambio': False}
    cambio = (ultimos.mean() - anteriores.mean()) / anteriores.mean() * 100
    if abs(cambio) > 15:
        return {'hay_cambio': True, 'tipo': 'subida' if cambio > 0 else 'bajada',
                'magnitud': abs(cambio)}
    return {'hay_cambio': False}


def generar_recomendaciones_v3(df, prediccion, metricas, info):
    recs = []

    # 1. Tendencia general
    primera = df['y'][:len(df)//2].mean()
    segunda = df['y'][len(df)//2:].mean()
    if primera > 0:
        if segunda > primera * 1.1:
            recs.append({'tipo': 'positivo',
                'texto': f'Crecimiento de {((segunda/primera-1)*100):.1f}% en el ultimo periodo. Aumenta inventario.'})
        elif segunda < primera * 0.9:
            recs.append({'tipo': 'alerta',
                'texto': f'Baja de {((1-segunda/primera)*100):.1f}% en ventas. Investiga causas.'})

    # 2. Mejor dia
    mejor, promedio = obtener_mejor_dia(df)
    if mejor != "No detectado":
        recs.append({'tipo': 'info',
            'texto': f'{mejor} es tu mejor dia (promedio ${promedio:.0f}). Promociones en dias debiles.'})

    # 3. Precision
    if metricas['Precision'] > 85:
        recs.append({'tipo': 'positivo',
            'texto': f'Modelo confiable ({metricas["Precision"]}%). Planifica con seguridad.'})
    elif metricas['Precision'] > 70:
        recs.append({'tipo': 'info',
            'texto': f'Modelo aceptable ({metricas["Precision"]}%). Usa como guia con margen.'})
    else:
        recs.append({'tipo': 'alerta',
            'texto': f'Precision baja ({metricas["Precision"]}%). Datos irregulares, usa con precaucion.'})

    # 4. Proxima semana
    prox = prediccion[prediccion['ds'] > df['ds'].max()]['yhat'].head(7).sum()
    ult = df['y'].tail(7).sum()
    if ult > 0:
        cambio = (prox - ult) / ult * 100
        if cambio > 5:
            recs.append({'tipo': 'positivo',
                'texto': f'Aumento esperado de {cambio:.1f}% proxima semana. Prepara stock.'})
        elif cambio < -5:
            recs.append({'tipo': 'alerta',
                'texto': f'Baja esperada de {abs(cambio):.1f}% proxima semana. Considera promociones.'})

    # 5. Cambio de tendencia
    cambio = detectar_cambio_tendencia(df)
    if cambio['hay_cambio']:
        icon = 'subida' if cambio['tipo'] == 'subida' else 'bajada'
        recs.append({'tipo': 'positivo' if cambio['tipo'] == 'subida' else 'alerta',
            'texto': f'Tendencia {icon} reciente ({cambio["magnitud"]:.1f}% en 14 dias). Ajusta estrategia.'})

    # 6. Stock recomendado
    pred_total = prediccion[prediccion['ds'] > df['ds'].max()]['yhat'].sum()
    stock = pred_total * 1.2
    recs.append({'tipo': 'info',
        'texto': f'Stock recomendado: ${stock:,.0f} (incluye 20% margen de seguridad).'})

    # 7. Dias de datos
    if info['dias'] < 90:
        recs.append({'tipo': 'alerta',
            'texto': f'Solo {info["dias"]} dias de datos. Con 6+ meses la precision mejorara significativamente.'})

    return recs


def evaluar_confiabilidad(df, mape):
    dias = (df['ds'].max() - df['ds'].min()).days
    pct_zeros = (df['y'] == 0).sum() / len(df) * 100
    varianza = df['y'].std() / (df['y'].mean() if df['y'].mean() != 0 else 1)

    confianza = 0
    detalles = []

    if dias >= 365:
        confianza += 35; detalles.append("✅ 1+ ano de datos")
    elif dias >= 180:
        confianza += 30; detalles.append("✅ 6+ meses de datos")
    elif dias >= 90:
        confianza += 25; detalles.append("✅ 3+ meses de datos")
    else:
        confianza += 10; detalles.append("⚠️ < 3 meses de datos")

    if pct_zeros < 5:
        confianza += 25; detalles.append("✅ Pocos dias sin ventas")
    elif pct_zeros < 20:
        confianza += 12; detalles.append("⚠️ Algunos dias sin ventas")
    else:
        confianza += 0; detalles.append("🔴 Muchos dias sin ventas")

    if 0.3 < varianza < 2:
        confianza += 20; detalles.append("✅ Variabilidad normal")
    elif varianza == 0:
        confianza += 0; detalles.append("🔴 Ventas constantes")
    else:
        confianza += 10; detalles.append("⚠️ Variabilidad alta")

    if mape < 10:
        confianza += 20; detalles.append("✅ Modelo muy preciso")
    elif mape < 20:
        confianza += 15; detalles.append("✅ Modelo preciso")
    elif mape < 30:
        confianza += 10; detalles.append("⚠️ Modelo aceptable")
    else:
        confianza += 5; detalles.append("🔴 Modelo con baja precision")

    nivel = "🟢 ALTA" if confianza >= 85 else ("🟡 MEDIA" if confianza >= 60 else "🔴 BAJA")
    return {"score": confianza, "nivel": nivel, "detalles": detalles}

# ============================================
# 6 GRAFICOS DE ANALISIS POR DIMENSION
# ============================================

def grafico_ventas_rama(df_raw, col_branch, col_ventas):
    if col_branch is None:
        return None
    ventas = df_raw.groupby(col_branch)[col_ventas].sum().reset_index()
    ventas = ventas.sort_values(col_ventas, ascending=True)
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#38ef7d', '#11998e']
    fig = go.Figure(data=[go.Bar(
        x=ventas[col_ventas], y=ventas[col_branch], orientation='h',
        marker=dict(color=colors[:len(ventas)], line=dict(color='white', width=1)),
        hovertemplate='<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>'
    )])
    fig.update_layout(title="Ventas por Rama/Sucursal", xaxis_title="Ventas ($)",
                      yaxis_title="", template="plotly_white", height=350,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def grafico_ventas_ciudad(df_raw, col_ciudad, col_ventas):
    if col_ciudad is None:
        return None
    ventas = df_raw.groupby(col_ciudad)[col_ventas].sum().reset_index()
    colors = ['#11998e', '#38ef7d', '#f5576c', '#667eea', '#764ba2', '#f093fb']
    fig = go.Figure(data=[go.Pie(
        labels=ventas[col_ciudad], values=ventas[col_ventas],
        marker=dict(colors=colors[:len(ventas)], line=dict(color='white', width=2)),
        hovertemplate='<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>'
    )])
    fig.update_layout(title="Distribucion por Ciudad", template="plotly_white",
                      height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def grafico_ventas_producto(df_raw, col_producto, col_ventas):
    if col_producto is None:
        return None
    ventas = df_raw.groupby(col_producto)[col_ventas].sum().reset_index()
    ventas = ventas.sort_values(col_ventas, ascending=True)
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#38ef7d', '#11998e',
              '#4facfe', '#00f2fe', '#f6d365', '#fda085']
    fig = go.Figure(data=[go.Bar(
        x=ventas[col_ventas], y=ventas[col_producto], orientation='h',
        marker=dict(color=colors[:len(ventas)], line=dict(color='white', width=1)),
        hovertemplate='<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>'
    )])
    fig.update_layout(title="Ventas por Linea de Producto", xaxis_title="Ventas ($)",
                      yaxis_title="", template="plotly_white", height=450,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def grafico_patron_dia_semana(df_raw, col_fecha, col_ventas):
    df_copy = df_raw.copy()
    df_copy['fecha_dt'] = pd.to_datetime(df_copy[col_fecha], dayfirst=True, errors='coerce')
    df_copy['dia_nombre'] = df_copy['fecha_dt'].dt.day_name()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dias_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miercoles',
               'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sabado', 'Sunday': 'Domingo'}
    ventas = df_copy.groupby('dia_nombre')[col_ventas].agg(['sum', 'mean', 'count']).reindex(dias_orden)
    ventas.index = [dias_es.get(d, d) for d in ventas.index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ventas.index, y=ventas['sum'], mode='lines+markers',
        name='Total', line=dict(color='#667eea', width=3), marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Total: $%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=ventas.index, y=ventas['mean'], mode='lines+markers',
        name='Promedio/trans', line=dict(color='#f5576c', width=2, dash='dash'),
        marker=dict(size=8), hovertemplate='<b>%{x}</b><br>Promedio: $%{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Patron por Dia de Semana", xaxis_title="Dia",
                      yaxis_title="Ventas ($)", template="plotly_white", hovermode='x unified',
                      height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def grafico_ventas_por_hora(df_raw, col_hora, col_ventas):
    if col_hora is None:
        return None
    df_copy = df_raw.copy()
    df_copy['hora_dt'] = pd.to_datetime(df_copy[col_hora], format='%H:%M:%S', errors='coerce')
    df_copy['hora'] = df_copy['hora_dt'].dt.hour
    ventas = df_copy.groupby('hora')[col_ventas].agg(['sum', 'count']).reset_index()

    fig = go.Figure(data=[go.Bar(
        x=ventas['hora'], y=ventas['sum'],
        marker=dict(color=ventas['sum'], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Ventas")),
        hovertemplate='<b>%{x}:00</b><br>Ventas: $%{y:,.0f}<br>Trans: %{customdata}<extra></extra>',
        customdata=ventas['count']
    )])
    fig.update_layout(title="Ventas por Hora del Dia", xaxis_title="Hora",
                      yaxis_title="Ventas ($)", template="plotly_white", height=400,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def grafico_heatmap_rama_ciudad(df_raw, col_branch, col_ciudad, col_ventas):
    if col_branch is None or col_ciudad is None:
        return None
    pivot = df_raw.pivot_table(values=col_ventas, index=col_branch, columns=col_ciudad, aggfunc='sum')
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale='Viridis', hovertemplate='<b>%{y} - %{x}</b><br>$%{z:,.0f}<extra></extra>',
        colorbar=dict(title="Ventas ($)")
    ))
    fig.update_layout(title="Heatmap: Rama vs Ciudad", xaxis_title="Ciudad",
                      yaxis_title="Rama", template="plotly_white", height=400,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def tabla_comparacion_periodos(df_raw, col_fecha, col_ventas):
    df_copy = df_raw.copy()
    df_copy['fecha_dt'] = pd.to_datetime(df_copy[col_fecha], dayfirst=True, errors='coerce')
    fecha_corte = df_copy['fecha_dt'].min() + pd.Timedelta(days=(df_copy['fecha_dt'].max() - df_copy['fecha_dt'].min()).days // 2)

    p1 = df_copy[df_copy['fecha_dt'] < fecha_corte]
    p2 = df_copy[df_copy['fecha_dt'] >= fecha_corte]

    datos = {
        'Metrica': ['Ventas Totales', 'Transacciones', 'Promedio', 'Maximo', 'Minimo', 'Dias unicos'],
        'Periodo 1': [
            f"${p1[col_ventas].sum():,.0f}", f"{len(p1):,}",
            f"${p1[col_ventas].mean():.0f}", f"${p1[col_ventas].max():.0f}",
            f"${p1[col_ventas].min():.0f}", f"{p1['fecha_dt'].nunique()}"
        ],
        'Periodo 2': [
            f"${p2[col_ventas].sum():,.0f}", f"{len(p2):,}",
            f"${p2[col_ventas].mean():.0f}", f"${p2[col_ventas].max():.0f}",
            f"${p2[col_ventas].min():.0f}", f"{p2['fecha_dt'].nunique()}"
        ]
    }
    return pd.DataFrame(datos)

# ============================================
# INTERFAZ STREAMLIT V3 - SALESPREDICT PRO
# ============================================

st.set_page_config(
    page_title="SalesPredict v3 - Prediccion Inteligente",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1rem; color: #888; margin-bottom: 2rem; }
    .kpi-card { border-radius: 14px; padding: 1.3rem 1rem; color: white; box-shadow: 0 6px 20px rgba(0,0,0,0.12); text-align: center; }
    .kpi-purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .kpi-blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .kpi-value { font-size: 2rem; font-weight: 800; }
    .kpi-label { font-size: 0.8rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-delta { font-size: 0.75rem; margin-top: 0.4rem; opacity: 0.85; }
    .escenario-card { background: #fff; border-radius: 12px; padding: 1.2rem; text-align: center; border: 1px solid #e9ecef; border-top: 4px solid; }
    .esc-optimista { border-top-color: #28a745; } .esc-esperado { border-top-color: #007bff; } .esc-conservador { border-top-color: #dc3545; }
    .rec-box { border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.5rem 0; border-left: 4px solid; font-size: 0.95rem; }
    .rec-pos { background: #e8f5e9; border-color: #28a745; } .rec-alt { background: #ffebee; border-color: #dc3545; } .rec-inf { background: #e3f2fd; border-color: #2196f3; }
    .conf-alta { color: #28a745; font-weight: 700; } .conf-media { color: #ff9800; font-weight: 700; } .conf-baja { color: #f44336; font-weight: 700; }
    .badge-gan { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .badge-norm { background: #f8f9fa; color: #6c757d; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; }
    .section-title { font-size: 1.2rem; font-weight: 700; color: #1a1a2e; margin: 1.5rem 0 1rem 0; padding-bottom: 0.4rem; border-bottom: 2px solid #e9ecef; }
    .footer { text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header">📊 SalesPredict v3</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediccion inteligente de ventas para retail y supermercados</div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Configuracion")
    st.divider()

    st.markdown("**🏪 Negocio**")
    nombre_negocio = st.text_input("Nombre:", placeholder="Ej: Supermercado Central", label_visibility="collapsed")

    st.markdown("**🌍 Pais**")
    pais = st.selectbox("Pais:", ["Bolivia", "Mexico", "Argentina", "Colombia", "Peru", "Chile",
         "Espana", "USA", "Brasil", "Ecuador", "Venezuela", "Paraguay"], label_visibility="collapsed")

    st.markdown("**📅 Prediccion**")
    dias_futuro = st.slider("Dias:", 7, 90, 30, 7, label_visibility="collapsed")

    st.markdown("**🔧 Opciones avanzadas**")
    usar_log = st.toggle("Transformacion log (para alta varianza)", value=False,
                         help="Activa si tus ventas varian mucho (ej: $1000-$10000)")

    st.divider()
    st.markdown("**📁 Datos**")
    archivo = st.file_uploader("CSV:", type=['csv'], label_visibility="collapsed")

    st.divider()
    st.markdown("**🔌 Librerias**")
    c1, c2 = st.columns(2)
    with c1: st.caption(f"{'🟢' if HAS_PMDARIMA else '🔴'} pmdarima")
    with c2: st.caption(f"{'🟢' if HAS_XGBOOST else '🔴'} xgboost")
    if not HAS_PMDARIMA or not HAS_XGBOOST:
        st.info("Mejora resultados:\n`pip install pmdarima xgboost`")

    st.divider()
    st.info("Requisitos: Minimo 30 dias | Formato fecha: DD/MM/YYYY")

# CUERPO
if archivo is not None:
    df_raw, info_carga = cargar_csv_seguro(archivo)

    if df_raw is None:
        st.error(f"Error cargando CSV: {info_carga}")
        st.stop()

    st.success(f"✅ CSV cargado: **{len(df_raw):,} filas**")

    # Detectar columnas
    columnas = detectar_columnas_clave(df_raw)

    with st.expander("👁️ Vista previa y columnas detectadas"):
        st.dataframe(df_raw.head(5), use_container_width=True)
        st.caption(f"Columnas detectadas: { {k:v for k,v in columnas.items() if v} }")

    # Selectores de columnas (con deteccion automatica como default)
    st.write("**Confirma las columnas:**")
    c1, c2 = st.columns(2)
    with c1:
        col_fecha_sel = st.selectbox("📅 Fecha:", df_raw.columns.tolist(),
                                     index=df_raw.columns.tolist().index(columnas['fecha']) if columnas['fecha'] in df_raw.columns else 0)
    with c2:
        col_ventas_sel = st.selectbox("💰 Ventas:", df_raw.columns.tolist(),
                                      index=df_raw.columns.tolist().index(columnas['ventas']) if columnas['ventas'] in df_raw.columns else 0)
else:
    st.info("👈 **Carga tu CSV en el panel izquierdo para comenzar**")
    st.stop()

st.divider()

# BOTON ANALIZAR
if st.button("🚀 ANALIZAR CON IA", type="primary", use_container_width=True):
    if not nombre_negocio:
        st.error("Escribe el nombre de tu negocio")
        st.stop()

    # PASO 1: LIMPIAR
    with st.spinner("Limpiando y preparando datos..."):
        df_limpio, info_val = limpiar_datos_v3(df_raw, col_fecha_sel, col_ventas_sel)

    if info_val['estado'] == "ERROR":
        st.error(info_val['mensaje']); st.stop()
    elif info_val['estado'] == "WARNING":
        st.warning(info_val['mensaje'])

    st.success(f"✅ **{info_val['registros']} dias** | Promedio: ${info_val['venta_promedio']:.0f}/dia | Std: ${info_val['venta_std']:.0f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Dias", info_val['dias'])
    c2.metric("📊 Registros", info_val['registros'])
    c3.metric("⚠️ Sin ventas", f"{info_val['pct_zeros']}%")

    st.divider()

    # PASO 2: ENTRENAR
    with st.spinner("Entrenando Prophet+, ARIMA, AutoARIMA, XGBoost..."):
        prediccion, metricas, error = analizar_v3(df_limpio, pais, dias_futuro)

    if error:
        st.error(f"Error: {error}"); st.stop()

    recomendaciones = generar_recomendaciones_v3(df_limpio, prediccion, metricas, info_val)
    st.success(f"✅ Ensemble: **{metricas['modelo_ganador']}** | MAPE: **{metricas['MAPE']}%** | Precision: **{metricas['Precision']}%**")

    # ============================================
    # TABS: Dashboard + Analisis Detallado
    # ============================================
    tab1, tab2 = st.tabs(["📊 Dashboard Principal", "📈 Analisis Detallado"])

    with tab1:
        # CONFIABILIDAD
        confianza = evaluar_confiabilidad(df_limpio, metricas['MAPE'])
        st.markdown('<div class="section-title">Confiabilidad del Analisis</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns([3, 1])
        with cc1:
            cls = "conf-alta" if "ALTA" in confianza['nivel'] else ("conf-media" if "MEDIA" in confianza['nivel'] else "conf-baja")
            st.markdown(f"**Nivel:** <span class='{cls}'>{confianza['nivel']}</span>", unsafe_allow_html=True)
            for d in confianza['detalles']: st.caption(d)
        with cc2:
            st.markdown(f"<div style='text-align:center;font-size:2.5rem;font-weight:800;color:#667eea;'>{confianza['score']}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;color:#888;font-size:0.8rem;'>de 100</div>", unsafe_allow_html=True)

        # KPIs
        st.markdown('<div class="section-title">Indicadores Clave</div>', unsafe_allow_html=True)
        ventas_total = df_limpio['y'].sum()
        pred_fut = prediccion[prediccion['ds'] > df_limpio['ds'].max()]
        pred_total = pred_fut['yhat'].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-label">Precision</div><div class="kpi-value">{metricas["Precision"]}%</div><div class="kpi-delta">{metricas["modelo_ganador"]}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-label">Error MAPE</div><div class="kpi-value">{metricas["MAPE"]}%</div><div class="kpi-delta">Walk Forward</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-label">Ventas Historicas</div><div class="kpi-value">${ventas_total:,.0f}</div><div class="kpi-delta">{info_val["dias"]} dias</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-label">Prediccion Total</div><div class="kpi-value">${pred_total:,.0f}</div><div class="kpi-delta">Prox. {dias_futuro} dias</div></div>', unsafe_allow_html=True)

        # ESCENARIOS
        st.markdown('<div class="section-title">Escenarios de Prediccion</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        e1.markdown(f'<div class="escenario-card esc-optimista"><div style="font-size:0.85rem;color:#28a745;font-weight:600;">🟢 OPTIMISTA</div><div style="font-size:1.6rem;font-weight:800;margin:0.5rem 0;">${pred_fut["yhat_upper"].sum():,.0f}</div><div style="font-size:0.8rem;color:#888;">Si todo va bien</div></div>', unsafe_allow_html=True)
        e2.markdown(f'<div class="escenario-card esc-esperado"><div style="font-size:0.85rem;color:#007bff;font-weight:600;">🔵 ESPERADO</div><div style="font-size:1.6rem;font-weight:800;margin:0.5rem 0;">${pred_fut["yhat"].sum():,.0f}</div><div style="font-size:0.8rem;color:#888;">Mas probable</div></div>', unsafe_allow_html=True)
        e3.markdown(f'<div class="escenario-card esc-conservador"><div style="font-size:0.85rem;color:#dc3545;font-weight:600;">🔴 CONSERVADOR</div><div style="font-size:1.6rem;font-weight:800;margin:0.5rem 0;">${pred_fut["yhat_lower"].sum():,.0f}</div><div style="font-size:0.8rem;color:#888;">Si algo sale mal</div></div>', unsafe_allow_html=True)

        # COMPARACION MODELOS
        st.markdown('<div class="section-title">Comparacion de Modelos</div>', unsafe_allow_html=True)
        mc = st.columns(4)
        modelos_data = [
            ('Prophet+', metricas.get('prophet_mape')),
            ('ARIMA', metricas.get('arima_mape')),
            ('AutoARIMA', metricas.get('autoarima_mape')),
            ('XGBoost', metricas.get('xgboost_mape'))
        ]
        for i, (nom, mape) in enumerate(modelos_data):
            with mc[i]:
                if mape is not None:
                    gan = nom.replace('+', '') in metricas['modelo_ganador'] or nom in metricas['modelo_ganador']
                    badge = '<span class="badge-gan">GANADOR</span>' if gan else '<span class="badge-norm">-</span>'
                    st.markdown(f"**{nom}** {badge}", unsafe_allow_html=True)
                    st.metric("MAPE", f"{mape}%")
                else:
                    st.markdown(f"**{nom}**"); st.caption("No disp.")

        with st.expander("Que significa MAPE?"):
            st.write(f"MAPE = {metricas['MAPE']}% | Escala: <10% Excelente | 10-20% Bueno | 20-30% Aceptable | >30% Bajo")

        # RECOMENDACIONES
        st.markdown('<div class="section-title">Recomendaciones</div>', unsafe_allow_html=True)
        for rec in recomendaciones:
            css = f"rec-{rec['tipo'][:3]}"
            icon = {'positivo': '✅', 'alerta': '⚠️', 'info': '💡'}.get(rec['tipo'], '💡')
            st.markdown(f'<div class="rec-box {css}"><span style="font-size:1.2rem;margin-right:0.5rem;">{icon}</span><strong>{rec["texto"]}</strong></div>', unsafe_allow_html=True)

        # GRAFICO PRINCIPAL
        st.markdown('<div class="section-title">Prediccion de Ventas</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_limpio['ds'], y=df_limpio['y'], name='Ventas reales', line=dict(color='#2196F3', width=2), hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=prediccion['ds'], y=prediccion['yhat'], name='Prediccion', line=dict(color='#FF5722', width=2, dash='dash'), hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=pd.concat([prediccion['ds'], prediccion['ds'][::-1]]), y=pd.concat([prediccion['yhat_upper'], prediccion['yhat_lower'][::-1]]), fill='toself', fillcolor='rgba(255,87,34,0.12)', line=dict(color='rgba(0,0,0,0)'), name='Intervalo 95%', hoverinfo='skip'))
        fig.add_vline(x=df_limpio['ds'].max(), line=dict(color='green', width=2, dash='dot'))
        fig.add_annotation(x=df_limpio['ds'].max(), y=1.05, yref='paper', text='Hoy', showarrow=False, font=dict(color='green', size=13))
        fig.update_layout(template='plotly_white', height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_title="Fecha", yaxis_title="Ventas ($)", margin=dict(l=40, r=40, t=80, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # TABLA + DESCARGA
        st.markdown('<div class="section-title">Proximos Dias</div>', unsafe_allow_html=True)
        pt = prediccion[prediccion['ds'] > df_limpio['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(dias_futuro)
        pt.columns = ['Fecha', 'Prediccion', 'Minimo', 'Maximo']
        pt['Fecha'] = pt['Fecha'].dt.strftime('%d/%m/%Y')
        pt = pt.round(2)
        pt['Cambio %'] = pt['Prediccion'].pct_change() * 100
        pt['Cambio %'] = pt['Cambio %'].round(1)
        st.dataframe(pt, use_container_width=True, hide_index=True)

        csv = pt.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar predicciones (CSV)", csv, f"prediccion_{nombre_negocio.replace(' ', '_')}.csv", "text/csv", use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Analisis por Dimension</div>', unsafe_allow_html=True)
        st.caption("Graficos generados a partir de tus datos sin procesar (transacciones)")

        # Fila 1: Rama + Ciudad
        g1, g2 = st.columns(2)
        fig_rama = grafico_ventas_rama(df_raw, columnas['branch'], col_ventas_sel)
        if fig_rama:
            with g1: st.plotly_chart(fig_rama, use_container_width=True)
        else:
            with g1: st.info("No se detecto columna de Rama/Sucursal")

        fig_ciudad = grafico_ventas_ciudad(df_raw, columnas['ciudad'], col_ventas_sel)
        if fig_ciudad:
            with g2: st.plotly_chart(fig_ciudad, use_container_width=True)
        else:
            with g2: st.info("No se detecto columna de Ciudad")

        # Fila 2: Producto + Dia semana
        g3, g4 = st.columns(2)
        fig_prod = grafico_ventas_producto(df_raw, columnas['producto'], col_ventas_sel)
        if fig_prod:
            with g3: st.plotly_chart(fig_prod, use_container_width=True)
        else:
            with g3: st.info("No se detecto columna de Producto")

        with g4:
            st.plotly_chart(grafico_patron_dia_semana(df_raw, col_fecha_sel, col_ventas_sel), use_container_width=True)

        # Fila 3: Hora + Heatmap
        g5, g6 = st.columns(2)
        fig_hora = grafico_ventas_por_hora(df_raw, columnas['hora'], col_ventas_sel)
        if fig_hora:
            with g5: st.plotly_chart(fig_hora, use_container_width=True)
        else:
            with g5: st.info("No se detecto columna de Hora")

        fig_heat = grafico_heatmap_rama_ciudad(df_raw, columnas['branch'], columnas['ciudad'], col_ventas_sel)
        if fig_heat:
            with g6: st.plotly_chart(fig_heat, use_container_width=True)
        else:
            with g6: st.info("Se requieren columnas de Rama y Ciudad para el heatmap")

        # Tabla comparacion
        st.markdown('<div class="section-title">Comparacion de Periodos</div>', unsafe_allow_html=True)
        st.dataframe(tabla_comparacion_periodos(df_raw, col_fecha_sel, col_ventas_sel), use_container_width=True, hide_index=True)

    # FOOTER
    st.markdown(f'<div class="footer">✅ {nombre_negocio} | SalesPredict v3.0 | {metricas["modelo_ganador"]} | Precision: {metricas["Precision"]}%</div>', unsafe_allow_html=True)
