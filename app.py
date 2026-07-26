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
    """MAPE simetrico (sMAPE) + MAPE filtrado. Mas estable con ceros y picos."""
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0, None)

    # sMAPE: penaliza menos dias con venta muy baja
    denom = (np.abs(real) + np.abs(pred)) / 2
    mask_s = denom > 1e-6
    smape = np.mean(np.abs(real[mask_s] - pred[mask_s]) / denom[mask_s]) * 100 if mask_s.any() else 999

    # MAPE clasico solo en dias con venta significativa (>5% de la mediana)
    umbral = max(np.median(real[real > 0]) * 0.05, 1.0) if (real > 0).any() else 1.0
    mask = real >= umbral
    if mask.sum() >= max(7, len(real) // 4):
        mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100
    else:
        mape = smape

    # Metrica final: promedio conservador (peor caso entre ambos)
    return float(np.mean([mape, smape]))


def correr_prophet_mejorado(df_train, df_test, feriados=None, usar_regressors=True, usar_log=False):
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
            changepoint_prior_scale=0.15,      # Menos overfitting en series ruidosas
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
        real_test = df_test['y'].values.copy()
        if usar_log:
            pred_test = np.expm1(pred_test)
            real_test = np.expm1(real_test)
        pred_test = np.clip(pred_test, 0, None)

        mape = calcular_mape(real_test, pred_test)
        mae = mean_absolute_error(real_test, pred_test)

        return {
            'nombre': 'Prophet+', 'mape': round(mape, 2),
            'mae': round(mae, 2), 'modelo': modelo,
            'pred_test': pred_test, 'usar_log': usar_log,
            'feriados': feriados, 'usar_regressors': usar_regressors
        }
    except Exception as e:
        return {'nombre': 'Prophet+', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_arima(df_train, df_test, usar_log=False):
    """ARIMA con transformacion log opcional."""
    try:
        serie = df_train['y'].copy()
        if usar_log:
            serie = np.log1p(serie)
        modelo = ARIMA(serie, order=(1, 1, 1))
        res = modelo.fit()
        pred = res.forecast(steps=len(df_test))
        pred_vals = pred.values if hasattr(pred, 'values') else np.asarray(pred)
        real = df_test['y'].values
        if usar_log:
            pred_vals = np.expm1(pred_vals)
        pred_vals = np.clip(pred_vals, 0, None)
        mape = calcular_mape(real, pred_vals)
        mae = mean_absolute_error(real, pred_vals)
        return {
            'nombre': 'ARIMA', 'mape': round(mape, 2),
            'mae': round(mae, 2), 'modelo': res,
            'pred_test': pred_vals, 'usar_log': usar_log
        }
    except Exception as e:
        return {'nombre': 'ARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def correr_autoarima(df_train, df_test, usar_log=False):
    if not HAS_PMDARIMA:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999,
                'error': 'Instala: pip install pmdarima'}
    try:
        serie = df_train['y'].copy()
        if usar_log:
            serie = np.log1p(serie)
        seasonal = len(df_train) >= 60
        modelo = auto_arima(
            serie,
            seasonal=seasonal,
            m=7 if seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            max_p=5, max_d=2, max_q=5,
            max_P=2, max_Q=2,
            n_jobs=1, trace=False,
            error_action='ignore'
        )
        pred = modelo.predict(n_periods=len(df_test))
        pred_vals = np.asarray(pred)
        real = df_test['y'].values
        if usar_log:
            pred_vals = np.expm1(pred_vals)
        pred_vals = np.clip(pred_vals, 0, None)
        mape = calcular_mape(real, pred_vals)
        mae = mean_absolute_error(real, pred_vals)
        return {
            'nombre': 'AutoARIMA', 'mape': round(mape, 2),
            'mae': round(mae, 2), 'modelo': modelo,
            'pred_test': pred_vals, 'usar_log': usar_log
        }
    except Exception as e:
        return {'nombre': 'AutoARIMA', 'mape': 999, 'mae': 999, 'error': str(e)}


def _features_test_sin_leakage(df_train, df_test, usar_log=False):
    """Genera features de test usando solo historia disponible (sin ver ventas futuras)."""
    history = df_train.copy().reset_index(drop=True)
    bloques = []

    for i in range(len(df_test)):
        fila = df_test.iloc[[i]].copy()
        fila['y'] = np.nan
        ventana = pd.concat([history, fila], ignore_index=True)
        feat = crear_features_v3(ventana, usar_log=usar_log).tail(1)
        bloques.append(feat)
        history = pd.concat(
            [history, df_test.iloc[[i]][['ds', 'y']].reset_index(drop=True)],
            ignore_index=True
        )

    return pd.concat(bloques, ignore_index=True)


def correr_xgboost_v3(df_train, df_test, usar_log=False):
    """XGBoost con features v3 y validacion interna (sin leakage en test)."""
    if not HAS_XGBOOST:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999,
                'error': 'Instala: pip install xgboost'}
    try:
        df_train_f = crear_features_v3(df_train.copy(), usar_log=usar_log)
        df_test_f = _features_test_sin_leakage(df_train, df_test, usar_log=usar_log)

        feature_cols = [c for c in df_train_f.columns
                        if c not in ['ds', 'y', 'y_raw']]

        X_all = df_train_f[feature_cols]
        y_all = df_train_f['y']
        X_test = df_test_f[feature_cols]
        y_test = df_test['y'].values

        # Early stopping solo con cola del train (evita leakage)
        val_size = max(14, int(len(X_all) * 0.15))
        X_tr, X_val = X_all.iloc[:-val_size], X_all.iloc[-val_size:]
        y_tr, y_val = y_all.iloc[:-val_size], y_all.iloc[-val_size:]

        modelo = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=3,
            random_state=42,
            early_stopping_rounds=25,
            eval_metric='mae'
        )
        modelo.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        pred = modelo.predict(X_test)
        if usar_log:
            pred = np.expm1(pred)
            y_test_eval = np.expm1(y_test)
        else:
            y_test_eval = y_test

        pred = np.clip(pred, 0, None)
        mape = calcular_mape(y_test_eval, pred)
        mae = mean_absolute_error(y_test_eval, pred)

        importance = dict(zip(feature_cols, modelo.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'nombre': 'XGBoost',
            'mape': round(mape, 2),
            'mae': round(mae, 2),
            'modelo': modelo,
            'feature_cols': feature_cols,
            'top_features': top_features,
            'pred_test': pred,
            'usar_log': usar_log
        }
    except Exception as e:
        return {'nombre': 'XGBoost', 'mape': 999, 'mae': 999, 'error': str(e)}

# ============================================
# VALIDACION WALK FORWARD + ENSEMBLE
# ============================================

def walk_forward_validation(df, pais, window_test=30, usar_log=False):
    """Valida en los ultimos N dias con ventana adaptativa."""
    df = df.sort_values('ds').reset_index(drop=True)
    n = len(df)
    window_test = min(window_test, max(14, int(n * 0.2)))
    fecha_corte = df['ds'].max() - pd.Timedelta(days=window_test)

    if (df['ds'].max() - fecha_corte).days < 14:
        split = int(n * 0.8)
        df_train = df.iloc[:split].copy()
        df_test = df.iloc[split:].copy()
    else:
        df_train = df[df['ds'] <= fecha_corte].copy()
        df_test = df[df['ds'] > fecha_corte].copy()

    if len(df_train) < 30:
        return None, "Pocos datos para entrenar (< 30 dias)"

    df_train_fit = df_train.copy()
    df_test_eval = df_test.copy()
    if usar_log:
        df_train_fit = df_train_fit.copy()
        df_train_fit['y'] = np.log1p(df_train_fit['y'])

    anos = df['ds'].dt.year.unique().tolist()
    anos += [max(anos) + 1]
    feriados = obtener_feriados(pais, anos)

    res_prophet = correr_prophet_mejorado(df_train_fit, df_test_eval, feriados, usar_log=usar_log)
    res_arima = correr_arima(df_train, df_test, usar_log=usar_log)
    res_autoarima = correr_autoarima(df_train, df_test, usar_log=usar_log)
    res_xgb = correr_xgboost_v3(df_train, df_test, usar_log=usar_log)

    resultados = [res_prophet, res_arima, res_autoarima, res_xgb]
    validos = [r for r in resultados if r.get('mape', 999) < 100 and 'pred_test' in r]

    if not validos:
        return None, "Ningun modelo convergio"

    validos.sort(key=lambda x: x['mape'])
    return validos, df_train, df_test, feriados


def ensemble_ponderado(resultados, real_test):
    """Combina predicciones de los 2-3 mejores modelos (peso inverso al MAPE)."""
    buenos = [r for r in resultados if r.get('mape', 999) < 50 and 'pred_test' in r]
    if not buenos:
        buenos = [resultados[0]]
    if len(buenos) == 1:
        r = buenos[0]
        return {
            'nombre': r['nombre'],
            'mape': r['mape'],
            'mae': r['mae'],
            'modelos': [r],
            'pesos': [1.0],
            'pred_test': r['pred_test'],
            'modelo_base': r
        }

    buenos = buenos[:3]
    pesos = np.array([1 / max(r['mape'], 0.5) for r in buenos], dtype=float)
    pesos = pesos / pesos.sum()

    preds = np.column_stack([r['pred_test'] for r in buenos])
    pred_e = np.clip((preds * pesos).sum(axis=1), 0, None)
    mape_e = calcular_mape(real_test, pred_e)
    mae_e = mean_absolute_error(real_test, pred_e)
    nombres = '+'.join([r['nombre'] for r in buenos])

    return {
        'nombre': f'Ensemble ({nombres})',
        'mape': round(mape_e, 2),
        'mae': round(mae_e, 2),
        'modelos': buenos,
        'pesos': pesos.tolist(),
        'pred_test': pred_e,
        'modelo_base': buenos[0]
    }


def _predecir_prophet_final(df, dias_futuro, feriados, usar_log=False):
    df_fit = df.copy()
    if usar_log:
        df_fit['y'] = np.log1p(df_fit['y'])

    modelo = Prophet(
        weekly_seasonality=len(df) > 30,
        yearly_seasonality=len(df) > 365,
        daily_seasonality=False,
        interval_width=0.95,
        holidays=feriados,
        changepoint_prior_scale=0.15,
        seasonality_mode='multiplicative',
        changepoint_range=0.9
    )

    regressors = []
    if len(df) > 60:
        df_f = crear_features_v3(df.copy(), usar_log=False)
        regressors = ['es_finde', 'es_quincena', 'es_fin_mes']
        for r in regressors:
            modelo.add_regressor(r)
        df_reg = df_fit.merge(df_f[['ds'] + regressors], on='ds', how='left')
        df_reg[regressors] = df_reg[regressors].fillna(0)
    else:
        df_reg = df_fit.copy()

    modelo.fit(df_reg)
    futuro = modelo.make_future_dataframe(periods=dias_futuro, freq='D')

    if regressors:
        futuro['dia_semana'] = futuro['ds'].dt.dayofweek
        futuro['es_finde'] = (futuro['dia_semana'] >= 5).astype(int)
        futuro['dia_mes'] = futuro['ds'].dt.day
        futuro['es_quincena'] = ((futuro['dia_mes'] >= 14) & (futuro['dia_mes'] <= 16)).astype(int)
        futuro['es_fin_mes'] = futuro['dia_mes'].isin([28, 29, 30, 31]).astype(int)
        futuro = futuro.drop(columns=['dia_semana', 'dia_mes'])

    pred = modelo.predict(futuro)
    if usar_log:
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            pred[col] = np.expm1(pred[col])
        pred[['yhat', 'yhat_lower', 'yhat_upper']] = pred[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

    return pred


def _predecir_xgboost_final(df, dias_futuro, usar_log=False):
    df_train_f = crear_features_v3(df.copy(), usar_log=usar_log)
    feature_cols = [c for c in df_train_f.columns if c not in ['ds', 'y', 'y_raw']]

    val_size = max(14, int(len(df_train_f) * 0.15))
    X_all, y_all = df_train_f[feature_cols], df_train_f['y']
    X_tr, X_val = X_all.iloc[:-val_size], X_all.iloc[-val_size:]
    y_tr, y_val = y_all.iloc[:-val_size], y_all.iloc[-val_size:]

    modelo = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
        min_child_weight=3, random_state=42,
        early_stopping_rounds=25, eval_metric='mae'
    )
    modelo.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    history = df.copy().reset_index(drop=True)
    filas = []
    ultima = history['ds'].max()

    for d in range(1, dias_futuro + 1):
        fecha = ultima + pd.Timedelta(days=d)
        fila = pd.DataFrame({'ds': [fecha], 'y': [np.nan]})
        ventana = pd.concat([history, fila], ignore_index=True)
        feat = crear_features_v3(ventana, usar_log=usar_log).tail(1)
        pred = float(modelo.predict(feat[feature_cols])[0])
        if usar_log:
            pred = float(np.expm1(pred))
        pred = max(0.0, pred)
        filas.append({'ds': fecha, 'yhat': pred, 'yhat_lower': pred * 0.85, 'yhat_upper': pred * 1.15})
        history = pd.concat([history, pd.DataFrame({'ds': [fecha], 'y': [pred]})], ignore_index=True)

    hist_df = df.copy()
    hist_df['yhat'] = hist_df['y']
    hist_df['yhat_lower'] = hist_df['y']
    hist_df['yhat_upper'] = hist_df['y']
    fut_df = pd.DataFrame(filas)
    return pd.concat([hist_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fut_df], ignore_index=True)


def _predecir_arima_final(df, dias_futuro, res_modelo, usar_log=False):
    serie = df['y'].copy()
    if usar_log:
        serie = np.log1p(serie)
    try:
        if HAS_PMDARIMA and res_modelo.get('nombre') == 'AutoARIMA':
            modelo = auto_arima(
                serie, seasonal=len(df) >= 60, m=7 if len(df) >= 60 else 1,
                stepwise=True, suppress_warnings=True, max_p=5, max_d=2, max_q=5,
                n_jobs=1, trace=False, error_action='ignore'
            )
            pred_f = modelo.predict(n_periods=dias_futuro)
        else:
            modelo = ARIMA(serie, order=(1, 1, 1)).fit()
            pred_f = modelo.forecast(steps=dias_futuro)
        pred_f = np.asarray(pred_f)
        if usar_log:
            pred_f = np.expm1(pred_f)
        pred_f = np.clip(pred_f, 0, None)

        fechas = pd.date_range(start=df['ds'].max() + pd.Timedelta(days=1), periods=dias_futuro, freq='D')
        hist_df = df.copy()
        hist_df['yhat'] = hist_df['y']
        hist_df['yhat_lower'] = hist_df['y']
        hist_df['yhat_upper'] = hist_df['y']
        fut_df = pd.DataFrame({
            'ds': fechas,
            'yhat': pred_f,
            'yhat_lower': pred_f * 0.85,
            'yhat_upper': pred_f * 1.15
        })
        return pd.concat([hist_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fut_df], ignore_index=True)
    except Exception:
        return _predecir_prophet_final(df, dias_futuro, None, usar_log)


def analizar_v3(df, pais, dias_futuro, usar_log=False):
    """Pipeline completo con ensemble real y modelo final alineado."""
    out = walk_forward_validation(df, pais, window_test=30, usar_log=usar_log)
    if out[0] is None:
        return None, None, out[1]

    resultados, df_train, df_test, feriados = out
    real_test = df_test['y'].values
    ganador = ensemble_ponderado(resultados, real_test)

    base = ganador.get('modelo_base', resultados[0])
    nombre_base = base['nombre']

    if 'XGBoost' in nombre_base and len(ganador.get('modelos', [])) == 1:
        prediccion = _predecir_xgboost_final(df, dias_futuro, usar_log)
    elif nombre_base in ('ARIMA', 'AutoARIMA') and len(ganador.get('modelos', [])) == 1:
        prediccion = _predecir_arima_final(df, dias_futuro, base, usar_log)
    else:
        prediccion = _predecir_prophet_final(df, dias_futuro, feriados, usar_log)

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
    page_title="SalesPredict — Predicción de ventas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .app-title { font-size: 1.75rem; font-weight: 700; color: #111827; margin: 0 0 0.25rem 0; }
    .app-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .step-badge {
        display: inline-block; background: #eef2ff; color: #4338ca;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
        text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 999px; margin-bottom: 0.35rem;
    }
    .step-title { font-size: 1.05rem; font-weight: 600; color: #111827; margin: 0 0 0.75rem 0; }
    .result-hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 12px; padding: 1.5rem 1.75rem; color: white; margin: 1rem 0 1.5rem 0;
    }
    .result-hero .label { font-size: 0.8rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.05em; }
    .result-hero .value { font-size: 2.2rem; font-weight: 800; line-height: 1.2; margin: 0.25rem 0; }
    .result-hero .hint { font-size: 0.85rem; opacity: 0.8; }
    .metric-plain {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 0.9rem 1rem; text-align: center;
    }
    .metric-plain .lbl { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-plain .val { font-size: 1.5rem; font-weight: 700; color: #111827; margin-top: 0.15rem; }
    .rec-item {
        padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
        border-left: 3px solid; font-size: 0.92rem; line-height: 1.45;
    }
    .rec-ok { background: #f0fdf4; border-color: #22c55e; color: #14532d; }
    .rec-warn { background: #fff7ed; border-color: #f97316; color: #7c2d12; }
    .rec-info { background: #eff6ff; border-color: #3b82f6; color: #1e3a8a; }
    div[data-testid="stSidebar"] { background: #fafafa; }
    .footer-note { text-align: center; color: #9ca3af; font-size: 0.78rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #f3f4f6; }
</style>
""", unsafe_allow_html=True)

# --- Encabezado ---
st.markdown('<p class="app-title">SalesPredict</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Sube tu CSV de ventas y obtén una proyección clara para los próximos días.</p>',
    unsafe_allow_html=True
)

# --- Sidebar minimalista ---
nombre_negocio = "Mi negocio"
usar_log = False

with st.sidebar:
    st.markdown("**Ajustes**")
    pais = st.selectbox(
        "País (feriados)",
        ["Bolivia", "Mexico", "Argentina", "Colombia", "Peru", "Chile",
         "Espana", "USA", "Brasil", "Ecuador", "Venezuela", "Paraguay"],
        help="Usamos los feriados de tu país para mejorar la predicción."
    )
    dias_futuro = st.slider("Días a predecir", 7, 90, 30, 7)

    with st.expander("Opciones avanzadas"):
        nombre_negocio = st.text_input("Nombre del negocio", placeholder="Ej: Supermercado Central")
        usar_log = st.toggle(
            "Suavizar picos extremos",
            value=False,
            help="Actívalo si tus ventas varían mucho de un día a otro."
        )
        if not HAS_PMDARIMA or not HAS_XGBOOST:
            st.caption("Tip: instala `pip install pmdarima xgboost` para comparar más modelos.")

    st.caption("Formato de fecha recomendado: DD/MM/YYYY · Mínimo 30 días de historial.")

# --- Paso 1: Carga de datos ---
st.markdown('<span class="step-badge">Paso 1</span>', unsafe_allow_html=True)
st.markdown('<p class="step-title">Sube tu archivo CSV</p>', unsafe_allow_html=True)

archivo = st.file_uploader(
    "Arrastra tu CSV aquí o haz clic para seleccionarlo",
    type=['csv'],
    label_visibility="collapsed"
)

if archivo is None:
    st.info("Necesitas un CSV con al menos una columna de **fecha** y otra de **ventas** (monto total por transacción o por día).")
    st.stop()

df_raw, info_carga = cargar_csv_seguro(archivo)
if df_raw is None:
    st.error(f"No se pudo leer el archivo: {info_carga}")
    st.stop()

columnas = detectar_columnas_clave(df_raw)
st.success(f"Archivo cargado: **{len(df_raw):,}** filas")

# --- Paso 2: Confirmar columnas ---
st.markdown('<span class="step-badge">Paso 2</span>', unsafe_allow_html=True)
st.markdown('<p class="step-title">Confirma las columnas correctas</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    idx_f = df_raw.columns.tolist().index(columnas['fecha']) if columnas['fecha'] in df_raw.columns else 0
    col_fecha_sel = st.selectbox("Columna de fecha", df_raw.columns.tolist(), index=idx_f)
with c2:
    idx_v = df_raw.columns.tolist().index(columnas['ventas']) if columnas['ventas'] in df_raw.columns else 0
    col_ventas_sel = st.selectbox("Columna de ventas ($)", df_raw.columns.tolist(), index=idx_v)

with st.expander("Vista previa de datos"):
    st.dataframe(df_raw.head(8), use_container_width=True, hide_index=True)
    detectadas = {k: v for k, v in columnas.items() if v}
    if detectadas:
        st.caption(f"Columnas extra detectadas: {detectadas}")

# --- Paso 3: Generar predicción ---
st.markdown('<span class="step-badge">Paso 3</span>', unsafe_allow_html=True)
st.markdown('<p class="step-title">Genera tu predicción</p>', unsafe_allow_html=True)

if st.button("Generar predicción", type="primary", use_container_width=True):
    if not nombre_negocio.strip():
        nombre_negocio = "Mi negocio"

    with st.spinner("Preparando tus datos..."):
        df_limpio, info_val = limpiar_datos_v3(df_raw, col_fecha_sel, col_ventas_sel)

    if info_val['estado'] == "ERROR":
        st.error(info_val['mensaje'])
        st.stop()
    if info_val['estado'] == "WARNING":
        st.warning(info_val['mensaje'])

    with st.spinner("Calculando predicción (puede tardar un minuto)..."):
        prediccion, metricas, error = analizar_v3(df_limpio, pais, dias_futuro, usar_log=usar_log)

    if error:
        st.error(f"No se pudo completar el análisis: {error}")
        st.stop()

    recomendaciones = generar_recomendaciones_v3(df_limpio, prediccion, metricas, info_val)
    confianza = evaluar_confiabilidad(df_limpio, metricas['MAPE'])
    pred_fut = prediccion[prediccion['ds'] > df_limpio['ds'].max()]
    pred_total = pred_fut['yhat'].sum()
    ventas_total = df_limpio['y'].sum()

    # --- Resultado principal (hero) ---
    st.markdown("---")
    precision_txt = "Alta" if metricas['Precision'] >= 85 else ("Media" if metricas['Precision'] >= 70 else "Baja")
    st.markdown(
        f'<div class="result-hero">'
        f'<div class="label">Ventas esperadas · próximos {dias_futuro} días</div>'
        f'<div class="value">${pred_total:,.0f}</div>'
        f'<div class="hint">Confianza {precision_txt.lower()} · basado en {info_val["dias"]} días de historial</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-plain"><div class="lbl">Precisión estimada</div>'
            f'<div class="val">{metricas["Precision"]}%</div></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-plain"><div class="lbl">Ventas históricas</div>'
            f'<div class="val">${ventas_total:,.0f}</div></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-plain"><div class="lbl">Promedio diario</div>'
            f'<div class="val">${info_val["venta_promedio"]:,.0f}</div></div>',
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f'<div class="metric-plain"><div class="lbl">Días analizados</div>'
            f'<div class="val">{info_val["dias"]}</div></div>',
            unsafe_allow_html=True
        )

    # --- Gráfico principal ---
    st.subheader("Proyección de ventas")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_limpio['ds'], y=df_limpio['y'], name='Ventas reales',
        line=dict(color='#2563eb', width=2),
        hovertemplate='%{x|%d/%m/%Y}<br>$%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=prediccion['ds'], y=prediccion['yhat'], name='Predicción',
        line=dict(color='#f97316', width=2, dash='dash'),
        hovertemplate='%{x|%d/%m/%Y}<br>$%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([prediccion['ds'], prediccion['ds'][::-1]]),
        y=pd.concat([prediccion['yhat_upper'], prediccion['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(249,115,22,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Rango probable',
        hoverinfo='skip'
    ))
    fig.add_vline(x=df_limpio['ds'].max(), line=dict(color='#22c55e', width=1, dash='dot'))
    fig.update_layout(
        template='plotly_white', height=420, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="", yaxis_title="Ventas ($)",
        margin=dict(l=40, r=20, t=30, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Escenarios en lenguaje simple ---
    st.subheader("Tres escenarios")
    e1, e2, e3 = st.columns(3)
    e1.metric("Optimista", f"${pred_fut['yhat_upper'].sum():,.0f}", help="Si las ventas van mejor de lo esperado")
    e2.metric("Esperado", f"${pred_fut['yhat'].sum():,.0f}", help="Escenario más probable")
    e3.metric("Conservador", f"${pred_fut['yhat_lower'].sum():,.0f}", help="Si las ventas van peor de lo esperado")

    # --- Recomendaciones ---
    st.subheader("Qué hacer con esto")
    css_map = {'positivo': 'rec-ok', 'alerta': 'rec-warn', 'info': 'rec-info'}
    for rec in recomendaciones[:5]:
        st.markdown(
            f'<div class="rec-item {css_map.get(rec["tipo"], "rec-info")}">{rec["texto"]}</div>',
            unsafe_allow_html=True
        )

    # --- Tabla descargable ---
    st.subheader("Detalle día a día")
    pt = pred_fut[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(dias_futuro).copy()
    pt.columns = ['Fecha', 'Predicción', 'Mínimo', 'Máximo']
    pt['Fecha'] = pt['Fecha'].dt.strftime('%d/%m/%Y')
    pt = pt.round(2)
    st.dataframe(pt, use_container_width=True, hide_index=True)

    csv_out = pt.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar predicción (CSV)",
        csv_out,
        f"prediccion_{nombre_negocio.replace(' ', '_')}.csv",
        "text/csv",
        use_container_width=True
    )

    # --- Detalle técnico (colapsado) ---
    with st.expander("Detalle técnico (para curiosos)"):
        st.caption(f"Modelo seleccionado: **{metricas['modelo_ganador']}** · Error MAPE: **{metricas['MAPE']}%**")
        st.caption(f"Nivel de confianza del análisis: **{confianza['nivel']}** ({confianza['score']}/100)")
        for d in confianza['detalles']:
            st.caption(d)

        st.markdown("**Comparación de modelos probados**")
        cols_m = st.columns(4)
        for i, (nom, mape) in enumerate([
            ('Prophet', metricas.get('prophet_mape')),
            ('ARIMA', metricas.get('arima_mape')),
            ('AutoARIMA', metricas.get('autoarima_mape')),
            ('XGBoost', metricas.get('xgboost_mape'))
        ]):
            with cols_m[i]:
                if mape is not None and mape < 999:
                    st.metric(nom, f"{mape}%", label="error MAPE")
                else:
                    st.caption(f"{nom}: no disponible")

        st.caption("MAPE = error porcentual medio. Menor es mejor. Objetivo ideal: ≤15% con datos estables.")

    # --- Análisis por dimensión (colapsado) ---
    with st.expander("Análisis por sucursal, producto, ciudad…"):
        st.caption("Gráficos basados en tus datos originales (transacciones).")
        g1, g2 = st.columns(2)
        fig_rama = grafico_ventas_rama(df_raw, columnas['branch'], col_ventas_sel)
        fig_ciudad = grafico_ventas_ciudad(df_raw, columnas['ciudad'], col_ventas_sel)
        with g1:
            st.plotly_chart(fig_rama, use_container_width=True) if fig_rama else st.caption("Sin columna de sucursal.")
        with g2:
            st.plotly_chart(fig_ciudad, use_container_width=True) if fig_ciudad else st.caption("Sin columna de ciudad.")

        g3, g4 = st.columns(2)
        fig_prod = grafico_ventas_producto(df_raw, columnas['producto'], col_ventas_sel)
        with g3:
            st.plotly_chart(fig_prod, use_container_width=True) if fig_prod else st.caption("Sin columna de producto.")
        with g4:
            st.plotly_chart(grafico_patron_dia_semana(df_raw, col_fecha_sel, col_ventas_sel), use_container_width=True)

        g5, g6 = st.columns(2)
        fig_hora = grafico_ventas_por_hora(df_raw, columnas['hora'], col_ventas_sel)
        fig_heat = grafico_heatmap_rama_ciudad(df_raw, columnas['branch'], columnas['ciudad'], col_ventas_sel)
        with g5:
            st.plotly_chart(fig_hora, use_container_width=True) if fig_hora else st.caption("Sin columna de hora.")
        with g6:
            st.plotly_chart(fig_heat, use_container_width=True) if fig_heat else st.caption("Se necesitan sucursal y ciudad.")

        st.markdown("**Comparación de periodos**")
        st.dataframe(tabla_comparacion_periodos(df_raw, col_fecha_sel, col_ventas_sel), use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="footer-note">{nombre_negocio} · SalesPredict · Precisión {metricas["Precision"]}%</div>',
        unsafe_allow_html=True
    )
