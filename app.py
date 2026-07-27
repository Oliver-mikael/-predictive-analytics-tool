import streamlit as st
import pandas as pd
import numpy as np
import warnings
import holidays
warnings.filterwarnings('ignore')

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
# CONSTANTES Y CONFIGURACION
# ============================================

MAX_FILAS_CACHE = 2_000_000  # Aviso para datasets muy grandes
CHUNK_SIZE = 200_000         # Lectura por bloques

# ============================================
# FUNCIONES DE UTILIDAD (CACHE + PERFORMANCE)
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_csv_seguro(archivo_bytes, encoding='latin1'):
    """
    Carga el CSV completo (por bloques si es grande).

    Antes se muestreaba con skiprows cuando el archivo era grande: eso rompe
    la suma diaria de ventas (se pierden transacciones de cada dia) y por si
    solo inflaba el error. Mejor leer todo y solo avisar si es enorme.
    """
    try:
        archivo_bytes.seek(0)
        bloques = pd.read_csv(archivo_bytes, encoding=encoding, chunksize=CHUNK_SIZE)
        df = pd.concat(bloques, ignore_index=True)
        total_filas = len(df)
        if total_filas > MAX_FILAS_CACHE:
            st.warning(f"Archivo muy grande ({total_filas:,} filas). El analisis puede tardar.")
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
# PARSEO DE FECHAS ROBUSTO
# ============================================

def parsear_fechas(serie):
    """
    Parsea fechas probando formatos dia-primero y mes-primero y quedandose
    con el que menos valores pierde. Devuelve (fechas, info).

    Un dayfirst=True fijo destroza CSVs en formato US (MM/DD/YYYY): las filas
    con dia > 12 quedan en NaT y el resto se transpone (12/03 -> 3 de dic).
    """
    texto = serie.astype(str).str.strip()
    candidatos = []

    for etiqueta, kwargs in [
        ("DD/MM/YYYY", {'dayfirst': True}),
        ("MM/DD/YYYY", {'dayfirst': False}),
    ]:
        parsed = pd.to_datetime(texto, errors='coerce', **kwargs)
        nulos = int(parsed.isna().sum())
        # Penalizacion extra: un formato correcto suele dar dias consecutivos
        dias_unicos = parsed.dropna().dt.normalize().nunique()
        rango = 1
        if dias_unicos > 1:
            rango = max(1, (parsed.max() - parsed.min()).days + 1)
        densidad = dias_unicos / rango
        candidatos.append((nulos, -densidad, etiqueta, parsed))

    candidatos.sort(key=lambda x: (x[0], x[1]))
    nulos, _, etiqueta, fechas = candidatos[0]

    info = {
        'formato': etiqueta,
        'no_parseadas': nulos,
        'pct_no_parseadas': round(nulos / max(len(texto), 1) * 100, 2)
    }
    return fechas, info


# ============================================
# LIMPIEZA Y FEATURE ENGINEERING
# ============================================

def limpiar_datos_v3(df, col_fecha, col_ventas):
    """
    Limpia y agrega a serie diaria.
    Cambios clave vs version anterior:
      - parseo de fecha auto (dia-primero vs mes-primero)
      - los dias sin datos NO se rellenan con 0 a ciegas: se detecta el patron
        de dias cerrados (ej. domingos) y el resto se marca como hueco real
      - los outliers NO se recortan aqui (se recortan dentro de cada fold con
        umbrales calculados solo con train, para no filtrar futuro)
    Retorna: (df_diario, info)
    """
    df_proc = pd.DataFrame()
    fechas, info_fecha = parsear_fechas(df[col_fecha])
    df_proc['ds'] = fechas.dt.normalize()
    df_proc['y'] = pd.to_numeric(df[col_ventas], errors='coerce')

    df_proc = df_proc.dropna()
    df_proc = df_proc[df_proc['y'] >= 0]
    df_proc = df_proc.sort_values('ds')

    if df_proc.empty:
        info = {'dias': 0, 'registros': 0, 'pct_zeros': 100, 'estado': 'ERROR',
                'mensaje': 'No se pudo interpretar ninguna fecha/monto valido.',
                'formato_fecha': info_fecha['formato'],
                'pct_no_parseadas': info_fecha['pct_no_parseadas']}
        return df_proc.assign(ds=pd.to_datetime([]), y=[]), info

    # Suma diaria
    df_diario = df_proc.groupby('ds', as_index=False)['y'].sum()

    # Grilla diaria completa: los dias ausentes quedan como NaN (no como 0)
    rango = pd.DataFrame({
        'ds': pd.date_range(start=df_diario['ds'].min(),
                            end=df_diario['ds'].max(), freq='D')
    })
    df_diario = rango.merge(df_diario, on='ds', how='left')

    # Dias cerrados estructurales: un dia de semana ausente casi siempre
    ausente = df_diario['y'].isna()
    dow = df_diario['ds'].dt.dayofweek
    dias_cerrados = []
    for d in range(7):
        mask = dow == d
        if mask.sum() >= 4 and ausente[mask].mean() >= 0.8:
            dias_cerrados.append(d)

    # Un dia cerrado no es un error de datos: es una venta 0 predecible.
    # Se marca para excluirlo de las metricas y se deja fuera del entrenamiento.
    df_diario['cerrado'] = dow.isin(dias_cerrados).astype(int)

    # Huecos aislados (no estructurales): interpolar en vez de meter 0,
    # que es lo que mas inflaba el MAPE en CSVs con dias faltantes.
    huecos = int((ausente & (df_diario['cerrado'] == 0)).sum())
    df_diario['y'] = df_diario['y'].interpolate(limit_direction='both')
    df_diario.loc[df_diario['cerrado'] == 1, 'y'] = 0.0

    dias = (df_diario['ds'].max() - df_diario['ds'].min()).days + 1
    dias_abiertos = int((df_diario['cerrado'] == 0).sum())
    abiertos = df_diario[df_diario['cerrado'] == 0]
    pct_zeros = (abiertos['y'] == 0).sum() / max(len(abiertos), 1) * 100
    cv = abiertos['y'].std() / abiertos['y'].mean() if abiertos['y'].mean() else 0

    if dias_abiertos < 30:
        estado, mensaje = "ERROR", f"Solo {dias_abiertos} dias con datos. Minimo 30."
    elif info_fecha['pct_no_parseadas'] > 20:
        estado, mensaje = "ERROR", (f"{info_fecha['pct_no_parseadas']}% de fechas ilegibles "
                                    f"(formato detectado: {info_fecha['formato']}). Revisa el CSV.")
    elif pct_zeros > 40:
        estado, mensaje = "ERROR", f"{pct_zeros:.1f}% de dias abiertos en 0. Datos muy fragmentados."
    elif pct_zeros > 20:
        estado, mensaje = "WARNING", f"{pct_zeros:.1f}% de dias en 0. Precision afectada."
    elif huecos > dias * 0.1:
        estado, mensaje = "WARNING", f"{huecos} dias sin registros se interpolaron."
    elif dias_abiertos < 90:
        estado, mensaje = "WARNING", f"Solo {dias_abiertos} dias. Recomendado 90+ para precision."
    else:
        estado, mensaje = "OK", "Datos validos."

    info = {
        'dias': dias_abiertos,
        'dias_calendario': dias,
        'registros': len(df_diario),
        'pct_zeros': round(pct_zeros, 2),
        'huecos_interpolados': huecos,
        'dias_cerrados': dias_cerrados,
        'cv': round(float(cv), 3),
        'estado': estado,
        'mensaje': mensaje,
        'formato_fecha': info_fecha['formato'],
        'pct_no_parseadas': info_fecha['pct_no_parseadas'],
        'fecha_min': df_diario['ds'].min(),
        'fecha_max': df_diario['ds'].max(),
        'venta_promedio': abiertos['y'].mean(),
        'venta_std': abiertos['y'].std()
    }

    return df_diario[['ds', 'y', 'cerrado']], info


def sugerir_log(df):
    """Log conviene con varianza alta y asimetria positiva, y sin ceros dominantes."""
    y = df.loc[df.get('cerrado', 0) == 0, 'y'] if 'cerrado' in df else df['y']
    y = y[y > 0]
    if len(y) < 30:
        return False
    cv = y.std() / y.mean()
    asimetria = float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-9))
    return bool(cv > 0.45 and asimetria > 0.6)


def features_calendario(fechas):
    """Features conocidas de antemano (no dependen de ventas => nunca hay leakage)."""
    f = pd.DataFrame({'ds': pd.to_datetime(fechas)})
    f['dia_semana'] = f['ds'].dt.dayofweek
    f['es_finde'] = (f['dia_semana'] >= 5).astype(int)
    f['dia_mes'] = f['ds'].dt.day
    f['mes'] = f['ds'].dt.month
    f['dias_en_mes'] = f['ds'].dt.days_in_month
    f['semana_ano'] = f['ds'].dt.isocalendar().week.astype(int)
    f['es_quincena'] = f['dia_mes'].between(14, 17).astype(int)
    # Fin de mes relativo (no fijo en 28-31: en febrero el 28 ya es fin de mes)
    f['es_fin_mes'] = (f['dias_en_mes'] - f['dia_mes'] <= 2).astype(int)
    f['es_inicio_mes'] = (f['dia_mes'] <= 3).astype(int)
    f['dia_semana_sin'] = np.sin(2 * np.pi * f['dia_semana'] / 7)
    f['dia_semana_cos'] = np.cos(2 * np.pi * f['dia_semana'] / 7)
    f['mes_sin'] = np.sin(2 * np.pi * f['mes'] / 12)
    f['mes_cos'] = np.cos(2 * np.pi * f['mes'] / 12)
    return f.drop(columns=['dias_en_mes'])


REGRESSORS_PROPHET = ['es_finde', 'es_quincena', 'es_fin_mes']
LAGS = [1, 2, 3, 7, 14, 21, 28]


def crear_features_v3(df, feriados_set=None):
    """
    Features para XGBoost. Todos los lags/medias usan shift(>=1), asi que la
    fila del dia a predecir se puede construir con y=NaN sin mirar el futuro.
    IMPORTANTE: no se hace dropna aqui (antes se borraba justamente la fila
    que se queria predecir y se predecia el dia equivocado).
    """
    df = df.sort_values('ds').reset_index(drop=True).copy()
    cal = features_calendario(df['ds'])
    df = pd.concat([df, cal.drop(columns=['ds'])], axis=1)

    if feriados_set is not None:
        fechas_norm = df['ds'].dt.normalize()
        df['es_feriado'] = fechas_norm.isin(feriados_set).astype(int)
        df['feriado_manana'] = (fechas_norm + pd.Timedelta(days=1)).isin(feriados_set).astype(int)
        df['feriado_ayer'] = (fechas_norm - pd.Timedelta(days=1)).isin(feriados_set).astype(int)

    y_prev = df['y'].shift(1)
    for lag in LAGS:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    df['ma_7'] = y_prev.rolling(7, min_periods=7).mean()
    df['ma_14'] = y_prev.rolling(14, min_periods=14).mean()
    df['ma_28'] = y_prev.rolling(28, min_periods=28).mean()
    df['std_7'] = y_prev.rolling(7, min_periods=7).std()
    # Nivel del mismo dia de semana en las ultimas 4 semanas (patron semanal)
    df['ma_dow_4'] = df[['lag_7', 'lag_14', 'lag_21', 'lag_28']].mean(axis=1)
    df['diff_7'] = df['lag_1'] - df['lag_8'] if 'lag_8' in df else df['lag_1'] - df['y'].shift(8)
    df['tendencia_7_28'] = df['ma_7'] / (df['ma_28'] + 1e-6)
    df['ratio_lag1_ma7'] = df['lag_1'] / (df['ma_7'] + 1e-6)
    df['ratio_dow'] = df['lag_7'] / (df['ma_7'] + 1e-6)
    return df


def columnas_features(df):
    return [c for c in df.columns if c not in ('ds', 'y', 'cerrado')]


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
                for fecha, nombre in clase(years=a).items():
                    lista.append({'holiday': nombre, 'ds': pd.Timestamp(fecha)})
            if lista:
                return pd.DataFrame(lista).drop_duplicates('ds')
    except Exception:
        pass
    return None


# ============================================
# METRICAS
# ============================================

def calcular_metricas(real, pred, mask_valida=None):
    """
    MAPE real (no inventado) + WAPE + sMAPE + MAE.
      - MAPE: solo sobre dias con venta > 0 (es indefinido en 0).
      - WAPE: sum|error| / sum(real). Es la metrica de seleccion: no explota
        con dias de venta minima y no premia predecir por debajo.
    """
    real = np.asarray(real, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    if mask_valida is not None:
        mask_valida = np.asarray(mask_valida, dtype=bool)
        real, pred = real[mask_valida], pred[mask_valida]

    if len(real) == 0:
        return {'mape': 999.0, 'wape': 999.0, 'smape': 999.0, 'mae': 999.0, 'n': 0}

    err = np.abs(real - pred)
    pos = real > 0
    mape = float(np.mean(err[pos] / real[pos]) * 100) if pos.any() else 999.0
    wape = float(err.sum() / real.sum() * 100) if real.sum() > 0 else 999.0
    denom = (np.abs(real) + np.abs(pred)) / 2
    ok = denom > 1e-9
    smape = float(np.mean(err[ok] / denom[ok]) * 100) if ok.any() else 999.0
    if not pos.any():
        mape = wape
    return {'mape': round(mape, 2), 'wape': round(wape, 2),
            'smape': round(smape, 2), 'mae': round(float(err.mean()), 2), 'n': int(len(real))}


def calcular_mape(real, pred):
    """Compatibilidad: MAPE real sobre dias con venta > 0."""
    return calcular_metricas(real, pred)['mape']


# ============================================
# MODELOS
# Todos comparten la misma firma: (df_train, horizonte, ctx) -> DataFrame(ds, yhat)
# Asi el modelo que se valida es EXACTAMENTE el que produce el forecast final.
# ============================================

def _recortar_outliers(df, p=0.99):
    """Winsoriza con umbral calculado SOLO con los datos de entrenamiento."""
    df = df.copy()
    abiertos = df['y'][df.get('cerrado', 0) == 0] if 'cerrado' in df else df['y']
    if len(abiertos) < 30:
        return df
    hi = abiertos.quantile(p)
    lo = abiertos.quantile(1 - p)
    df['y'] = df['y'].clip(lower=max(lo, 0), upper=hi)
    return df


def _tr(y, usar_log):
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None)) if usar_log else np.asarray(y, dtype=float)


def _inv(z, usar_log):
    z = np.asarray(z, dtype=float)
    if usar_log:
        z = np.expm1(np.clip(z, -20, 30))
    return np.clip(z, 0, None)


def _fechas_futuras(df_train, horizonte):
    return pd.date_range(start=df_train['ds'].max() + pd.Timedelta(days=1),
                         periods=horizonte, freq='D')


def _df_entrenable(df_train, usar_log):
    """Serie de entrenamiento: sin dias cerrados y en la escala del modelo."""
    d = df_train.copy()
    if 'cerrado' in d:
        d = d[d['cerrado'] == 0]
    d = d[['ds', 'y']].dropna().reset_index(drop=True)
    d['y'] = _tr(d['y'], usar_log)
    return d


def modelo_baseline(df_train, horizonte, ctx):
    """
    Baseline estacional: nivel robusto reciente x perfil de dia de semana.
    Sirve de piso de comparacion; en series muy ruidosas suele ganar a todo.
    """
    d = _df_entrenable(df_train, usar_log=False)
    if len(d) < 14:
        return None
    ult = d.tail(28)
    nivel = float(ult['y'].median())
    perfil = ult.assign(dow=ult['ds'].dt.dayofweek).groupby('dow')['y'].median()
    perfil = (perfil / max(nivel, 1e-6)).clip(0.4, 2.0)
    fechas = _fechas_futuras(df_train, horizonte)
    factor = np.array([perfil.get(f.dayofweek, 1.0) for f in fechas])
    return pd.DataFrame({'ds': fechas, 'yhat': np.clip(nivel * factor, 0, None)})


def modelo_prophet(df_train, horizonte, ctx):
    d = _df_entrenable(df_train, ctx['usar_log'])
    if len(d) < 30:
        return None
    usar_reg = len(d) > 60
    modelo = Prophet(
        weekly_seasonality=len(d) > 30,
        yearly_seasonality=len(d) > 540,
        daily_seasonality=False,
        interval_width=0.8,
        holidays=ctx.get('feriados'),
        # Series diarias de ventas son ruidosas: 0.15 sobreajustaba la tendencia
        # y la extrapolaba al futuro. 0.03 da una tendencia mucho mas estable.
        changepoint_prior_scale=ctx.get('cps', 0.03),
        seasonality_prior_scale=ctx.get('sps', 8.0),
        seasonality_mode='multiplicative' if (not ctx['usar_log'] and ctx.get('multiplicativo', True)) else 'additive',
        changepoint_range=0.9
    )
    if usar_reg:
        for r in REGRESSORS_PROPHET:
            modelo.add_regressor(r, standardize=False)
        cal = features_calendario(d['ds'])
        d = pd.concat([d, cal[REGRESSORS_PROPHET]], axis=1)

    modelo.fit(d)
    fechas = _fechas_futuras(df_train, horizonte)
    futuro = pd.DataFrame({'ds': fechas})
    if usar_reg:
        futuro = pd.concat([futuro, features_calendario(fechas)[REGRESSORS_PROPHET]], axis=1)
    pred = modelo.predict(futuro)

    out = pd.DataFrame({'ds': fechas})
    out['yhat'] = _inv(pred['yhat'].values, ctx['usar_log'])
    out['yhat_lower'] = _inv(pred['yhat_lower'].values, ctx['usar_log'])
    out['yhat_upper'] = _inv(pred['yhat_upper'].values, ctx['usar_log'])
    return out


def modelo_sarima(df_train, horizonte, ctx):
    """AutoARIMA estacional (m=7) si pmdarima esta disponible; si no, SARIMAX fijo."""
    d = _df_entrenable(df_train, ctx['usar_log'])
    if len(d) < 40:
        return None
    serie = d['y'].to_numpy()
    fechas = _fechas_futuras(df_train, horizonte)
    estacional = len(serie) >= 70

    if HAS_PMDARIMA:
        modelo = auto_arima(
            serie, seasonal=estacional, m=7 if estacional else 1,
            d=None, D=1 if estacional else 0,
            start_p=0, start_q=0, max_p=3, max_q=3, max_P=1, max_Q=1,
            stepwise=True, suppress_warnings=True, error_action='ignore',
            information_criterion='aicc', n_jobs=1, trace=False
        )
        pred = np.asarray(modelo.predict(n_periods=horizonte))
    else:
        orden_est = (1, 0, 1, 7) if estacional else (0, 0, 0, 0)
        res = SARIMAX(serie, order=(1, 1, 1), seasonal_order=orden_est,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = np.asarray(res.forecast(steps=horizonte))

    return pd.DataFrame({'ds': fechas, 'yhat': _inv(pred, ctx['usar_log'])})


def _xgb_regresor(n_estimators):
    return xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=2.0,
        min_child_weight=5, random_state=42, n_jobs=2, eval_metric='mae'
    )


def modelo_xgboost(df_train, horizonte, ctx):
    """
    XGBoost recursivo. Dos correcciones importantes:
      - el early stopping se usa solo para elegir n_estimators y luego se
        reentrena con TODO el train (antes el modelo final perdia el 15% final,
        que es justo la parte mas informativa).
      - la prediccion multi-paso es recursiva con sus propias predicciones,
        igual en validacion y en produccion (antes la validacion se alimentaba
        con las ventas reales del test: leakage y MAPE optimista).
    """
    if not HAS_XGBOOST:
        return None
    d = _df_entrenable(df_train, ctx['usar_log'])
    if len(d) < 60:
        return None

    feriados_set = ctx.get('feriados_set')
    feats = crear_features_v3(d, feriados_set)
    cols = columnas_features(feats)
    entren = feats.dropna(subset=cols)
    if len(entren) < 40:
        return None

    X, y = entren[cols], entren['y']
    val_size = int(np.clip(len(X) * 0.15, 14, 60))
    if len(X) - val_size >= 30:
        es = _xgb_regresor(600)
        es.set_params(early_stopping_rounds=40)
        es.fit(X.iloc[:-val_size], y.iloc[:-val_size],
               eval_set=[(X.iloc[-val_size:], y.iloc[-val_size:])], verbose=False)
        n_arboles = max(50, int(getattr(es, 'best_iteration', 250) or 250))
    else:
        n_arboles = 250

    modelo = _xgb_regresor(n_arboles)
    modelo.fit(X, y, verbose=False)

    # Forecast recursivo
    historia = d.copy()
    fechas = _fechas_futuras(df_train, horizonte)
    preds = []
    for fecha in fechas:
        ventana = pd.concat(
            [historia.tail(80), pd.DataFrame({'ds': [fecha], 'y': [np.nan]})],
            ignore_index=True
        )
        fila = crear_features_v3(ventana, feriados_set).tail(1)
        assert fila['ds'].iloc[0] == fecha
        z = float(modelo.predict(fila[cols])[0])
        preds.append(z)
        historia = pd.concat([historia, pd.DataFrame({'ds': [fecha], 'y': [z]})],
                             ignore_index=True)

    out = pd.DataFrame({'ds': fechas, 'yhat': _inv(preds, ctx['usar_log'])})
    imp = sorted(zip(cols, modelo.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
    out.attrs['top_features'] = [(c, round(float(v), 3)) for c, v in imp]
    return out


MODELOS = [
    ('Baseline estacional', modelo_baseline),
    ('Prophet', modelo_prophet),
    ('AutoARIMA' if HAS_PMDARIMA else 'SARIMA', modelo_sarima),
    ('XGBoost', modelo_xgboost),
]


# ============================================
# WALK FORWARD VALIDATION (VARIOS CORTES) + ENSEMBLE
# ============================================

def construir_folds(df, horizonte, n_folds):
    """Cortes expansivos: [.. t] entrena, (t, t+h] evalua. Sin solapamiento."""
    n = len(df)
    folds = []
    for k in range(n_folds, 0, -1):
        fin = n - (k - 1) * horizonte
        ini = fin - horizonte
        if ini < 40:  # minimo de historia para entrenar algo util
            continue
        folds.append((df.iloc[:ini].copy(), df.iloc[ini:fin].copy()))
    return folds


def _pesos_desde_errores(errores, max_modelos=3):
    """Pesos ~ 1/error^2 sobre los mejores modelos (y solo si son competitivos)."""
    validos = {k: v for k, v in errores.items() if np.isfinite(v) and v < 100}
    if not validos:
        return {}
    orden = sorted(validos.items(), key=lambda x: x[1])
    mejor = orden[0][1]
    elegidos = [(k, v) for k, v in orden[:max_modelos] if v <= mejor * 1.35]
    pesos = np.array([1.0 / max(v, 0.5) ** 2 for _, v in elegidos])
    pesos = pesos / pesos.sum()
    return {k: float(p) for (k, _), p in zip(elegidos, pesos)}


def walk_forward_validation(df, pais, horizonte=30, n_folds=3, usar_log=False, progreso=None):
    """
    Valida con varios cortes temporales (no uno solo). Devuelve metricas
    promediadas por modelo, el ensemble evaluado de forma honesta (los pesos
    de cada fold vienen de folds anteriores) y los pesos finales.
    """
    df = df.sort_values('ds').reset_index(drop=True)
    n_abiertos = int((df.get('cerrado', pd.Series(0, index=df.index)) == 0).sum())
    if n_abiertos < 45:
        return None, "Pocos datos para validar (< 45 dias con ventas)"

    horizonte = int(np.clip(horizonte, 7, max(14, int(len(df) * 0.25))))
    n_folds = max(1, min(n_folds, max(1, (len(df) - 40) // horizonte)))
    folds = construir_folds(df, horizonte, n_folds)
    if not folds:
        return None, "Serie demasiado corta para validacion temporal"

    anos = sorted(df['ds'].dt.year.unique().tolist())
    anos = anos + [anos[-1] + 1]
    feriados = obtener_feriados(pais, anos)
    ctx = {
        'usar_log': usar_log,
        'feriados': feriados,
        'feriados_set': set(feriados['ds'].dt.normalize()) if feriados is not None else None,
    }

    errores_por_fold = []   # [{modelo: wape}]
    metricas_modelo = {}    # {modelo: [dict metricas]}
    ensemble_metricas = []
    pred_ens_rel = []       # errores relativos del ensemble (para intervalos)
    total = len(folds) * len(MODELOS)
    hecho = 0

    for df_train, df_test in folds:
        df_train_w = _recortar_outliers(df_train)
        mask = (df_test.get('cerrado', 0) == 0).values if 'cerrado' in df_test else np.ones(len(df_test), bool)
        real = df_test['y'].to_numpy(dtype=float)
        preds_fold = {}
        errs_fold = {}

        for nombre, fn in MODELOS:
            hecho += 1
            if progreso:
                progreso(hecho / total, nombre)
            try:
                out = fn(df_train_w, len(df_test), ctx)
            except Exception:
                out = None
            if out is None or out['yhat'].isna().any():
                continue
            yhat = out['yhat'].to_numpy(dtype=float)[:len(df_test)]
            if len(yhat) != len(df_test):
                continue
            if 'cerrado' in df_test:
                yhat = np.where(df_test['cerrado'].values == 1, 0.0, yhat)
            m = calcular_metricas(real, yhat, mask)
            if m['wape'] >= 100:
                continue
            preds_fold[nombre] = yhat
            errs_fold[nombre] = m['wape']
            metricas_modelo.setdefault(nombre, []).append(m)

        if not preds_fold:
            continue

        # Ensemble honesto: pesos calculados con folds ANTERIORES
        historicos = {}
        for e in errores_por_fold:
            for k, v in e.items():
                historicos.setdefault(k, []).append(v)
        pesos_previos = _pesos_desde_errores(
            {k: float(np.mean(v)) for k, v in historicos.items() if k in preds_fold}
        ) if historicos else {}
        if not pesos_previos:
            pesos_previos = {k: 1.0 / len(preds_fold) for k in preds_fold}

        suma = sum(pesos_previos.get(k, 0) for k in preds_fold)
        if suma > 0:
            mezcla = sum(preds_fold[k] * (pesos_previos.get(k, 0) / suma) for k in preds_fold)
            ensemble_metricas.append(calcular_metricas(real, mezcla, mask))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where((real > 0) & mask, mezcla / np.maximum(real, 1e-6), np.nan)
            pred_ens_rel.extend([v for v in rel if np.isfinite(v)])

        errores_por_fold.append(errs_fold)

    if not metricas_modelo:
        return None, "Ningun modelo convergio"

    resumen = {}
    for nombre, lista in metricas_modelo.items():
        resumen[nombre] = {k: round(float(np.mean([m[k] for m in lista])), 2)
                           for k in ('mape', 'wape', 'smape', 'mae')}
        resumen[nombre]['folds'] = len(lista)

    pesos_finales = _pesos_desde_errores({k: v['wape'] for k, v in resumen.items()})

    if ensemble_metricas:
        ens = {k: round(float(np.mean([m[k] for m in ensemble_metricas])), 2)
               for k in ('mape', 'wape', 'smape', 'mae')}
    else:
        mejor = min(resumen.items(), key=lambda x: x[1]['wape'])
        ens = dict(mejor[1])
    ens['folds'] = len(ensemble_metricas) or 1

    # Intervalo empirico a partir de los errores de validacion del ensemble
    if len(pred_ens_rel) >= 20:
        ratios = np.array(pred_ens_rel)
        banda = (float(np.quantile(ratios, 0.9)), float(np.quantile(ratios, 0.1)))
    else:
        banda = (1.25, 0.8)

    return {
        'resumen': resumen,
        'ensemble': ens,
        'pesos': pesos_finales,
        'ctx': ctx,
        'horizonte_validacion': horizonte,
        'n_folds': len(folds),
        'banda': banda,
    }, None


def _forecast_final(df, dias_futuro, ctx, pesos, banda):
    """Reentrena los modelos del ensemble con TODA la serie y combina con los mismos pesos."""
    df_w = _recortar_outliers(df)
    fn_por_nombre = dict(MODELOS)
    salidas, usados = {}, {}
    intervalo_modelo = None

    for nombre, peso in sorted(pesos.items(), key=lambda x: -x[1]):
        fn = fn_por_nombre.get(nombre)
        if fn is None:
            continue
        try:
            out = fn(df_w, dias_futuro, ctx)
        except Exception:
            out = None
        if out is None or out['yhat'].isna().any():
            continue
        salidas[nombre] = out
        usados[nombre] = peso
        if 'yhat_lower' in out and intervalo_modelo is None:
            intervalo_modelo = out

    if not salidas:
        raise RuntimeError("No se pudo generar el pronostico final")

    total = sum(usados.values())
    usados = {k: v / total for k, v in usados.items()}
    fechas = next(iter(salidas.values()))['ds']
    yhat = sum(salidas[k]['yhat'].to_numpy(dtype=float) * w for k, w in usados.items())

    fut = pd.DataFrame({'ds': fechas, 'yhat': np.clip(yhat, 0, None)})
    if 'cerrado' in df:
        cerrados = set(df.loc[df['cerrado'] == 1, 'ds'].dt.dayofweek.unique())
        if cerrados:
            fut.loc[fut['ds'].dt.dayofweek.isin(cerrados), 'yhat'] = 0.0

    q_alto, q_bajo = banda
    fut['yhat_lower'] = np.clip(fut['yhat'] / max(q_alto, 1.01), 0, None)
    fut['yhat_upper'] = fut['yhat'] / min(max(q_bajo, 0.3), 0.99)

    hist = df[['ds', 'y']].copy()
    hist['yhat'] = hist['y']
    hist['yhat_lower'] = hist['y']
    hist['yhat_upper'] = hist['y']
    completo = pd.concat([hist[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fut],
                         ignore_index=True)
    return completo, usados


def analizar_v3(df, pais, dias_futuro, usar_log=False, n_folds=3, progreso=None):
    """
    Pipeline: walk-forward multi-corte -> pesos honestos -> reentreno con toda
    la serie -> forecast del MISMO ensemble que se valido.
    """
    horizonte_val = int(np.clip(dias_futuro, 14, 30))
    val, error = walk_forward_validation(
        df, pais, horizonte=horizonte_val, n_folds=n_folds,
        usar_log=usar_log, progreso=progreso
    )
    if val is None:
        return None, None, error

    try:
        prediccion, pesos_usados = _forecast_final(
            df, dias_futuro, val['ctx'], val['pesos'], val['banda']
        )
    except Exception as e:
        return None, None, str(e)

    nombres = '+'.join(pesos_usados.keys())
    ens = val['ensemble']
    metricas = {
        'modelo_ganador': nombres if len(pesos_usados) > 1 else next(iter(pesos_usados)),
        'MAPE': ens['mape'],
        'WAPE': ens['wape'],
        'sMAPE': ens['smape'],
        'MAE': ens['mae'],
        'Precision': max(0, round(100 - ens['mape'], 2)),
        'folds': val['n_folds'],
        'horizonte_validacion': val['horizonte_validacion'],
        'pesos': {k: round(v, 3) for k, v in pesos_usados.items()},
        'por_modelo': val['resumen'],
        'usar_log': usar_log,
    }
    return prediccion, metricas, None


# ============================================
# ANALISIS Y RECOMENDACIONES
# ============================================

def solo_dias_abiertos(df):
    """Excluye los dias cerrados: sus ceros no son ventas caidas."""
    if 'cerrado' in df.columns:
        return df[df['cerrado'] == 0].reset_index(drop=True)
    return df


def obtener_mejor_dia(df):
    dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves',
                    'Viernes', 'Sabado', 'Domingo']
    df_temp = solo_dias_abiertos(df).copy()
    df_temp['dia'] = df_temp['ds'].dt.dayofweek
    ventas = df_temp.groupby('dia')['y'].mean()
    if ventas.sum() == 0:
        return "No detectado", 0
    idx = ventas.idxmax()
    return dias_nombres[idx], ventas[idx]


def detectar_cambio_tendencia(df, window=14):
    df = solo_dias_abiertos(df)
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
    abiertos = solo_dias_abiertos(df)

    # 1. Tendencia general
    primera = abiertos['y'][:len(abiertos)//2].mean()
    segunda = abiertos['y'][len(abiertos)//2:].mean()
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
    ult = df['y'].tail(7).sum()  # ultimos 7 dias de calendario, comparable con el forecast
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
    abiertos = solo_dias_abiertos(df)
    dias = (df['ds'].max() - df['ds'].min()).days
    pct_zeros = (abiertos['y'] == 0).sum() / max(len(abiertos), 1) * 100
    varianza = abiertos['y'].std() / (abiertos['y'].mean() if abiertos['y'].mean() != 0 else 1)

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
    df_copy['fecha_dt'] = parsear_fechas(df_copy[col_fecha])[0]
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
    df_copy['fecha_dt'] = parsear_fechas(df_copy[col_fecha])[0]
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
modo_log = "Automático"
n_folds = 3

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
        modo_log = st.radio(
            "Suavizar picos extremos (log)",
            ["Automático", "Sí", "No"],
            index=0, horizontal=True,
            help="Comprime los picos. En automático se activa solo si tus ventas son muy asimétricas."
        )
        n_folds = st.slider(
            "Cortes de validación", 1, 4, 3,
            help="Más cortes = medición del error más confiable, pero más lento."
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

    st.caption(
        f"Fechas interpretadas como **{info_val['formato_fecha']}** "
        f"({info_val['pct_no_parseadas']}% ilegibles) · "
        f"{info_val['dias']} días con ventas"
        + (f" · días cerrados detectados: {len(info_val['dias_cerrados'])}" if info_val['dias_cerrados'] else "")
    )

    usar_log = sugerir_log(df_limpio) if modo_log == "Automático" else (modo_log == "Sí")

    barra = st.progress(0.0, text="Validando modelos...")

    def _progreso(pct, nombre):
        barra.progress(min(pct, 1.0), text=f"Validando {nombre}...")

    prediccion, metricas, error = analizar_v3(
        df_limpio, pais, dias_futuro,
        usar_log=usar_log, n_folds=n_folds, progreso=_progreso
    )
    barra.empty()

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

        st.caption(
            f"Validación temporal: {metricas['folds']} corte(s) de "
            f"{metricas['horizonte_validacion']} días · "
            f"transformación log: {'sí' if metricas['usar_log'] else 'no'}"
        )
        st.caption(f"WAPE {metricas['WAPE']}% · sMAPE {metricas['sMAPE']}% · MAE ${metricas['MAE']:,.0f}")

        st.markdown("**Comparación de modelos (promedio de todos los cortes)**")
        tabla = pd.DataFrame([
            {'Modelo': nom, 'MAPE %': m['mape'], 'WAPE %': m['wape'],
             'MAE': m['mae'], 'Cortes': m['folds'],
             'Peso en ensemble': metricas['pesos'].get(nom, 0)}
            for nom, m in sorted(metricas['por_modelo'].items(), key=lambda x: x[1]['wape'])
        ])
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.caption(
            "MAPE = error porcentual medio sobre días con ventas. WAPE = error total / ventas totales "
            "(no explota en días de venta baja) y es la métrica con la que se eligen y ponderan los modelos. "
            "Objetivo ideal: ≤15% con datos estables."
        )

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
