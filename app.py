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
# FUNCIONES DE PREDICCIÓN
# ============================================

def limpiar_datos(df, col_fecha, col_ventas):
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

    rango = pd.DataFrame({
        'ds': pd.date_range(
            start=df_limpio['ds'].min(),
            end=df_limpio['ds'].max(),
            freq='D'
        )
    })
    df_limpio = rango.merge(df_limpio, on='ds', how='left')
    df_limpio['y'] = df_limpio['y'].fillna(0)

    Q1 = df_limpio['y'].quantile(0.25)
    Q3 = df_limpio['y'].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 3 * IQR
    df_limpio['y'] = df_limpio['y'].clip(upper=limite_superior)

    return df_limpio


def obtener_feriados(pais, años):
    paises_map = {
        'Bolivia': holidays.Bolivia,
        'México': holidays.Mexico,
        'Argentina': holidays.Argentina,
        'Colombia': holidays.Colombia,
        'Perú': holidays.Peru,
        'Chile': holidays.Chile,
        'España': holidays.Spain,
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
            for año in años:
                f = clase_feriados(years=año)
                for fecha, nombre in f.items():
                    feriados_lista.append({
                        'holiday': nombre,
                        'ds': pd.Timestamp(fecha)
                    })
            return pd.DataFrame(feriados_lista)
    except:
        pass
    return None


def correr_prophet(df_train, df_test, feriados=None):
    modelo = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=len(df_train) > 365,
        daily_seasonality=False,
        interval_width=0.95,
        holidays=feriados
    )
    modelo.fit(df_train)

    futuro = modelo.make_future_dataframe(
        periods=len(df_test), freq='D'
    )
    pred = modelo.predict(futuro)
    pred_test = pred['yhat'].tail(len(df_test)).values
    real_test = df_test['y'].values

    mask = real_test > (real_test.mean() * 0.1)
    # Solo calcula error en días con ventas relevantes
    if mask.sum() > 0:
        mape = np.mean(
            np.abs((real_test[mask] - pred_test[mask]) / real_test[mask])
        ) * 100
    else:
        mape = np.mean(
            np.abs((real_test - pred_test) / (real_test + 1))
        ) * 100
    mae = mean_absolute_error(real_test, pred_test)

    return {'nombre': 'Prophet', 'mape': round(mape, 2),
            'mae': round(mae, 2)}


def correr_arima(df_train, df_test):
    try:
        modelo = ARIMA(df_train['y'], order=(1, 1, 1))
        resultado = modelo.fit()
        pred = resultado.forecast(steps=len(df_test))
        real_test = df_test['y'].values
        pred_values = pred.values

        mask = real_test > (real_test.mean() * 0.1)
        if mask.sum() > 0:
            mape = np.mean(
                np.abs((real_test[mask] - pred_values[mask]) / real_test[mask])
            ) * 100
        else:
            mape = np.mean(
                np.abs((real_test - pred_values) / (real_test + 1))
            ) * 100
        mae = mean_absolute_error(real_test, pred_values)

        return {'nombre': 'ARIMA', 'mape': round(mape, 2),
                'mae': round(mae, 2)}
    except:
        return {'nombre': 'ARIMA', 'mape': 999, 'mae': 999}


def analizar(df, pais, dias_futuro):
    años = df['ds'].dt.year.unique().tolist()
    años += [max(años) + 1]
    feriados = obtener_feriados(pais, años)

    split = int(len(df) * 0.70)
    df_train = df[:split]
    df_test = df[split:]

    res_prophet = correr_prophet(df_train, df_test, feriados)
    res_arima = correr_arima(df_train, df_test)

    resultados = [res_prophet, res_arima]
    resultados.sort(key=lambda x: x['mape'])
    ganador = resultados[0]

    if ganador['nombre'] == 'Prophet':
        modelo_final = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=len(df) > 365,
            daily_seasonality=False,
            interval_width=0.95,
            holidays=feriados
        )
        modelo_final.fit(df)
        futuro = modelo_final.make_future_dataframe(
            periods=dias_futuro, freq='D'
        )
        prediccion = modelo_final.predict(futuro)
    else:
        modelo_final = ARIMA(df['y'], order=(1, 1, 1))
        res_final = modelo_final.fit()
        pred_hist = res_final.predict(start=0, end=len(df) - 1)
        pred_fut = res_final.forecast(steps=dias_futuro)
        fechas_fut = pd.date_range(
            start=df['ds'].max(),
            periods=dias_futuro + 1, freq='D'
        )[1:]
        std = df['y'].std()
        prediccion = pd.DataFrame({
            'ds': pd.concat([df['ds'], pd.Series(fechas_fut)]).reset_index(drop=True),
            'yhat': pd.concat([pred_hist, pred_fut]).reset_index(drop=True),
            'trend': pd.concat([pred_hist, pred_fut]).reset_index(drop=True)
        })
        prediccion['yhat_upper'] = prediccion['yhat'] + 1.96 * std
        prediccion['yhat_lower'] = prediccion['yhat'] - 1.96 * std
        prediccion['weekly'] = 0

    metricas = {
        'modelo_ganador': ganador['nombre'],
        'MAPE': ganador['mape'],
        'MAE': ganador['mae'],
        'Precision': round(100 - ganador['mape'], 2),
        'prophet_mape': res_prophet['mape'],
        'arima_mape': res_arima['mape']
    }

    return prediccion, metricas


def obtener_mejor_dia(prediccion):
    dias_nombres = ['Lunes', 'Martes', 'Miércoles',
                     'Jueves', 'Viernes', 'Sábado', 'Domingo']
    weekly = prediccion[['ds', 'weekly']].copy()
    weekly['dia'] = weekly['ds'].dt.dayofweek
    weekly_avg = weekly.groupby('dia')['weekly'].mean()
    if weekly_avg.abs().sum() == 0:
        return "No detectado"
    return dias_nombres[weekly_avg.idxmax()]


def generar_recomendaciones(df, prediccion, metricas, mejor_dia):
    recomendaciones = []

    primera_mitad = df['y'][:len(df)//2].mean()
    segunda_mitad = df['y'][len(df)//2:].mean()

    if primera_mitad > 0:
        if segunda_mitad > primera_mitad * 1.1:
            cambio = (segunda_mitad / primera_mitad - 1) * 100
            recomendaciones.append({
                'tipo': 'positivo', 'icono': '📈',
                'texto': f'Tu negocio creció {cambio:.1f}% en el último período. '
                         f'Considera aumentar tu inventario.'
            })
        elif segunda_mitad < primera_mitad * 0.9:
            cambio = (1 - segunda_mitad / primera_mitad) * 100
            recomendaciones.append({
                'tipo': 'alerta', 'icono': '📉',
                'texto': f'Tus ventas bajaron {cambio:.1f}%. '
                         f'Revisa qué cambió en este período.'
            })

    if mejor_dia != "No detectado":
        recomendaciones.append({
            'tipo': 'info', 'icono': '⭐',
            'texto': f'{mejor_dia} es tu mejor día. Considera promociones '
                     f'especiales otros días para equilibrar.'
        })

    if metricas['Precision'] > 90:
        recomendaciones.append({
            'tipo': 'positivo', 'icono': '✅',
            'texto': f'Predicción muy confiable ({metricas["Precision"]}%). '
                     f'Puedes planificar compras con seguridad.'
        })
    elif metricas['Precision'] > 75:
        recomendaciones.append({
            'tipo': 'info', 'icono': 'ℹ️',
            'texto': f'Predicción confiable ({metricas["Precision"]}%). '
                     f'Usa como guía, mantén margen de error.'
        })
    else:
        recomendaciones.append({
            'tipo': 'alerta', 'icono': '⚠️',
            'texto': f'Predicción con {metricas["Precision"]}% de confianza. '
                     f'Tus datos son irregulares, usa con precaución.'
        })

    proxima_semana = prediccion[
        prediccion['ds'] > df['ds'].max()
    ]['yhat'].head(7).sum()
    ultima_semana = df['y'].tail(7).sum()

    if ultima_semana > 0 and proxima_semana > ultima_semana * 1.05:
        recomendaciones.append({
            'tipo': 'positivo', 'icono': '🚀',
            'texto': 'Se espera un aumento de ventas la próxima semana. '
                     'Prepara inventario adicional.'
        })

    return recomendaciones


# ============================================
# INTERFAZ STREAMLIT
# ============================================

st.set_page_config(page_title="Predictive Analytics Tool", layout="wide")

st.title("📊 Predictive Analytics Tool")
st.write("Herramienta de predicción con IA")
st.divider()

col1, col2 = st.columns(2)

with col1:
    nombre_negocio = st.text_input(
        "🏪 Nombre de tu negocio:",
        placeholder="Ej: Tienda El Alto"
    )

with col2:
    pais = st.selectbox(
        "🌎 Selecciona tu país:",
        ["Bolivia", "México", "Argentina", "Colombia", "Perú",
         "Chile", "España", "USA", "Brasil", "Ecuador",
         "Venezuela", "Paraguay"]
    )

dias_futuro = st.slider(
    "📅 ¿Cuántos días quieres predecir?",
    min_value=7, max_value=90, value=30, step=7
)

st.divider()

archivo = st.file_uploader(
    "📁 Sube tu archivo CSV de ventas:",
    type=['csv'],
    help="Debe tener columnas de fecha y ventas"
)

df_raw = None
if archivo is not None:
    df_raw = pd.read_csv(archivo, encoding='latin1')
    st.success(f"✅ Archivo cargado: {len(df_raw)} filas")
    st.write("**Vista previa:**")
    st.dataframe(df_raw.head(3))

    st.write("**Selecciona las columnas correctas:**")
    col_a, col_b = st.columns(2)
    with col_a:
        col_fecha = st.selectbox("📅 Columna de FECHA:", df_raw.columns.tolist())
    with col_b:
        col_ventas = st.selectbox("💰 Columna de VENTAS:", df_raw.columns.tolist())
else:
    st.info("👆 Sube tu CSV para continuar")

st.divider()

if st.button("🚀 ANALIZAR CON IA", type="primary", use_container_width=True):
    if archivo is None:
        st.error("❌ Primero sube tu archivo CSV")
    elif not nombre_negocio:
        st.error("❌ Escribe el nombre de tu negocio")
    else:
        with st.spinner("🔄 Analizando tus datos con IA..."):
            df_limpio = limpiar_datos(df_raw, col_fecha, col_ventas)
            st.success(f"✅ Datos limpios: {len(df_limpio)} registros")

            prediccion, metricas = analizar(df_limpio, pais, dias_futuro)
            mejor_dia = obtener_mejor_dia(prediccion)
            recomendaciones = generar_recomendaciones(
                df_limpio, prediccion, metricas, mejor_dia
            )

        st.divider()
        st.subheader("📊 Resultados del Análisis")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Precisión", f"{metricas['Precision']}%")
        m2.metric("📉 Error (MAPE)", f"{metricas['MAPE']}%")
        m3.metric("🤖 Modelo", metricas['modelo_ganador'])
        m4.metric("📅 Días analizados", len(df_limpio))
        
        if metricas['Precision'] < 0:
            st.error(
                "⚠️ Tus datos son muy irregulares (picos extremos y días sin ventas). "
                "Este tipo de modelo funciona mejor con ventas diarias más estables "
                "(tiendas, restaurantes, retail). Contáctanos para un análisis personalizado."
            )
            
        st.write("**Comparación de modelos:**")
        col_p, col_a = st.columns(2)
        col_p.metric(
            "Prophet MAPE", f"{metricas['prophet_mape']}%",
            delta="ganador" if metricas['modelo_ganador'] == 'Prophet' else None
        )
        col_a.metric(
            "ARIMA MAPE", f"{metricas['arima_mape']}%",
            delta="ganador" if metricas['modelo_ganador'] == 'ARIMA' else None
        )

        st.divider()
        st.subheader("💡 Recomendaciones para tu negocio")
        for rec in recomendaciones:
            if rec['tipo'] == 'positivo':
                st.success(f"{rec['icono']} {rec['texto']}")
            elif rec['tipo'] == 'alerta':
                st.warning(f"{rec['icono']} {rec['texto']}")
            else:
                st.info(f"{rec['icono']} {rec['texto']}")

        st.divider()
        st.write("**Predicción de ventas:**")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_limpio['ds'], y=df_limpio['y'],
            name='Ventas reales',
            line=dict(color='#2196F3', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=prediccion['ds'], y=prediccion['yhat'],
            name=f'Predicción ({metricas["modelo_ganador"]})',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([prediccion['ds'], prediccion['ds'][::-1]]),
            y=pd.concat([prediccion['yhat_upper'], prediccion['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,87,34,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo 95%'
        ))

        fecha_hoy = df_limpio['ds'].max()
        fig.add_shape(
            type='line', x0=fecha_hoy, x1=fecha_hoy, y0=0, y1=1,
            yref='paper', line=dict(color='green', width=2, dash='dot')
        )
        fig.add_annotation(
            x=fecha_hoy, y=1, yref='paper', text='Hoy',
            showarrow=False, yshift=10, font=dict(color='green', size=12)
        )

        fig.update_layout(
            template='plotly_dark', height=450, hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Próximos días:**")
        pred_futuras = prediccion[
            prediccion['ds'] > df_limpio['ds'].max()
        ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(dias_futuro)
        pred_futuras.columns = ['Fecha', 'Predicción', 'Mínimo', 'Máximo']
        pred_futuras['Fecha'] = pred_futuras['Fecha'].dt.strftime('%d/%m/%Y')
        pred_futuras = pred_futuras.round(2)
        st.dataframe(pred_futuras, use_container_width=True)

        st.success(f"✅ Análisis completado para {nombre_negocio}")
