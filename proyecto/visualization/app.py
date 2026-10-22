"""Dashboard de consumo eléctrico residencial: KPIs y evolución temporal."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import altair as alt


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO / "src"))

from kpis import cargar_datos_limpios, calcular_kpis 
from grafica_temporal import consumo_por_periodo  
from consumo_zonas import consumo_por_zona, ZONAS
from pearson import grafica_pearson

PROJECT_DIR = Path(__file__).resolve().parents[1]


RUTA_DATOS = PROYECTO / "data" / "data-processed" / "household_power_consumption_clean.csv"

st.set_page_config(page_title="Consumo eléctrico residencial", page_icon="⚡", layout="wide")
st.title("Dashboard de consumo eléctrico residencial")
st.caption("Mediciones por minuto de un hogar en Francia (2006-2010).")


@st.cache_data
def obtener_datos() -> pd.DataFrame:
    return cargar_datos_limpios(RUTA_DATOS)


datos = obtener_datos()
fecha_min = datos["Datetime"].min().date()
fecha_max = datos["Datetime"].max().date()

st.subheader("Periodo de análisis")
col_inicio, col_fin = st.columns(2)
with col_inicio:
    fecha_inicio = st.date_input(
        "Fecha inicial",
        value=pd.Timestamp("2006-12-16").date(),
        min_value=fecha_min,
        max_value=fecha_max,
    )
with col_fin:
    fecha_fin = st.date_input(
        "Fecha final",
        value=pd.Timestamp("2009-12-31").date(),
        min_value=fecha_min,
        max_value=fecha_max,
    )

if fecha_inicio > fecha_fin:
    st.warning("La fecha inicial debe ser anterior o igual a la fecha final.")
    st.stop()

resultados = calcular_kpis(datos, inicio=fecha_inicio, fin=fecha_fin)

st.subheader("Indicadores clave (KPIs)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Consumo energético total", f"{resultados['consumo_total_kwh']:,.2f} kWh")
col2.metric("Voltaje promedio", f"{resultados['voltaje_promedio_v']:,.2f} V")
col3.metric("Intensidad promedio", f"{resultados['intensidad_promedio_a']:,.2f} A")
col4.metric("Datos completos", f"{resultados['datos_completos_pct']:.3f} %")

st.caption(
    "Periodo efectivo: "
    f"{pd.Timestamp(resultados['inicio_efectivo']):%d/%m/%Y %H:%M} – "
    f"{pd.Timestamp(resultados['fin_efectivo']):%d/%m/%Y %H:%M} · "
    f"Minutos observados: {resultados['minutos_observados']:,} / "
    f"{resultados['minutos_esperados']:} "
)

st.subheader("Evolución temporal del consumo")
serie = consumo_por_periodo(datos, fecha_inicio, fecha_fin, "Mes")
st.line_chart(serie, x="Datetime", y="consumo_kwh", x_label="Fecha", y_label="Consumo energético (kWh)")


st.info(
    "El consumo total corresponde a la energía de los minutos observados. "
    "Los intervalos sin mediciones no se imputan y pueden causar una subestimación."
)


st.subheader("Consumo eléctrico por zona")

zonas = consumo_por_zona(datos, fecha_inicio, fecha_fin)

grafica_zonas = alt.Chart(zonas).mark_bar().encode(
    x=alt.X(
        "Zona:N",
        title="Zona del hogar",
        sort=list(ZONAS.values()),
        axis=alt.Axis(labelAngle=0)
    ),
    y=alt.Y(
        "Consumo (kWh):Q",
        title="Consumo energético (kWh)"
    ),
    tooltip=[
        "Zona:N",
        alt.Tooltip("Consumo (kWh):Q", format=",.2f")
    ]
)

st.altair_chart(grafica_zonas, use_container_width=True)

st.caption(
    "Las tres zonas corresponden únicamente a los submedidores del dataset; "
    "no representan la totalidad del consumo del hogar."
)

with st.expander("Ver datos por zona"):
    st.dataframe(
        zonas.style.format({
            "Consumo (kWh)": "{:,.2f}",
            "Participación (%)": "{:.1f}%"
        }),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Correlaciones de Pearson")

fig_pearson = grafica_pearson(datos)

st.pyplot(fig_pearson)
plt.close(fig_pearson)


st.subheader("Análisis de componentes principales (PCA)")

# Carpeta donde se encuentran las gráficas
ruta_figuras = PROJECT_DIR / "notebooks" / "figures"

# Gráfica de varianza explicada
st.image(
    ruta_figuras / "pca_scree_plot.png",
    use_container_width=True
)

# Resultados obtenidos mediante PCA
st.write(
    "Se seleccionaron **5 componentes principales**, "
    "que conservan el **97.23 % de la varianza**."
)

# Proyección bidimensional
st.subheader("Proyección bidimensional PC1–PC2")

st.image(
    ruta_figuras / "pca_proyeccion_2d.png",
    use_container_width=True
)
