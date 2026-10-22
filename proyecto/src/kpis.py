
"""Cuatro KPIs para describir el dataset."""

from pathlib import Path
import pandas as pd


def cargar_datos_limpios(ruta: str | Path) -> pd.DataFrame:
    """Carga el dataset previamente procesado."""
    return pd.read_csv(
        ruta,
        sep=";",
        parse_dates=["Datetime"]
    )


def calcular_kpis(
    datos: pd.DataFrame,
    inicio="2006-12-16",
    fin="2009-12-31"
) -> dict:

    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)
    desde = max(inicio, datos["Datetime"].min())
    hasta = min(
        fin + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
        datos["Datetime"].max()
    )

    periodo = datos.loc[
        datos["Datetime"].between(desde, hasta)
    ]
    observados = len(periodo)
    esperados = int(
        (hasta - desde) / pd.Timedelta(minutes=1)
    ) + 1

    return {
        "consumo_total_kwh":
            periodo["Global_active_power"].sum() / 60,

        "voltaje_promedio_v":
            periodo["Voltage"].mean(),

        "intensidad_promedio_a":
            periodo["Global_intensity"].mean(),

        "datos_completos_pct":
            observados / esperados * 100,

        "minutos_observados": observados,
        "minutos_esperados": esperados,
        "minutos_faltantes": esperados - observados,
        "inicio_efectivo": desde.isoformat(),
        "fin_efectivo": hasta.isoformat()
    }
