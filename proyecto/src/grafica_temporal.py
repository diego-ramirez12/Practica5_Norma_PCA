import pandas as pd

FRECUENCIAS = {"Hora": "h", "Día": "D", "Mes": "ME"}


def consumo_por_periodo(
    datos: pd.DataFrame,
    inicio,
    fin,
    agregacion: str = "Día",
) -> pd.DataFrame:
    """Regresa la energía observada (kWh) agrupada por hora, día o mes.
    Cada fila de entrada representa la potencia media en un minuto (kW).
    Los periodos sin registros quedan en NaN, no se interpretan como cero.
    """
    desde = pd.Timestamp(inicio)
    hasta = pd.Timestamp(fin) + pd.Timedelta(days=1)
    periodo = datos.loc[
        (datos["Datetime"] >= desde) & (datos["Datetime"] < hasta),
        ["Datetime", "Global_active_power"],
    ]

    serie = (
        periodo.set_index("Datetime")["Global_active_power"]
        .resample(FRECUENCIAS[agregacion])
        .sum(min_count=1)
        .div(60)
    )
    return serie.rename("consumo_kwh").reset_index()
