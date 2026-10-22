import pandas as pd


ZONAS = {
    "Sub_metering_1": "Cocina",
    "Sub_metering_2": "Lavandería",
    "Sub_metering_3": "Agua caliente y aire acondicionado",
}


def consumo_por_zona(datos: pd.DataFrame, inicio, fin) -> pd.DataFrame:
    """Suma la energía medida por cada submedidor durante el periodo (kWh)."""
    desde = pd.Timestamp(inicio)
    hasta = pd.Timestamp(fin) + pd.Timedelta(days=1)
    periodo = datos.loc[(datos["Datetime"] >= desde) & (datos["Datetime"] < hasta)]

    consumo = periodo[list(ZONAS)].sum() / 1000
    resultado = pd.DataFrame({
        "Zona": list(ZONAS.values()),
        "Consumo (kWh)": consumo.to_numpy(),
    })
    total = resultado["Consumo (kWh)"].sum()
    resultado["Participación (%)"] = (
        resultado["Consumo (kWh)"] / total * 100 if total > 0 else 0.0
    )
    return resultado