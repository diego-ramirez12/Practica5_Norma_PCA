import pandas as pd
import matplotlib.pyplot as plt


def grafica_pearson(datos: pd.DataFrame):
    """Genera la matriz de correlación del conjunto de entrenamiento."""

    # Seleccionar los datos anteriores a 2010
    entrenamiento = datos.loc[
        datos["Datetime"] < "2010-01-01"
    ]

    # Seleccionar las siete variables numéricas
    numericas = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    # Calcular las correlaciones
    matriz_pearson = entrenamiento[numericas].corr(
        method="pearson"
    )

    # Código de visualización del compañero
    fig, ax = plt.subplots(figsize=(10, 8))
    imagen = ax.imshow(
        matriz_pearson,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(numericas)),
        labels=numericas,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(numericas)),
        labels=numericas
    )

    ax.set_title(
        "Correlación de Pearson — entrenamiento (antes de 2010)"
    )

    for i in range(len(numericas)):
        for j in range(len(numericas)):
            valor = matriz_pearson.iat[i, j]

            ax.text(
                j, i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color="white" if abs(valor) >= 0.65 else "black",
                fontsize=9
            )

    fig.colorbar(
        imagen,
        ax=ax,
        label="Coeficiente r de Pearson",
        shrink=0.8
    )

    fig.tight_layout()

    return fig
