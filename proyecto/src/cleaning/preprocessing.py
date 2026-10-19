import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "data-raw" / "household_power_consumption.txt"
DEFAULT_CLEAN_PATH = (
    PROJECT_DIR / "data" / "data-processed" / "household_power_consumption_clean.csv"
)

EXPECTED_COLUMNS = (
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)
NUMERIC_COLUMNS = EXPECTED_COLUMNS[2:]


def load_data(file_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Lee el TXT original sin modificarlo ni eliminar o imputar registros.

    Las mediciones se leen como float64 y Date/Time como texto.
    Si se indica una ruta relativa, se interpreta desde el directorio actual.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {file_path}")

    # Validar el encabezado antes de cargar todas las mediciones.
    header = pd.read_csv(file_path, sep=";", encoding="utf-8", nrows=0)
    if tuple(header.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Las columnas del archivo no coinciden con las esperadas. "
            f"Esperadas: {list(EXPECTED_COLUMNS)}. "
            f"Encontradas: {list(header.columns)}."
        )

    return pd.read_csv(
        file_path,
        sep=";",
        decimal=".",
        encoding="utf-8",
        na_values=["?", ""],
        keep_default_na=False,
        dtype={
            "Date": "string",
            "Time": "string",
            **{column: "float64" for column in NUMERIC_COLUMNS},
        },
        low_memory=False,
    )


def inspect_data(data: pd.DataFrame) -> dict:
    """Calcla el diagnóstico inicial sin modificar el DataFrame recibido.

    Los duplicados se cuentan después de su primera aparición. El rango y los
    duplicados temporales se calculan únicamente con fechas y horas válidas.
    La conversión temporal es auxiliar: Date y Time conservan sus tipos originales.
    """
    missing = data.isna()
    column_summary = pd.DataFrame(
        {
            "tipo": data.dtypes.astype(str),
            "faltantes": missing.sum(),
            "porcentaje_faltantes": missing.mean() * 100,
        }
    )
    timestamps = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    valid_timestamps = timestamps.dropna()

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "column_summary": column_summary,
        "rows_with_missing_values": int(missing.any(axis=1).sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_timestamps": int(valid_timestamps.duplicated().sum()),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "start": valid_timestamps.min(),
        "end": valid_timestamps.max(),
        "statistics": data[list(NUMERIC_COLUMNS)].describe().T,
    }


def print_inspection(inspection: dict) -> None:
    """Muestra las dimensiones, la calidad inicial y los estadísticos básicos."""
    print(f"Dimensiones: {inspection['rows']:,} filas y {inspection['columns']} columnas.")
    print("\nTipos y valores faltantes por columna (porcentaje sobre las filas):")
    print(inspection["column_summary"].to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nFilas con al menos un faltante: {inspection['rows_with_missing_values']:,}")
    print(
        "Filas duplicadas completas, después de la primera: "
        f"{inspection['duplicate_rows']:,}"
    )
    print(
        "Marcas temporales repetidas, después de la primera: "
        f"{inspection['duplicate_timestamps']:,}"
    )
    print(f"Fechas/horas faltantes o inválidas: {inspection['invalid_timestamps']:,}")
    print(f"Rango temporal válido: {inspection['start']} a {inspection['end']}")
    print("\nEstadísticas descriptivas (excluyen faltantes; std muestral):")
    print(inspection["statistics"].to_string(float_format=lambda x: f"{x:.6f}"))




# ================================================ Preprocesamiento ================================================

# 1. Conversión de tipos
def convert_types(data: pd.DataFrame) -> pd.DataFrame:
    """Devulve una copia con Datetime y siete mediciones float64, ordenada.

    Rechaza fechas/horas inválidas, números no convertibles y valores infinitos.
    Conserva los faltantes originales y todas las filas, incluidos duplicados.
    """
    datetime_text = (
        data["Date"].astype("string") + " " + data["Time"].astype("string")
    )
    timestamps = pd.to_datetime(
        datetime_text,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    valid_format = datetime_text.str.fullmatch(
        r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}",
        na=False,
    )
    invalid_timestamps = int((timestamps.isna() | ~valid_format).sum())
    if invalid_timestamps:
        raise ValueError(
            f"Se encontraron {invalid_timestamps:,} fechas/horas faltantes o inválidas. "
            "Se requiere día/mes/año hh:mm:ss, con día y mes de uno o dos dígitos."
        )

    converted = data[list(NUMERIC_COLUMNS)].copy()
    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce").astype("float64")
        invalid = data[column].notna() & numeric.isna()
        infinite = numeric.notna() & ~np.isfinite(numeric)
        invalid_count = int((invalid | infinite).sum())
        if invalid_count:
            raise ValueError(
                f"La columna {column} contiene {invalid_count:,} valores "
                "no convertibles o no finitos."
            )
        converted[column] = numeric

    converted.insert(0, "Datetime", timestamps.astype("datetime64[ns]"))
    return converted.sort_values("Datetime", kind="stable").reset_index(drop=True)


# 2. Eliminación de duplicados
def deduplicate_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Conserva la primera fila identica y devuelve los datos y su resumen.

    Recibe el resultado de convert_types(). Una fila se identifica mediante
    Datetime y las siete mediciones, incluyendo la posición de sus faltantes.
    Si un instante tiene mediciones distintas, lanza un error para su revisión.
    El DataFrame recibido permanece intacto.
    """
    expected_columns = ["Datetime", *NUMERIC_COLUMNS]
    if list(data.columns) != expected_columns:
        raise ValueError("La deduplicación requiere las columnas de convert_types().")
    if (
        not pd.api.types.is_datetime64_any_dtype(data["Datetime"])
        or data["Datetime"].isna().any()
    ):
        raise ValueError("Datetime debe tener tipo temporal y no contener faltantes.")

    deduplicated = data.drop_duplicates(keep="first")
    conflicting_rows = deduplicated["Datetime"].duplicated(keep=False)
    if conflicting_rows.any():
        conflicting_times = deduplicated.loc[conflicting_rows, "Datetime"].nunique()
        raise ValueError(
            f"Se encontraron {conflicting_times:,} instantes con mediciones distintas "
            f"({int(conflicting_rows.sum()):,} filas tras retirar duplicados idénticos). "
            "Es necesario revisar estos conflictos antes de continuar."
        )

    summary = {
        "rows_before": len(data),
        "rows_after": len(deduplicated),
        "duplicate_rows_removed": len(data) - len(deduplicated),
        "conflicting_timestamps": 0,
    }
    return deduplicated.reset_index(drop=True), summary


# 3. Tratamiento de faltantes
def handle_missing_values(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retira filas sin mediciones y resume los huecos entre filas conservadas.

    Recibe los datos convertidos, ordenados y deduplicados. Rechaza faltantes
    parciales para revisar su tratamiento. Conserva las fechas y los valores
    observados, sin imputar ni modificar el DataFrame recibido.
    """
    if list(data.columns) != ["Datetime", *NUMERIC_COLUMNS]:
        raise ValueError("El tratamiento de faltantes requiere el esquema convertido.")
    timestamps = data["Datetime"]
    if (
        not pd.api.types.is_datetime64_any_dtype(timestamps)
        or timestamps.isna().any()
        or not timestamps.is_monotonic_increasing
        or timestamps.duplicated().any()
    ):
        raise ValueError("Datetime debe ser temporal, válido, único y estar ordenado.")

    minute = pd.Timedelta(minutes=1)
    intervals = timestamps.diff().dropna()
    if (intervals % minute).ne(pd.Timedelta(0)).any():
        raise ValueError("Los intervalos temporales deben ser múltiplos de un minuto.")

    missing = data[list(NUMERIC_COLUMNS)].isna()
    all_missing = missing.all(axis=1)
    partial_missing = missing.any(axis=1) & ~all_missing
    partial_count = int(partial_missing.sum())
    if partial_count:
        raise ValueError(
            f"Se encontraron {partial_count:,} filas con faltantes parciales. "
            "Es necesario revisar la estrategia antes de continuar."
        )

    cleaned = data.loc[~all_missing].reset_index(drop=True)
    remaining_intervals = cleaned["Datetime"].diff().dropna()
    gaps = remaining_intervals[remaining_intervals > minute]
    missing_minutes = gaps // minute - 1
    summary = {
        "rows_before": len(data),
        "rows_removed": int(all_missing.sum()),
        "rows_after": len(cleaned),
        "partial_missing_rows": partial_count,
        "remaining_missing_values": int(cleaned.isna().sum().sum()),
        "temporal_gaps": len(gaps),
        "missing_minutes_between_records": int(missing_minutes.sum()),
        "longest_gap_missing_minutes": int(missing_minutes.max()) if len(gaps) else 0,
    }
    return cleaned, summary


# 4. Validación y guardado del dataset limpio
def validate_clean_data(data: pd.DataFrame) -> dict:
    """Valida los datos limpios y devuelve su resumen."""
    if list(data.columns) != ["Datetime", *NUMERIC_COLUMNS]:
        raise ValueError("El dataset limpio debe contener Datetime y siete mediciones.")

    missing_count = int(data.isna().sum().sum())
    if missing_count:
        raise ValueError(f"El dataset limpio contiene {missing_count:,} faltantes.")

    timestamps = data["Datetime"]
    # Si Datetime es único, tampoco puede haber filas completas duplicadas.
    if not timestamps.is_unique:
        raise ValueError("El dataset limpio contiene marcas temporales repetidas.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("El dataset limpio debe estar ordenado cronológicamente.")
    minute = pd.Timedelta(minutes=1)
    intervals = timestamps.diff().dropna()
    if (intervals % minute).ne(pd.Timedelta(0)).any():
        raise ValueError("Los intervalos temporales deben ser múltiplos de un minuto.")

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": missing_count,
        "duplicate_rows": 0,
        "duplicate_timestamps": 0,
        "start": timestamps.iloc[0],
        "end": timestamps.iloc[-1],
    }


def load_clean_data(file_path: str | Path = DEFAULT_CLEAN_PATH) -> pd.DataFrame:
    """Lee y valida el CSV limpio, recuperando sus tipos y precisión."""
    data = pd.read_csv(
        file_path,
        sep=";",
        dtype={"Datetime": "string", **{c: "float64" for c in NUMERIC_COLUMNS}},
        float_precision="round_trip",
    )
    data["Datetime"] = pd.to_datetime(
        data["Datetime"], format="ISO8601"
    ).astype("datetime64[ns]")
    validate_clean_data(data)
    return data


def save_clean_data(
    data: pd.DataFrame,
    file_path: str | Path = DEFAULT_CLEAN_PATH,
    source_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[Path, dict]:
    """Valida y guarda el CSV tras comprobar su lectura, protegiendo el original."""
    output_path = Path(file_path).resolve()
    raw_directory = (PROJECT_DIR / "data" / "data-raw").resolve()
    if output_path == Path(source_path).resolve() or output_path.is_relative_to(raw_directory):
        raise ValueError("El destino no puede sobrescribir el archivo fuente ni data-raw.")
    
    validation = validate_clean_data(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / output_path.name
        data.to_csv(
            temporary_path,
            index=False,
            sep=";",
        )
        restored = load_clean_data(temporary_path)
        pd.testing.assert_frame_equal(
            data.reset_index(drop=True), restored, check_exact=True
        )
        temporary_path.replace(output_path)
    return output_path, validation


def main() -> None:
    """Ejecuta la carga, inspección, limpieza, validación y guardado del dataset."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al TXT original; por defecto se usa data/data-raw/ del proyecto.",
    )
    args = parser.parse_args()
    data = load_data(args.file_path)

    print(f"Archivo: {args.file_path.resolve()}")
    initial_inspection = inspect_data(data)
    print_inspection(initial_inspection)

    converted = convert_types(data)
    print("\nResultado de la conversión de tipos:")
    print(f"Dimensiones: {len(converted):,} filas y {len(converted.columns)} columnas.")
    print(converted.dtypes.to_string())
    print(f"Orden cronológico: {converted['Datetime'].is_monotonic_increasing}")
    missing_rows = int(converted[list(NUMERIC_COLUMNS)].isna().any(axis=1).sum())
    print(f"Filas con mediciones faltantes conservadas: {missing_rows:,}")

    deduplicated, summary = deduplicate_data(converted)
    print("\nResultado de la deduplicación:")
    print(f"Filas antes: {summary['rows_before']:,}")
    print(f"Duplicados idénticos eliminados: {summary['duplicate_rows_removed']:,}")
    print(f"Filas después: {summary['rows_after']:,}")
    print(f"Instantes con mediciones distintas: {summary['conflicting_timestamps']:,}")
    missing_rows = int(deduplicated[list(NUMERIC_COLUMNS)].isna().any(axis=1).sum())
    print(f"Filas con mediciones faltantes conservadas: {missing_rows:,}")

    cleaned, missing_summary = handle_missing_values(deduplicated)
    print("\nResultado del tratamiento de faltantes:")
    print(f"Filas antes: {missing_summary['rows_before']:,}")
    print(f"Filas sin ninguna medición retiradas: {missing_summary['rows_removed']:,}")
    print(f"Filas con faltantes parciales: {missing_summary['partial_missing_rows']:,}")
    print(f"Filas después: {missing_summary['rows_after']:,}")
    print(f"Valores faltantes restantes: {missing_summary['remaining_missing_values']:,}")
    print(f"Huecos entre registros conservados: {missing_summary['temporal_gaps']:,}")
    print(
        "Minutos ausentes entre registros conservados: "
        f"{missing_summary['missing_minutes_between_records']:,}"
    )
    print(
        "Minutos ausentes en el hueco más largo: "
        f"{missing_summary['longest_gap_missing_minutes']:,}"
    )
    print(f"Orden cronológico: {cleaned['Datetime'].is_monotonic_increasing}")

    output_path, validation = save_clean_data(cleaned, source_path=args.file_path)
    print("\nValidación final y comparación con el estado inicial:")
    print(f"Filas: {initial_inspection['rows']:,} -> {validation['rows']:,}")
    print(f"Columnas: {initial_inspection['columns']} -> {validation['columns']}")
    initial_missing = int(initial_inspection["column_summary"]["faltantes"].sum())
    print(f"Valores faltantes: {initial_missing:,} -> {validation['missing_values']:,}")
    print(
        f"Filas duplicadas: {initial_inspection['duplicate_rows']:,} "
        f"-> {validation['duplicate_rows']:,}"
    )
    print(f"Marcas temporales repetidas finales: {validation['duplicate_timestamps']:,}")
    print(f"Rango temporal final: {validation['start']} a {validation['end']}")
    print("Tipos, valores finitos, orden temporal y lectura exacta del CSV: correctos.")
    print(f"Dataset limpio guardado en: {output_path}")


if __name__ == "__main__":
    main()
