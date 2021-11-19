from typing import Optional
import jsonlines
from pathlib import Path
import pandas as pd
import typer
from biopsias.form.generate_dataset import generate_dataset
from biopsias import config


def write_json_line(
    row: pd.DataFrame, feature_column: str, writer: jsonlines.jsonlines.Writer
):
    """Writes a json line with the desired data of a `pd.DataFrame` row.

    Parameters
    ----------
    row : pd.DataFrame
        Row to write data from.
    feature_column : str
        Name of the column selected.
    writer : jsonlines.jsonlines.Writer
        Writer that writes the json line.
    """
    writer.write(
        {
            "text": row[feature_column],
            "estudio": row.name,
        }
    )


def main():
    """Creates a new data file in a format ready to be annotated from prodigy."""

    dataset = generate_dataset(config.all_biopsias_path, config.parsed_forms_path)

    with jsonlines.open(
        Path("data/clean/prodigy/all_data_prodigy.jsonl"), mode="w"
    ) as writer:
        # Escribe una línea del dataset para que pueda ser anotado en prodigy
        dataset.apply(write_json_line, feature_column="x_join", writer=writer, axis=1)

    with jsonlines.open(
        Path("data/clean/prodigy/nan_histological_type_prodigy.jsonl"), mode="w"
    ) as writer:
        # Escribe una línea del dataset para que pueda ser anotado en prodigy
        dataset[dataset["y_histological_type"].isna()].apply(
            write_json_line, feature_column="x_diagnostico", writer=writer, axis=1
        )


if __name__ == "__main__":
    typer.run(main)
