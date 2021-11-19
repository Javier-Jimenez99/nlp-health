from pathlib import Path
from typing import Iterable, Union

import typer
import pandas as pd
import numpy as np
from biopsias.form.generate_dataset import generate_dataset
from biopsias import config


def get_value_counts(
    all_biopsias_path: Union[str, Path] = "data/clean/Biopsias_HUPM_all.parquet",
    parsed_forms_path: Union[str, Path] = "data/clean/parsed_forms_raw.json",
):
    """Calculate relative and absolute value counts for the desired fields of the dataset.
    Parameters
    ----------
    all_biopsias_path : Union[str, Path]
        Path to parquet file that contains all the information about biopsias diagnosis.
    parsed_forms_path : Union[str, Path]
        Path to JSON file that contains all the forms from the biopsias diagnosis parsed.
    """
    all_biopsias_path = Path(all_biopsias_path)
    parsed_forms_path = Path(parsed_forms_path)

    dataset = generate_dataset(
        all_biopsias_path,
        parsed_forms_path,
    )

    print(f"Número de estudios en el dataset: {len(dataset)}")

    fields = [c for c in dataset.columns if "y" in c]

    # De los textos tokenizados no tienen sentido obtener el value counts.
    # Las labels van a estar muy desbalanceadas hacia la clase 'O' y el value counts tampoco da mucha información.
    fields.remove("y_size_text_tokenized")
    fields.remove("y_size_representation")

    for f in fields:
        if f == "y_intervention":
            # Flatten classes column
            classes = pd.Series(np.concatenate(dataset[f]))
        else:
            classes = dataset[f]

        result_df = classes.value_counts(dropna=False).to_frame(name="Absoluto")
        result_df["Porcentaje"] = (
            100 * result_df["Absoluto"] / result_df["Absoluto"].sum()
        )

        if np.nan in result_df.index:
            # Remove NaN from df
            result_df_no_nan = result_df.copy().drop(np.nan, axis=0)

            result_df["Porcentaje_no_NaN"] = (
                100 * result_df_no_nan["Absoluto"] / result_df_no_nan["Absoluto"].sum()
            )

        print(f"\n {f.capitalize()}:")
        print(result_df)


def main():
    get_value_counts(Path(config.all_biopsias_path), Path(config.parsed_forms_path))


if __name__ == "__main__":
    typer.run(main)
