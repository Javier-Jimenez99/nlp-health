from typing import Optional
import jsonlines
from pathlib import Path
import pandas as pd
import typer
from biopsias.form.generate_dataset import generate_dataset
from biopsias import config
from biopsias.form.measure_rules import define_nlp


def write_json_line(
    row: pd.DataFrame, feature_column: str, writer: jsonlines.jsonlines.Writer, nlp
):
    """Writes a json line with the desired prodigy format of a `pd.DataFrame` row.

    Parameters
    ----------
    row : pd.DataFrame
        Row to write data from.
    feature_column : str
        Name of the column selected.
    writer : jsonlines.jsonlines.Writer
        Writer that writes the json line.
    """
    text = row[feature_column]
    spans = []
    doc = nlp(text)
    for ent in doc.ents:
        span = {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        spans.append(span)

    writer.write({"text": text, "estudio": row.name, "spans": spans})


def main(
    feature_column: Optional[str] = "x_join",
):
    """Creates a new data file in a format ready to be annotated from prodigy.

    Parameters
    ----------
    feature_column: Optional[str]]
        The column to be used in the annotation.

    """

    dataset = generate_dataset(config.all_biopsias_path, config.parsed_forms_path)
    nlp = define_nlp()

    with jsonlines.open(
        Path("data/clean/prodigy/measure_ner.jsonl"), mode="w"
    ) as writer:
        # Escribe una línea del dataset para que pueda ser anotado en prodigy
        dataset.apply(
            write_json_line,
            feature_column=feature_column,
            writer=writer,
            nlp=nlp,
            axis=1,
        )


if __name__ == "__main__":
    typer.run(main)
