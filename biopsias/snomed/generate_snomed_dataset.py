import pandas as pd
from typing import Union, Iterable
from pathlib import Path
import json


def generate_snomed_dataset(
    path_all_snomed_data: Union[Path, str],
    x_columns: Iterable[str] = ["Diagnostico"],
    y_columns: Iterable[str] = ["ConceptID_Morphologic", "CodTopografico"],
) -> pd.DataFrame:
    """Generate the dataset necessary to train models to solve snomed problems.

    Parameters
    ----------
    path_all_snomed_data : Union[Path, str]
        Path to all the snomed data.
    x_columns : Iterable[str], optional
        Columns used as input of the model, by default ["Diagnostico"]
    y_columns : Iterable[str], optional
        Columns used as output of the model, by default ["ConceptID_Morphologic", "CodTopografico"]

    Returns
    -------
    pd.DataFrame
        Dataset necessary to train models to solve snomed problems.
    """
    all_data_df = (
        pd.read_csv(path_all_snomed_data, low_memory=False)
        .set_index("Estudio")
        .dropna(subset=[x_columns[0]])
    )
    all_data_df = all_data_df[x_columns + y_columns]

    if len(x_columns) > 1:
        all_data_df["x_join"] = all_data_df.agg(
            lambda x: " ".join(
                [f"\n{col}: {x[col]}" for col in x_columns if isinstance(x[col], str)]
            ),
            axis=1,
        )

    renames = {x_col: f"x_{x_col}" for x_col in x_columns}
    renames.update({y_col: f"y_{y_col}" for y_col in y_columns})

    all_data_df = all_data_df.rename(columns=renames)

    return all_data_df


def get_topk_XY(
    dataset: pd.DataFrame,
    x_column: str,
    y_column: str,
    topk: int = -1,
    new_label: str = "otro",
    translate_labels_path: Union[str, Path] = None,
) -> pd.DataFrame:
    """Generate inputs and outputs for specific models, using top k labels.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to extract information from.
    x_column : str
        Column used as input of the model.
    y_column : str
        Column used as output of the model.
    topk : int, optional
        Top labels used to train.
        If a label is not in the top is replaced by ´new_label´.
        If topk is equal to -1 only the classes with a single element are replaced.
    new_label : str, optional
        Replace when label is not on top, by default "otro"
    translate_labels_path : Union[str, Path], optional
        Renames for labels used, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with "x" and "y" columns to train.
    """
    simple_dataset = (
        dataset[[x_column, y_column]]
        .rename(columns={x_column: "x", y_column: "y"})
        .dropna(subset=["x", "y"])
    )

    # Remove multilabel:
    simple_dataset = simple_dataset[simple_dataset["y"].str.contains(",") == False]

    value_counts = simple_dataset["y"].value_counts()

    if topk != -1:
        labels = list(value_counts.index)[:topk]

        simple_dataset["y"] = simple_dataset["y"].apply(
            lambda y: new_label if y not in labels else y
        )
    else:
        single_values = list(value_counts[value_counts == 1].index)

        simple_dataset["y"] = simple_dataset["y"].apply(
            lambda y: new_label if y in single_values else y
        )

    if translate_labels_path is not None:
        with open(translate_labels_path, "r", encoding="utf8") as f:
            translate_dict = json.load(f)["codes"]

            simple_dataset["y"] = simple_dataset["y"].apply(
                lambda y: translate_dict[y] if y is not new_label else y
            )

    return simple_dataset
