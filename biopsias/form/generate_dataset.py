from transformers.utils import logging
from biopsias import config
import re
from pathlib import Path
from typing import Iterable, Union, Optional

import numpy as np
import pandas as pd
from biopsias.form.interventions import load_json_as_df
from biopsias.form.string_matching import clean_name_hard, create_dict_fuzzymatch
from biopsias.form.measure_rules import get_series_entities

import spacy
from logging import warning
import json


def join_X_data(
    biopsias_df: pd.DataFrame,
    x_columns: Iterable[str] = (
        "Datos clínicos",
        "Diagnóstico",
        "Macro",
    ),
) -> pd.Series:
    """Generate dataset input.

    Parameters
    ----------
    biopsias_df : pd.Dataframe
        Dataframe that contains all the information about biopsias diagnosis.
    x_columns : Iterable[str], optional
        Columns to aggregate text from, by default ("Datos clínicos", "Diagnóstico", "Macro")

    Returns
    -------
    pd.Series
        Input data to train a model.
    """

    X = biopsias_df.agg(
        lambda x: " ".join(
            [f"\n{col}: {x[col]}" for col in x_columns if isinstance(x[col], str)]
        ),
        axis=1,
    ).to_frame()

    X = X.set_index(biopsias_df["Estudio"], verify_integrity=True)

    return X.iloc[:, 0]


# LAS Y ESPECIFICAS DE CADA CAMPO PUEDEN ESTAR AQUI O EN UN SCRIPT ESPECIFICO
def generate_Y_intervention(parsed_form_df: pd.DataFrame) -> pd.Series:
    """Generate dataset output for intervention.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput intervention data.
    """
    Y = parsed_form_df["intervention_manual"]

    Y = Y.apply(lambda x: np.nan if x == np.nan or x == ["invalid"] else x)

    return Y.rename("y_intervention")


def generate_Y_is_multiple(parsed_form_df: pd.DataFrame) -> pd.Series:
    """Generate dataset output for is_multiple.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput is_multiple data.
    """
    is_multiple_raw = parsed_form_df["tumor_features.is_multiple"]

    # Clean names
    serie_y = is_multiple_raw.apply(
        lambda x: clean_name_hard(x) if not isinstance(x, float) else np.nan
    )

    Y = pd.Series(np.nan, index=serie_y.index, name="y_is_multiple")
    Y[serie_y.str.match(r"si(\s\()?").fillna(False)] = 1
    Y[serie_y.str.match(r"no(\s\()?").fillna(False)] = 0

    return Y


def generate_Y_caracter(parsed_form_df: pd.DataFrame) -> pd.Series:
    """Generate dataset output for caracter. It is a binary output that determine if it is "infiltrante" or not.
    If "infiltrante" appears the class must be 1, even if "microinvasor" appears.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput caracter data.
    """
    caracter_raw = parsed_form_df["tumor_features.caracter"]

    # Clean names
    serie_y = caracter_raw.apply(
        lambda x: clean_name_hard(x) if not isinstance(x, float) else np.nan
    )

    Y = pd.Series(np.nan, index=serie_y.index, name="y_is_caracter_infiltrante")

    # Is important to mantain this order because if "infiltrante" appears the class must be 1, even if "microinvasor" appears.
    Y[serie_y.str.match(r"microinvasor").fillna(False)] = 0
    Y[serie_y.str.match(r"infiltrant").fillna(False)] = 1

    return Y


def generate_Y_histological_degree(parsed_form_df: pd.DataFrame) -> pd.Series:
    """Generate dataset output for histological_degree. It is a multiclass output represented by interger numbers from 0 to 3.
    If it is 0 means that there is no degree, this happens when it isn't a "carcinoma ductal infiltrante" or the form is not correctly filled.
    These errors can't be prevented because, there is no clear difference between the two possible cases.

    TODO: utilizar el tipo histológico para diferenciar entre las posibles razones de que sea 0

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput histological_degree data.
    """
    histological_degree_raw = parsed_form_df["tumor_features.histological_degree"]

    # Clean names
    serie_y = histological_degree_raw.apply(
        lambda x: clean_name_hard(x) if not isinstance(x, float) else np.nan
    )

    Y_levels = serie_y.str.extract(r"grado\s?(?P<level>i{1,3}).*", flags=re.IGNORECASE)[
        "level"
    ].str.strip()

    Y = Y_levels.map(lambda x: x.count("i") if x is not np.nan else 0)

    return Y.rename("y_histological_degree")


def _get_histological_type_class(
    histological_type_text: str,
    histological_type_classes: dict[str, Iterable[str]],
    all_choices: Iterable[str],
    no_match_value: Optional[str] = np.nan,
) -> str:
    """Obtain the class that correspond with the histological type text.

    Parameters
    ----------
    histological_type_text : str
        Text of the histological type.
    histological_type_classes : dict[str, Iterable[str]]
        Classes and the choices for each one.
    all_choices : Iterable[str]
        Choices of all classes grouped.
    no_match_value: Optional[str]
        Value to represent no matched histological types, by default `np.nan`.

    Returns
    -------
    str
        Class of the histological type text.
    """
    if histological_type_text is not np.nan:
        match_dic = create_dict_fuzzymatch(
            [histological_type_text], all_choices, score_cutoff=85
        )
        if histological_type_text in match_dic:
            match_text = match_dic[histological_type_text].match

            for clas, choices in histological_type_classes.items():
                if match_text in choices:
                    return clas
        else:
            # Returns `no_match_value` when there is no match between histological_type_text and the choices
            # It's caused by an error on the text.
            return no_match_value
    else:
        return np.nan


def generate_Y_histological_type(
    parsed_form_df: pd.DataFrame, no_match_value: Optional[str] = np.nan
) -> pd.Series:
    """Generate dataset output for histological_type. It is a multiclass output represented by 13 classes:
        - "carcinoma ductal infiltrante"
        - "carcinoma lobulillar infiltrante"
        - "carcinoma tubular"
        - "carcinoma cribiforme"
        - "carcinoma metaplásico"
        - "carcinoma medular"
        - "carcinoma con diferenciacion apocrina"
        - "carcinoma adenoide quistico"
        - "carcinoma mucoepidermoide"
        - "carcinoma polimorfo"
        - "carcinoma mucinoso"
        - "carcinoma papilar invasivo"
        - "carcinoma inflamatorio"

    If the histological type text don't match any class `no_match_value` is set as output.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.
    no_match_value: Optional[str]
        Value to represent no matched histological types, by default `np.nan`.

    Returns
    -------
    pd.Series
        Ouput histological_type data.
    """
    histological_type_raw = parsed_form_df["tumor_features.histological_type"]

    # Clean names
    serie_y = histological_type_raw.apply(
        lambda x: clean_name_hard(x) if not isinstance(x, float) else np.nan
    )

    histological_type_classes = {
        "carcinoma ductal infiltrante": [
            "carcinoma ductal infiltrante",
            "carcinoma ductal infiltrante in situ",
            "cdi",
            "carcinoma ductal",  # TODO: confirmar que esta opción es válida.
            # Es decir, que cuando aparece ductal obligatoriamente es CDI.
        ],
        "carcinoma lobulillar infiltrante": [
            "carcinoma lobulillar infiltrante",
            "carcinoma lobulillar infiltrante in situ",
        ],
        "carcinoma tubular": ["carcinoma tubular"],
        "carcinoma cribiforme": ["carcinoma cribiforme"],
        "carcinoma metaplásico": ["carcinoma metaplásico"],
        "carcinoma medular": [
            "carcinoma medular",
            "carcinoma medular atipico",
            "carcinoma invasivo con caracteristicas medulares",
        ],
        "carcinoma con diferenciacion apocrina": [
            "carcinoma con diferenciación apocrina"
        ],
        "carcinoma adenoide quistico": ["carcinoma adenoide quistico"],
        "carcinoma mucoepidermoide": ["carcinoma mucoepidermoide"],
        "carcinoma polimorfo": ["carcinoma polimorfo"],
        "carcinoma mucinoso": [
            "carcinoma mucinoso",
            "carcinoma con diferenciacion de celulas en anillo de sello",
        ],
        "carcinoma papilar invasivo": [
            "carcinoma papilar invasivo",
            "carcinoma micropapilar invasivo",
            "carcinoma papilar infiltrante",
            "carcinoma micropapilar infiltrante",
            "carcinoma papilar",
        ],
        "carcinoma inflamatorio": ["carcinoma inflamatorio"],
    }

    # Convert "ca " in "carcinoma"
    serie_y = serie_y.str.replace(
        r"^ca\s", "carcinoma ", regex=True, flags=re.IGNORECASE
    )

    all_choices = []
    for choices in histological_type_classes.values():
        all_choices.extend(choices)

    Y = serie_y.apply(
        _get_histological_type_class,
        no_match_value=no_match_value,
        all_choices=all_choices,
        histological_type_classes=histological_type_classes,
    )

    return Y.rename("y_histological_type")


def generate_Y_is_carcinoma_ductal_infiltrante(
    parsed_form_df: pd.DataFrame,
) -> pd.Series:
    """Generate dataset output to determine if the study involves any kind of "carcinoma ductal infiltrante" or not.
    It is a multiclass output.

    If the histological type text don't match any choice `False` is set.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput is_carcinoma_ductal_infiltrante data.
    """

    histological_type_series = generate_Y_histological_type(parsed_form_df)
    # TODO: confirmar qué otros tipos de carcinomas son un subconjunto de CDI.
    # Por ejemplo: carcinoma mucinoso
    Y = histological_type_series == "carcinoma ductal infiltrante"
    Y[histological_type_series.isna()] = np.nan

    return Y.rename("y_is_cdi")


def generate_Y_histological_degree_with_cdi(
    parsed_form_df: pd.DataFrame,
) -> pd.Series:
    """Generate dataset output to determine the histological degree only over the cases that are CDI.
    It is a multiclass output.

    Parameters
    ----------
    parsed_form_df: pd.DataFrame
        Dataframe containing all the fields of the forms parsed.

    Returns
    -------
    pd.Series
        Ouput histological degree with CDI data.
    """
    is_cdi = generate_Y_is_carcinoma_ductal_infiltrante(parsed_form_df)
    cdi_df = parsed_form_df[is_cdi == 1]
    cdi_df_no_nan = cdi_df.dropna(subset=["tumor_features.histological_degree"])

    return generate_Y_histological_degree(cdi_df_no_nan).rename(
        "y_histological_degree_cdi"
    )


def process_measure_text(measure_text: str) -> float:
    """Extract the measure number from text.

    Parameters
    ----------
    measure_text : str
        Text that contains the measure.

    Returns
    -------
    float
        Number extracted from the measure.
    """
    measure_cleaned = (
        measure_text.lower()
        .strip()
        .replace(" ", "")
        .replace(",", ".")
        .replace("'", ".")
    )

    measures = []
    number_pattern = r"\d+(\.\d+)?"
    x_pattern = r"[(por)xy]"
    match = re.match(
        rf"(?i)(?P<number1>{number_pattern})({x_pattern}(?P<number2>{number_pattern}))?({x_pattern}(?P<number3>{number_pattern}))?(?P<unit>[mc]?m)",
        measure_cleaned,
    )

    measures = []
    if match is not None:
        unit = match.group("unit")

        for group_name in ("number1", "number2", "number3"):
            number = match.group(group_name)

            if number is not None:
                number = float(number)
                if unit == "mm":
                    number *= 10
                elif unit == "m":
                    number /= 100
                elif not unit == "cm":
                    raise ValueError(f"Unit {unit} not recognized.")

                measures.append(number)

    return measures


def _get_measure_representation(
    row: pd.DataFrame, tokenizer: spacy.tokenizer
) -> pd.DataFrame:
    """Extract the text tokenized and the NER labels used to train a model.

    Parameters
    ----------
    row : pd.DataFrame
        Row of the dataframe that cointains all the data related with the size measures.
    tokenizer: spacy.Tokenizer
        Tokenizer of the `spacy.Language` used to process measurements.

    Returns
    -------
    pd.DataFrame
        `pd.DataFrame` that contains two columns, one for
    """
    text_tokenized = [ent.text for ent in tokenizer(row["diagnostico"])]
    try:
        measure_index = row["ent_text"].index(row["form_size"])

        start = row["ent_start_token"][measure_index]
        end = row["ent_end_token"][measure_index]

        # First is filled by "O", that represents no entity
        tokens_labels = ["O"] * len(text_tokenized)

        # Mark the start of the entity with "B-..."
        tokens_labels[start] = "B-TUMOR_SIZE"

        # Mark the continuation of the entity with "I-...", if it is not only one token.
        if not (end - 1 == start):
            tokens_labels[start + 1 : end] = ["I-TUMOR_SIZE"] * (end - (start + 1))

        return text_tokenized, tokens_labels

    except ValueError:
        return text_tokenized, np.nan


def generate_Y_automatic_size(
    parsed_form_df: pd.DataFrame, text_series: pd.Series
) -> tuple[Union[None, list[str]]]:
    """Get the text tokenized and the NER labels used to train a model.

    Parameters
    ----------
    parsed_form_df : pd.DataFrame
        Dataframe containing all the fields of the forms parsed.
    text_series : pd.Series
        Series containing all the diagnosis texts.

    Returns
    -------
    pd.DataFrame
        Dataframe containing diagnosis text, form size and entities values.
    """
    # Get entities from texts and generate a df containing all entities values,
    # "diagnostico" and the size extraxted from the form.
    measures_extracted_df, nlp = get_series_entities(text_series, return_nlp=True)
    measures_extracted_df["diagnostico"] = text_series
    measures_extracted_df["form_size"] = parsed_form_df["tumor_features.size"]

    # Delete NaN values, because it wont cause a match
    measures_extracted_df = measures_extracted_df.dropna(
        subset=("ent_text", "form_size")
    )

    # Parse size measures to allow comparison
    measures_extracted_df["ent_text"] = measures_extracted_df["ent_text"].apply(
        lambda x: [process_measure_text(text) for text in x]
    )
    measures_extracted_df["form_size"] = measures_extracted_df["form_size"].apply(
        process_measure_text
    )

    # Position of the desire size measure
    (
        measures_extracted_df["y_size_text_tokenized"],
        measures_extracted_df["y_size_representation"],
    ) = zip(
        *measures_extracted_df.apply(
            _get_measure_representation, tokenizer=nlp.tokenizer, axis=1
        )
    )

    return measures_extracted_df


def generate_dataset(
    all_biopsias_path: Union[str, Path],
    parsed_forms_path: Union[str, Path],
    x_columns: Iterable[str] = ("Datos clínicos", "Diagnóstico", "Macro", "Micro"),
) -> pd.DataFrame:
    """Generate input and output interventions dataset.

    Parameters
    ----------
    all_biopsias_path : Union[str, Path]
        Path to parquet file that contains all the information about biopsias diagnosis.
    parsed_forms_path : Union[str, Path]
        Path to JSON file that contains all the forms from the biopsias diagnosis parsed.
    x_columns : Iterable[str], optional
        Columns to aggregate text from, by default ("Datos clínicos", "Diagnóstico", "Macro", "Micro")

    Returns
    -------
    pd.DataFrame
        Generate dataset input and output to train a model.
    """

    all_biopsias_path = Path(all_biopsias_path)
    parsed_forms_path = Path(parsed_forms_path)

    all_biopsias_df = pd.read_parquet(all_biopsias_path)

    # Get only valid biopsias
    biopsias_df = all_biopsias_df[all_biopsias_df["valid"]]

    # Select the desired columns, rename them and set studies as index.
    renames = {col: f"x_{clean_name_hard(col).replace(' ','_')}" for col in x_columns}
    dataset = biopsias_df[list(x_columns) + ["Estudio"]].rename(columns=renames)
    dataset = dataset.set_index("Estudio")

    # Calculate X data
    dataset["x_join"] = join_X_data(biopsias_df, x_columns)

    # Calculate Y data
    parsed_form_df = load_json_as_df(parsed_forms_path)

    Y_functions = {
        generate_Y_intervention,
        generate_Y_is_multiple,
        generate_Y_caracter,
        generate_Y_histological_degree,
        generate_Y_histological_type,
        generate_Y_is_carcinoma_ductal_infiltrante,
        generate_Y_histological_degree_with_cdi,
    }

    Y_columns_list = []
    for function in Y_functions:
        Y_columns_list.append(function(parsed_form_df))

    Y_columns_list.append(
        generate_Y_automatic_size(parsed_form_df, dataset["x_diagnostico"])[
            [
                "y_size_text_tokenized",
                "y_size_representation",
            ]
        ]
    )

    dataset = pd.concat([dataset] + Y_columns_list, axis=1)

    # Delete invalid rows where interventions are NaN
    valid_dataset = dataset.dropna(subset=["y_intervention"])

    return valid_dataset


def generate_histological_type_with_annotations(
    top_k: int,
    annotations_path: Union[
        str, Path
    ] = "data/clean/prodigy/annotations/histological_type_annotations.jsonl",
) -> pd.DataFrame:
    """Generate a dataset including automatic and manual annotated histological types.

    Parameters
    ----------
    top_k : int
        Determine the number of classes that will be used.
        The rest of the classes will be joined in a class "other".
    annotations_path : Union[str, Path]
        Path to the JSON file that contains the annotations.

    Returns
    -------
    pd.DataFrame
        Dataframe that contains the data necesasy to train an histological type model.
    """

    annotations = pd.read_json(
        annotations_path,
        lines=True,
    )

    # Manual annotated data is cleaned and prepared to match with the labels of the automatic annotation
    annotated_data = (
        annotations[["text", "histological_type_classification"]]
        .rename(columns={"text": "x", "histological_type_classification": "y"})
        .dropna(subset=["y"])
    )
    annotated_data["y"] = annotated_data["y"].map(
        lambda x: x.split(".")[1].lower().strip()
    )

    # Calculation of automatic annotation
    automatic_dataset = (
        generate_dataset(
            config.all_biopsias_path,
            config.parsed_forms_path,
        )[["x_diagnostico", "y_histological_type"]]
        .rename(columns={"x_diagnostico": "x", "y_histological_type": "y"})
        .dropna(subset=["y"])
    )

    # Join all the data
    all_data = pd.concat([annotated_data, automatic_dataset], ignore_index=True)

    # Clean text
    all_data["x"] = (
        all_data["x"]
        .str.replace(r"\#\#entry_\d*\#\#", "", regex=True, flags=re.IGNORECASE)
        .str.replace(r"\d+((\,|\.)\d+)?", "", regex=True, flags=re.IGNORECASE)
    )

    # If there are too few elemtens from one class that class is eliminated and added to a new class "others"
    value_counts = all_data["y"].value_counts()

    topk = list(value_counts.index)[:top_k]

    for ind in value_counts.index:
        if ind not in topk:
            all_data["y"][all_data["y"] == ind] = "otro"

    return all_data


def generate_annotations_tumor_size_dataset(
    annotations_path: Union[Path, str], section: str = None
) -> pd.DataFrame:
    """Generate the dataset used to train a model for tumor size extraction.

    Parameters
    ----------
    annotations_path : Union[Path, str]
        Path to tumor size annotations.
    section : str, optional
        Section that will be used to train the model, by default None.
        If it is None, the extire text is used.

    Returns
    -------
    pd.DataFrame
        Dataframe that contains:
        - "x_text": raw text used to train
        - "y_tokens": text tokenized
        - "y_labels": label of each token
    """
    sections = ["Datos", "Diagnóstico", "Macro", "Micro"]
    annotations_list = []

    with open(annotations_path, "r") as f:
        lines = f.readlines()

    for l in lines:
        annotation_dict = json.loads(l)

        x_text = annotation_dict["text"]
        y_tokens = [t["text"] for t in annotation_dict["tokens"]]
        y_labels = ["O"] * len(y_tokens)

        for span in annotation_dict["spans"]:
            y_labels[span["token_start"]] = "B-TUMOR_SIZE"

            if span["token_start"] < span["token_end"]:
                y_labels[span["token_start"] + 1 : span["token_end"] + 1] = [
                    "I-TUMOR_SIZE"
                ] * (span["token_end"] - span["token_start"])

        if section is None:
            annotations_list.append(
                {"x_text": x_text, "y_tokens": y_tokens, "y_labels": y_labels}
            )
        else:
            section_index = sections.index(section)
            try:
                start_section = y_tokens.index(section)

                if section_index == len(sections) - 1:
                    end_section = -1
                else:
                    end_section = y_tokens.index(sections[section_index + 1])

                annotations_list.append(
                    {
                        "x_text": " ".join(y_tokens[start_section + 1 : end_section]),
                        "y_tokens": y_tokens[start_section + 1 : end_section],
                        "y_labels": y_labels[start_section + 1 : end_section],
                    }
                )
            except ValueError:
                # warning(f"Section {section} not found")
                pass

    return pd.DataFrame(annotations_list)


if __name__ == "__main__":
    dataset = generate_dataset(
        Path(config.all_biopsias_path),
        Path(config.parsed_forms_path),
    )

    print(dataset)
