import json
import re
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import typer
from biopsias.config import hardcoded_invalid_studies
from biopsias.form.interventions import (
    exist_mandatory_intervention,
    load_manual_labeled_interventions,
)
from biopsias.form.parse_form import parse_form
from biopsias import config
from biopsias.form.parse_form_2 import parse_form_2
from joblib import Parallel, delayed
from tqdm import tqdm


class StudyFormConflicts(Exception):
    pass


# Funciones como esta, la de pasar de JSON a DataFrame y la de eliminar los null
# o vacíos deberían de meterse en un archivo utils para JSONs o dict.
# No lo he hecho aún porque creo que encajarían directamente en Doraemon.
def len_dict_recursive(dic: dict) -> int:
    """Count the number of elements inside a dictionary with several levels.

    Parameters
    ----------
    dic : dict
        Dictionary to count elements.

    Returns
    -------
    int
        Number of elements inside the dictionary.
    """
    size = 0

    for value in dic.values():
        if isinstance(value, dict):
            size += len_dict_recursive(value)
        else:
            size += 1

    return size


def assert_no_conflict(*dicts: Iterable[dict]) -> None:
    """Raise an exception if the dictionaries has different values between them in fields with the same key.
    Comparisons are made at the same level for all dictionaries, so it wont take in count keys in other levels.

    Parameters
    ----------
    dicts: Iterable[dict]
        Dictionary to count elements.

    Raises
    ----------
    AssertionError
        Dictionaries has different values between them in fields with the same key.
    """
    all_keys = set(chain(*(dic.keys() for dic in dicts)))
    array_dics = np.array(dicts)
    for key in all_keys:
        one_hot = [(key in dic.keys()) for dic in dicts]
        options = [dic[key] for dic in array_dics[one_hot]]
        if sum(one_hot) > 1:
            # If all values are dictionaries, move on to next level
            if all(isinstance(option, dict) for option in options):
                assert_no_conflict(*options)
            else:
                # check if no all items are the same, they can be dict, so cannot create set
                cond = any(option != options[0] for option in options[1:])
                if cond:
                    # there are inconsistencies
                    assert_string = f"Inconsistencies found in field {key.__repr__()}:"
                    for i, option in enumerate(options, 1):
                        assert_string += f"\n{i:2d})\t{option.__repr__()}"

                    raise AssertionError(assert_string)
    return None


def join_parse_study_rows(
    study_group: pd.DataFrame,
    manual_annotated_labels: pd.DataFrame,
    study: Optional[str] = None,
    forms_col: Iterable[str] = ("Texto del complementario", "Notas (Muestra)"),
    output_col: str = "Formulario",
) -> Union[None, tuple[dict, dict[str, dict[str, Union[str, dict[str, str]]]]]]:
    """Join rows that belongs to one specific study and
    parse the respective form.

    Parameters
    ----------
    study_group : pd.DataFrame
        Dataframe that contains the rows that belongs to one specific study.
        There can't be more than one form, in separated rows.
    manual_annotated_labels: pd.DataFrame
        Dataframe containing manual annotated interventions texts.
    study: Optional[str]
        It is added to the StudyFormConflicts message.
    forms_col : Iterable[str]
        Columns to search for forms.
    output_col : str
        Form column new name.

    Returns
    -------
    Union[None, tuple[dict, dict[str, dict[str, Union[str, dict[str, str]]]]]]
        If there is a form, joined rows and parsed form are returned.
        If there isn't a form, `None` is returned.

    Raises
    -------
    StudyFormConflicts
        Study found with several conflictive forms.
    """
    dic_aggregated_rows = {}

    columns = study_group.columns

    # parse the form from `forms_col`
    ser_forms = pd.concat([study_group[col] for col in forms_col], axis=0)
    ser_forms = ser_forms.drop_duplicates().dropna()

    # Basic rule to detect if a form is present in the string
    mask = ser_forms.map(lambda x: "tipo de intervención:" in x.lower().strip())
    n_forms = mask.sum()
    ser_forms = ser_forms[mask]

    if n_forms >= 1:
        # Parse the forms
        parsed_forms_series = ser_forms.apply(
            parse_form, manual_annotated_labels=manual_annotated_labels
        )
        parsed_forms_2_series = ser_forms.apply(
            parse_form_2, manual_annotated_labels=manual_annotated_labels
        )

        all_parsed_forms = pd.concat([parsed_forms_series, parsed_forms_2_series])

        try:
            assert_no_conflict(*all_parsed_forms.to_list())
        except AssertionError as a:
            raise StudyFormConflicts(f"Study: {study}\n{a}")

        # we join the information from all parsed forms
        parsed_form = {}
        for form in all_parsed_forms:
            parsed_form.update(form)

        if n_forms > 1:
            # we save the information into the resulting pandas row
            dic_aggregated_rows[output_col] = ""
            for i, form_text in enumerate(ser_forms, 1):
                dic_aggregated_rows[output_col] += f"##ENTRY_{i}##\n{form_text}\n\n"
            dic_aggregated_rows[output_col] = dic_aggregated_rows[output_col].strip()
        else:
            dic_aggregated_rows[output_col] = f"##ENTRY_1##\n{ser_forms.iloc[0]}"

    else:
        parsed_form = None

    # Determine if an study has mandatory interventions
    # "intervention_manual" must be a field of the parsed form and the list of labels must be valid
    dic_aggregated_rows["valid"] = (
        parsed_form is not None
        and (parsed_form.get("intervention_manual") is not None)
        and exist_mandatory_intervention(parsed_form["intervention_manual"])
    )

    # now we parse the information not related to the form and aggregate it
    for col in set(columns) - set(forms_col):
        clean_data = study_group[col].drop_duplicates().dropna()

        if clean_data.size > 1:
            # If there are more than one column with different information join them
            agg_str = ""
            for i, d in enumerate(clean_data, 1):
                agg_str += f"##ENTRY_{i}## {d}\n\n"
            dic_aggregated_rows[col] = agg_str.strip()

        elif clean_data.size == 1:
            # If there is only one value select it
            dic_aggregated_rows[col] = f"##ENTRY_1## {clean_data.iloc[0]}"
        elif clean_data.size == 0:
            # If there is no data add it as NaN
            dic_aggregated_rows[col] = None

    return dic_aggregated_rows, parsed_form


def select_and_parse_form_studies(
    excel_path: Union[str, Path],
    forms_col: Iterable[str] = ("Texto del complementario", "Notas (Muestra)"),
    output_col: str = "Formulario",
    n_jobs: int = -3,
    manual_annotated_labels: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Union[str, dict[str, str]]]]]:
    """Aggregate the rows which belong to the same study into a single row,
    keeping only those studies in which a valid form is present.

    Parameters
    ----------
    excel_path : Union[str, Path]
        Path to excel file to extract raw dataframe.
    forms_col : Iterable[str]
        Columns to search for forms.
    output_col : str
        Form column new name.
    n_jobs : int
        Number of jobs used to process the studies, default 8.
    manual_annotated_labels: pd.Series
        Dataframe containing manual annotated interventions texts, by default None.
        The `index` correspond with the intervention text `values` are the intervetions labels.
        If it is None, the manual annotated labels are extracted inside the function originally from:
        https://docs.google.com/spreadsheets/d/1mAoGChwEOAvush7vz5pxhvBn3oRJ4QFO/edit#gid=662190549
        This functionality is added to allow multiprocessing.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, Union[str, dict[str, str]]]]]
        Dataframe with a new column of forms without duplicated studies and parsed forms.
    """

    # It is used to check the year of the intervention, allowing to find it inside raw data
    year = re.match(r".*Biopsias_HUPM_(?P<year>.*)\.xls", str(excel_path)).group("year")

    df = pd.read_excel(excel_path)

    groupby_df = df.groupby("Estudio")

    if manual_annotated_labels is None:
        manual_annotated_labels = load_manual_labeled_interventions()[["x", "y"]]
        manual_annotated_labels = manual_annotated_labels.set_index(
            "x", verify_integrity=True
        )["y"]

    total_studies = len(groupby_df)
    if n_jobs != 1:
        # Parallelize the processing
        # This Progressbar shows just the queuing of jobs,
        # but the execution least 0.01s, so it's a good estimator.
        studies_processed = Parallel(n_jobs=n_jobs)(
            delayed(join_parse_study_rows)(
                group, manual_annotated_labels, study, forms_col, output_col
            )
            for study, group in tqdm(
                groupby_df,
                total=total_studies,
                desc=f"Biopsias from {year}",
            )
            if study not in hardcoded_invalid_studies
        )
    else:
        print("Running w/o multiprocessing...")
        studies_processed = [
            join_parse_study_rows(
                group, manual_annotated_labels, study, forms_col, output_col
            )
            for study, group in tqdm(
                groupby_df,
                total=total_studies,
                desc=f"Biopsias from {year}",
            )
            if study not in hardcoded_invalid_studies
        ]

    forms_dict = {}
    rows_list = []
    # Join forms and create a list to convert dicts to a Dataframe.
    for processing_study_output in studies_processed:
        dic_aggregated_rows, parsed_form = processing_study_output

        dic_aggregated_rows["year"] = year

        rows_list.append(dic_aggregated_rows)

        if parsed_form is not None:
            forms_dict[dic_aggregated_rows["Estudio"]] = parsed_form

    # Create studies dataframe
    df_out = pd.DataFrame(rows_list)

    return df_out, forms_dict


def process_xls_files(
    folder: Union[str, Path],
    biopsias_save_path: Union[str, Path] = None,
    forms_json_path: Union[str, Path] = None,
    n_jobs: int = -3,
) -> tuple[pd.DataFrame, dict[str, dict[str, Union[str, dict[str, str]]]]]:
    """Process all the "xls" files inside a folder and save the results to "csv", "parquet" and "json" files.
    For each file: aggregate all the rows that belongs to the same study, parse forms
    and select the studies where forms appear.

    Parameters
    ----------
    folder : Union[str, Path]
        Folder to find "xls" files.
    biopsias_save_path : Union[str, Path], optional
        File to save dataframe result as parquet and CSV, by default None.
        This path can't have defined a file extension
        If it is None the dataframe is not saved.
    forms_json_path : Union[str, Path], optional
        File to save dataframe result, by default None.
        If it is None the dataframe is not saved.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, Union[str, dict[str, str]]]]]
        Dataframe with a new column of forms without duplicated studies and parsed forms.
    """
    files = sorted(Path(folder).glob("*.xls"))
    df_list = []
    parsed_forms = {}

    for f in files:
        df, dic = select_and_parse_form_studies(f, n_jobs=n_jobs)

        parsed_forms.update(dic)
        df_list.append(df)

    final_df = pd.concat(df_list).sort_values(["year", "Estudio"], ignore_index=True)

    if biopsias_save_path is not None:
        biopsias_save_path = Path(biopsias_save_path)
        final_df.to_csv(biopsias_save_path.with_suffix(".csv"), sep=";", index=False)
        final_df.to_parquet(biopsias_save_path.with_suffix(".parquet"), index=False)

    if forms_json_path is not None:
        forms_json_path = Path(forms_json_path)
        with open(forms_json_path, "w") as f:
            f.write(json.dumps(parsed_forms, indent=4, ensure_ascii=False))

    return final_df, parsed_forms


def main(n_jobs: int = -3):
    # The extension is deleted with `with_suffix` to save
    # the file in two diferent formats.
    process_xls_files(
        "data/raw/xls_clean",
        Path(config.all_biopsias_path).with_suffix(""),
        Path(config.parsed_forms_path),
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    typer.run(main)
