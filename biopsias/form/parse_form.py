from __future__ import annotations

from typing import Optional, Union, Iterable
import re
import copy
import json
import warnings
import logging

from biopsias.form.interventions import get_labels, load_manual_labeled_interventions
from biopsias.form.string_matching import clean_name_hard

import pandas as pd


def get_line(
    lines: Iterable[str],
    field: str = "- caracter",
    index: int = 0,
    replaces: Iterable[str] = ("- caracter",),
) -> str:
    """Get the data of a desired field, when there is the only one in a line.

    Parameters
    ----------
    lines : Iterable[str]
        Text divided by lines.
    field : str, optional
        Field to extract the data, by default "- caracter"
    index : int, optional
        Displacement of the data (in lines) once the field was found, by default 0.
        If it is 0 the data is in the same line of the field title, if it is 1 is in the following line.
    replaces : Iterable[str], optional
        Strings to delete to the data found, by default ("- caracter")

    Returns
    -------
    str
        Data from an specific field cleaned.
    """
    result = None

    for i, l in enumerate(lines):
        if l.find(field) != -1:
            result = lines[i + index]

            for c in replaces:
                result = re.sub(c, "", result)

            result = result.strip()

            break

    return result


def parse_tumor_features(lines: Iterable[str]) -> dict[str, Union[str, dict[str, str]]]:
    """Parse the section called "CARACTERÍSTICAS DEL TUMOR".

    Parameters
    ----------
    lines: Iterable[str]
        Text of the section splited by line.

    Returns
    -------
    dict[str, Union[str, dict[str, str]]]
        Each field of the section parsed in JSON format.
    """

    result_dict = {}

    # SEARCH size and location
    for line in lines:
        if line.find("- tamaño:") != -1:
            if line.find("localización:") != -1:
                line = line.replace("- tamaño:", "").split("localización:")

                result_dict["size"] = line[0].strip()
                result_dict["location"] = line[1].strip()

            else:
                result_dict["size"] = line.replace("- tamaño:", "").strip()

            break

    # SEARCH multiple
    found = False
    for line in lines:
        if line.find("- si es múltiple:") != -1:
            result_dict["multiple"] = {}
            match = re.search(
                r"nº de tumores:(?P<n_tumors>.*)localización:(?P<location>.*)(tamaño del mayor:(?P<biggest_size>.*)|$)",
                line,
            )

            if match is not None:
                result_dict["multiple"]["n_tumors"] = match.group("n_tumors").strip()

                if match.group("location") is not None:
                    result_dict["multiple"]["location"] = match.group(
                        "location"
                    ).strip()

                if match.group("biggest_size") is not None:
                    result_dict["multiple"]["biggest_size"] = match.group(
                        "biggest_size"
                    ).strip()
                    break

                found = True

        if found:
            match = re.search(
                r"tamaño del mayor:(?P<biggest_size>.*)$",
                line,
            )

            if match is not None:
                if match.group("biggest_size") is not None:
                    result_dict["multiple"]["biggest_size"] = match.group(
                        "biggest_size"
                    ).strip()

                break

    search_dict = {
        "caracter": {
            "field": "- caracter",
            "index": 0,
            "replaces": ["- caracter", ":"],
        },
        "is_multiple": {
            "field": "- múltiple",
            "index": 0,
            "replaces": ["- múltiple", ":"],
        },
        "margin": {
            "field": "infiltración del margen quirúrgico / márgenes de la biopsia",
            "index": 1,
            "replaces": [],
        },
        "histological_type": {
            "field": "- tipo histológico",
            "index": 0,
            "replaces": ["- tipo histológico", ":"],
        },
        "histological_degree": {
            "field": "- grado histológico de bloom y richardson  (solo cdi)",
            "index": 1,
            "replaces": [r"\*", ":"],
        },
        "tubule_formation": {
            "field": "· formación de túbulos",
            "index": 1,
            "replaces": [r"\·", ":"],
        },
        "nuclear_degree": {
            "field": "· grado nuclear",
            "index": 1,
            "replaces": [r"\·", ":"],
        },
        "mythosical_degree": {
            "field": "· grado mitósico",
            "index": 1,
            "replaces": [r"\·", ":"],
        },
        "vascular_invasion": {
            "field": "- invasión vascular peritumoral",
            "index": 0,
            "replaces": ["- invasión vascular peritumoral", ":"],
        },
        "invasion_front": {
            "field": "- frente de invasión",
            "index": 0,
            "replaces": ["- frente de invasión", ":"],
        },
        "tumor_necrosis": {
            "field": "- necrosis tumoral  (comp. infiltrante)",
            "index": 0,
            "replaces": [r"- necrosis tumoral  \(comp. infiltrante\)", ":"],
        },
        "lymphoplasmcitary_response": {
            "field": "- respuesta linfoplasmocitaria estromal",
            "index": 0,
            "replaces": ["- respuesta linfoplasmocitaria estromal", ":"],
        },
        "perinueral_infiltration": {
            "field": "- infiltración perineural",
            "index": 0,
            "replaces": ["- infiltración perineural", ":"],
        },
        "skin_infiltration": {
            "field": "- infiltración de la piel",
            "index": 0,
            "replaces": ["- infiltración de la piel", ":"],
        },
        "vascular_infiltration": {
            "field": "- infiltración vascular in dermis",
            "index": 0,
            "replaces": ["- infiltración vascular in dermis", ":"],
        },
        "nipple_infiltration": {
            "field": "- infiltración del pezón",
            "index": 0,
            "replaces": ["- infiltración del pezón", ":"],
        },
        "muscle_wall_infiltration": {
            "field": "- infiltración pared muscular",
            "index": 0,
            "replaces": ["- infiltración pared muscular", ":"],
        },
        "insitu_component": {
            "field": "- componente in situ",
            "index": 0,
            "replaces": ["- componente in situ", ":"],
        },
        "nuclear_degree_insitu": {
            "field": "- grado nuclear del componente in situ",
            "index": 0,
            "replaces": ["- grado nuclear del componente in situ", ":"],
        },
        "architectural_patern_insitu": {
            "field": "- patrón arquitectural del componente in situ",
            "index": 0,
            "replaces": ["- patrón arquitectural del componente in situ", ":"],
        },
        "comedo_necrosis": {
            "field": "- necrosis tipo comedo",
            "index": 0,
            "replaces": ["- necrosis tipo comedo", ":"],
        },
    }

    for term, term_dict in search_dict.items():
        result_dict[term] = get_line(lines, **term_dict)

    return result_dict


def parse_parenquima_features(lines: Iterable[str]) -> dict[str, str]:
    """Parse the section called "CARACTERÍSTICAS DEL PARÉNQUIMA MAMARIO NO TUMORAL".

    Parameters
    ----------
    lines: Iterable[str]
        Text of the section splited by line.

    Returns
    -------
    dict[str, str]
        Each field of the section parsed in JSON format.
    """
    # Need to delete the first line
    if lines[0] == ":":
        lines.pop(0)

    result_dict = {}

    if lines:
        if (
            lines[0].find("- calcificaciones:") == -1
            and lines[0].find("- mastopatía no proliferativa:") == -1
        ):
            result_dict["type"] = lines[0].replace("-", "").strip()

        search_dict = {
            "calcifications": {
                "field": "- calcificaciones:",
                "index": 0,
                "replaces": ["- calcificaciones:"],
            },
            "mastopathy": {
                "field": "- mastopatía no proliferativa:",
                "index": 0,
                "replaces": ["- mastopatía no proliferativa:"],
            },
        }

        for term, term_dict in search_dict.items():
            result_dict[term] = get_line(lines, **term_dict)

    return result_dict


def parse_ganglion_features(lines: Iterable[str]) -> dict[str, str]:
    """Parse the section called "DATOS DE LOS GANGLIOS AXILARES".

    Parameters
    ----------
    lines: Iterable[str]
        Text of the section splited by line.

    Returns
    -------
    dict[str, str]
        Each field of the section parsed in JSON format.
    """

    result_dict = {}

    search_dict = {
        "isolated_ganglions": {
            "field": "- número total de ganglios aislados:",
            "index": 0,
            "replaces": ["- número total de ganglios aislados:"],
        },
        "isolated_metastasis": {
            "field": "- número de ganglios con metástasis:",
            "index": 0,
            "replaces": ["- número de ganglios con metástasis:"],
        },
        "size_biggest_metastasis": {
            "field": "- tamaño de la  metástasis mayor (tejido tumoral):",
            "index": 0,
            "replaces": [r"- tamaño de la  metástasis mayor \(tejido tumoral\):"],
        },
        "size_smallest_metastasis": {
            "field": "metástasis  menor",
            "index": 0,
            "replaces": ["metástasis  menor", r"\(\s*\"\s*\"\s*\)", ":"],
        },
        "metastasis_type": {
            "field": "- tipo de metástasis:",
            "index": 0,
            "replaces": ["- tipo de metástasis:"],
        },
    }

    for term, term_dict in search_dict.items():
        result_dict[term] = get_line(lines, **term_dict)

    # SEARCH n_ganglions_lvl
    found = False
    level = 1
    for line in lines:
        # Se busca en el formulario hasta encontrar el título deseado
        if (
            not found
            and line.find("- número de ganglios con metástasis por niveles") != -1
        ):
            # Al encontrarse se crea la sección que va a ser rellenada
            found = True
            result_dict["n_ganglions_lvl"] = {}

        # Cada uno de los valores siempre empieza por '*nivel'
        # Por cada nivel encontrado se crea un nuevo campo y almacenan los datos
        elif found and "*nivel" in line:
            result_dict["n_ganglions_lvl"][f"level_{level}"] = line.replace(
                f"*nivel {'i'*level}", ""
            ).strip()

            # En el formulario de ejemplo se especifica que como máximo solo puede haber 3 niveles
            if level > 3:
                logging.warning(f"Reached level {level}, which is bigger than 3.")

            level += 1

    return result_dict


def parse_hormone_receptors_features(lines: Iterable[str]) -> dict[str, str]:
    """Parse the section called "ESTUDIO DE RECEPTORES HORMONALES (ESTRÓGENOS Y PROGESTERONA , MÉTODO INMUNOHISTOQUÍMICO)".

    Parameters
    ----------
    lines: Iterable[str]
        Text of the section splited by line.

    Returns
    -------
    dict[str, str]
        Each field of the section parsed in JSON format.
    """

    result_dict = {}

    for line in lines:
        # Check if appear result and re,
        # to remove not desired entries like: "PENDIENTE"
        if "resultado:" in line and "re:" in line:
            line_split = line.split("re:")[1]

            # If rp appears store both, rp and re.
            if "rp:" in line:
                line_split = line_split.split("rp:")
                result_dict["re"] = line_split[0].strip()
                result_dict["rp"] = line_split[1].strip()
            else:
                result_dict["re"] = line_split.strip()

    return result_dict


def parse_inmunohistochemical_study_features(lines: Iterable[str]) -> dict[str, str]:
    """Parse the section called "ESTUDIO INMUNOHISTOQUIMICO DE HER2/neu:".

    Parameters
    ----------
    lines: Iterable[str]
        Text of the section splited by line.

    Returns
    -------
    dict[str, str]
        Each field of the section parsed in JSON format.
    """

    result_dict = {}

    for line in lines:
        if "resultado:" in line:
            line_split = line.split("resultado:")[1]

            if "(puntuación:" in line:
                line_split = line_split.split("(puntuación:")
                result_dict["result"] = line_split[0].strip()
                result_dict["score"] = line_split[1].replace(")", "").strip()
            else:
                result_dict["result"] = line_split.strip()

    return result_dict


def remove_dict_none_recursive(dic: dict) -> dict:
    """Remove None and empty fields from dict recursivelly.
    Parameters
    ----------
    dic : dict
        Dict to remove None and empty fields from.

    Returns
    -------
    dict
        Dict without Nones and empty fields.
    """
    # Cant iterate over dic, because it throws an error when try to execute del.
    dic = copy.deepcopy(dic)
    keys = list(dic.keys())
    for key in keys:
        # Get the value of the key
        value = dic[key]

        # If it is another dict inside the first one call it recursively,
        # until it reaches the last level.
        if isinstance(value, dict):
            dic[key] = remove_dict_none_recursive(value)

        # Get the value of the key again to prevent modifications
        value = dic[key]

        # If it is the last level and None appear delete this field.
        # If all of the fields inside value where deleted remove this key too.
        if value is None or len(value) == 0:
            del dic[key]
    return dic


def get_intervention_text(
    form_text: str,
    manual_annotated_labels: Optional[pd.Series] = None,
    init_text: str = "tipo de intervención:",
) -> dict[str, str]:
    """Extract the intervention raw text and the intervention annotated automatically and manually.
    We assume that the form always starts with `init_text`

    Parameters
    ----------
    form_text : str
        Form in string format.
    manual_annotated_labels: pd.Series
        Dataframe containing manual annotated interventions texts, by default None.
        The `index` correspond with the intervention text `values` are the intervetions labels.
        If it is None, the manual annotated labels are extracted inside the function originally from:
        https://docs.google.com/spreadsheets/d/1mAoGChwEOAvush7vz5pxhvBn3oRJ4QFO/edit#gid=662190549
        This functionality is added to allow multiprocessing in `xls_raw_preprocessing.py`.
    init_text : str
        Text that identifies the begining of the form, by default "tipo de intervención:".

    Returns
    -------
    dict[str,str]
        Intervention raw text and the intervention annotated automatically and manually.
    """

    if manual_annotated_labels is None:
        manual_annotated_labels = load_manual_labeled_interventions()[["x", "y"]]
        manual_annotated_labels = manual_annotated_labels.set_index(
            "x", verify_integrity=True
        )["y"]

    # select only text including intervention and following it
    form_start_idx = form_text.find(init_text)

    # intervention not found
    if form_start_idx == -1:
        return dict()
    else:
        logging.debug(
            f"`parse_form` found text before {init_text.__repr__()}:\n{form_text[:form_start_idx].__repr__()}"
        )

    form_text_lstrip = form_text[form_start_idx:]
    all_form_lines = form_text_lstrip.split("\n")

    form_lines = [line.strip() for line in all_form_lines if len(line.strip()) > 0]
    if len(form_lines) == 0:
        return dict()

    # initialize output dictionary with fields parsed
    result = dict()
    intervention_text = form_lines[0].replace("tipo de intervención:", "").strip()
    # A veces la intervención en vez de estar en la primera está en la segunda linea
    # Cuando ocurre esto intervention_text queda vacío y se toma la segunda linea
    # Esta se supone que será el valor deseado
    if len(intervention_text) == 0:
        intervention_text = form_lines[1].strip()

        # Si aparece "biopsias /" o "nº biopsias" en la intervención significa que se ha pasado al siguiente campo y no había "tipo de intervención".
        # En este momento se tiene en cuenta "º", pero cuando se pasa por clean_name_hard este desaparece
        if "biopsias /" in intervention_text or "nº biopsias" in intervention_text:
            intervention_text = None

    if intervention_text is not None:
        result["intervention_raw"] = clean_name_hard(intervention_text)
        if result["intervention_raw"]:
            with warnings.catch_warnings():
                # This warning is ignored because it is raise when it isn't found a fuzzy match
                # and this happen many times during the processing of the forms.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Element\s.*\shas\sno\smatch\swith\sthreshold\s\d{0,2}",
                    category=UserWarning,
                )
                result["intervention_automatic"] = get_labels(
                    result["intervention_raw"]
                )

            if result["intervention_raw"] in manual_annotated_labels.index:
                result["intervention_manual"] = manual_annotated_labels[
                    result["intervention_raw"]
                ]

    return result


def parse_form(
    form_text: str,
    manual_annotated_labels: Optional[pd.Series] = None,
    init_text: str = "tipo de intervención:",
) -> dict[str, dict[str, Union[str, dict[str, str]]]]:
    """Parse the first part and three sections of the form.
    We assume that the form always starts with `init_text`

    Parameters
    ----------
    form_text : str
        Form in string format.
    manual_annotated_labels: pd.Series
        Dataframe containing manual annotated interventions texts, by default None.
        The `index` correspond with the intervention text `values` are the intervetions labels.
        If it is None, the manual annotated labels are extracted inside the function originally from:
        https://docs.google.com/spreadsheets/d/1mAoGChwEOAvush7vz5pxhvBn3oRJ4QFO/edit#gid=662190549
        This functionality is added to allow multiprocessing in `xls_raw_preprocessing.py`.
    init_text : str
        Text that identifies the begining of the form, by default "tipo de intervención:".

    Returns
    -------
    dict[str, dict[str, Union[str, dict[str, str]]]]
        Form parsed in JSON format

    Example
    -------
    ```
    TIPO DE INTERVENCIÓN:
    TUMORECTOMÍA Y BSGC

    NÚMERO DE BIOPSIAS / CITOLOGÍAS PREVIAS: H. DE CEUTA

    1- CARACTERÍSTICAS DEL TUMOR:

    - TAMAÑO:  1,7   cm     INDETERMINABLE (fragmentado, artefactado)        LOCALIZACIÓN: UC INF

    - CARACTER:  INFILTRANTE
    - MÚLTIPLE:    NO

    -
    -INFILTRACIÓN DEL MARGEN QUIRÚRGICO / MÁRGENES DE LA BIOPSIA:

    LIBRE (a más de 1cm):  INCLUYENDO LAS AMPLIACIONES DE MUSCULO PECTORAL.

    - TIPO HISTOLÓGICO:
    CARCINOMA DUCTAL INFILTRANTE

    ...
    ```
    """

    form_text = form_text.lower().strip()

    result = get_intervention_text(form_text, manual_annotated_labels, init_text)

    simple_line_features = {
        "n_biopsias": {
            "field": "número de biopsias / citologías previas:",
            "index": 0,
            "replaces": ["número de biopsias / citologías previas:"],
        },
        "proliferative_index": {
            "field": "6- indice proliferativo ( ki67 )",
            "index": 0,
            "replaces": [r"6- indice proliferativo \( ki67 \):"],
        },
    }

    all_form_lines = form_text.split("\n")
    form_lines = [line.strip() for line in all_form_lines if len(line.strip()) > 0]

    for term, term_dict in simple_line_features.items():
        result[term] = get_line(form_lines, **term_dict)

    # SEARCH tumor_features
    TITLES = [
        "1- características del tumor",
        "2- características del parénquima mamario no tumoral",
        "3-  datos de los ganglios axilares",
        "4- estudio de receptores hormonales (estrógenos y progesterona , método inmunohistoquímico)",
        "5-  estudio inmunohistoquimico de her2/neu",
        "5- estudio inmunohistoquimico de her2/neu",
        "6- indice proliferativo ( ki67 )",
        "6- p",
        "7- p",
        "7-   comentarios  /  notas adicionales",
        "7- comentarios",
        "8-   comentarios  /  notas adicionales",
    ]

    # "titles" es una lista porque hay secciones representadas con diferentes títulos
    sections_dict = {
        "tumor_features": {
            "titles": ["1- características del tumor"],
            "function": parse_tumor_features,
        },
        "parenquima_features": {
            "titles": ["2- características del parénquima mamario no tumoral"],
            "function": parse_parenquima_features,
        },
        "ganglion_features": {
            "titles": ["3-  datos de los ganglios axilares"],
            "function": parse_ganglion_features,
        },
        "hormone_receptors": {
            "titles": [
                "4- estudio de receptores hormonales (estrógenos y progesterona , método inmunohistoquímico)"
            ],
            "function": parse_hormone_receptors_features,
        },
        "inmunohistochemical_study": {
            "titles": [
                "5-  estudio inmunohistoquimico de her2/neu",
                "5- estudio inmunohistoquimico de her2/neu",
            ],
            "function": parse_inmunohistochemical_study_features,
        },
    }

    for key, feature_dict in sections_dict.items():
        # Se busca el título que tiene la sección en este formulario
        for feature_title in feature_dict["titles"]:
            if form_text.find(feature_title) != -1:

                title_index = TITLES.index(feature_title)

                # Se busca cuál es el siguiente título más cercano y
                # se toma el texto entre ambos

                for title in TITLES[title_index + 1 :]:
                    if form_text.find(title) != -1:
                        feature_text = (
                            form_text.split(feature_title)[1].split(title)[0].strip()
                        )

                    # Si se llega al final sin encontrar ninguno se toma hasta el final del formulario
                    elif title == TITLES[-1]:
                        feature_text = form_text.split(feature_title)[1].strip()

                    else:
                        feature_text = None

                    if feature_text is not None:
                        lines = feature_text.split("\n")

                        lines = [line.strip() for line in lines if line]

                        result[key] = feature_dict["function"](lines)
                        break

            break

    # Search for pT and pN section
    for line in form_lines:
        if "- pt" in line:
            start = "- pt"

            if "pn" in line:
                result["pt"] = re.search(rf"{start}(.*)pn", line).group(1).strip()
                result["pn"] = re.search(r"pn(.*)", line).group(1).strip()
            else:
                result["pt"] = re.search(r"pt(.*)", line).group(1).strip()

            break

    result = remove_dict_none_recursive(result)

    return result
