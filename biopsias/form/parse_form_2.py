import re
from typing import Optional, Union
from functools import cache
import pandas as pd
from biopsias.form.parse_form import get_intervention_text

pattern_intro = r"(?:\n|^)\s*\d\.?-?\s*"


@cache
def prepare_single_regex():
    # the order is important
    # 4.- GRADO NUCLEAR DE ...", "5.- NECROSIS", "6.- PATRÓN ARQUITECTURAL", "7.- MÁRGENES DE LA BIOPSIA" and "8.- MICROCALCIFICACIONES:
    dic_title_regex = {
        "histological_type": "tipo",
        "location": r"localizaci[oó]n",
        "size": r"tama[nñ]o",
        "histological_degree": "grado\\s*nuclear\\s*de\\s*van\\s*nuys\\s*\\(s[oó]lo\\s*cdis\\)",
        "tumor_necrosis": "necrosis",
        "architectural_patern_insitu": r"patr[oó]n\s*arquitectural\s*\(solo\s*cdis\)",
        "margin": r"m[áa]rgenes\s*de\s*la\s*biopsia\s*\(solo\s*cdis.\s*siempre\s*marcados\s*con\s*tinta\s*china\)",
    }

    # build one regex for each field
    last_regex = "microcalcificaciones|$"

    # combine the whole dictionary in a single regex
    regex_captures = "".join(
        [
            rf"{pattern_intro}{regex}:?\s*(?P<{title}>(.|\n)*)"
            for title, regex in dic_title_regex.items()
        ]
    )
    # add non-capturing regex
    regex_stop = pattern_intro + last_regex
    full_regex = rf"{regex_captures}{regex_stop}"
    pattern = re.compile(
        full_regex,
        flags=re.IGNORECASE,  # | re.DOTALL,
    )

    return pattern


@cache
def prepare_regex():

    # the order is important
    # 4.- GRADO NUCLEAR DE ...", "5.- NECROSIS", "6.- PATRÓN ARQUITECTURAL", "7.- MÁRGENES DE LA BIOPSIA" and "8.- MICROCALCIFICACIONES:
    dic_title_regex = {
        "histological_type": "tipo",
        "location": r"localizaci[oó]n",
        "size": r"tama[nñ]o",
        "van_nuys_degree": "grado\\s*nuclear\\s*de\\s*van\\s*nuys\\s*\\(s[oó]lo\\s*cdis\\)",
        "tumor_necrosis": "necrosis",
        "architectural_patern_insitu": r"patr[oó]n\s*arquitectural\s*\(solo\s*cdis\)",
        "margin": r"m[áa]rgenes\s*de\s*la\s*biopsia\s*\(solo\s*cdis.\s*siempre\s*marcados\s*con\s*tinta\s*china\)",
    }

    # build one regex for each field
    last_regex = "microcalcificaciones|$"
    all_regexes = list(dic_title_regex.values())
    dic_individual_patterns = {}

    for i, (title, regex) in enumerate(dic_title_regex.items()):
        capture_regex = rf"(?P<{title}>(.|\n)*?)"
        init = rf"{pattern_intro}{regex}:?\s*"
        rest_terms = all_regexes.copy() + [last_regex]
        rest_terms.pop(i)
        end_regex = pattern_intro + "(?:" + "|".join(rest_terms) + ")"
        complete_regex = init + capture_regex + end_regex
        dic_individual_patterns[title] = re.compile(
            complete_regex,
            flags=re.IGNORECASE,  # | re.DOTALL,
        )

    return dic_individual_patterns


def parse_form_2(
    form_text: str,
    manual_annotated_labels: Optional[pd.Series] = None,
    init_text: str = "tipo de intervención:",
    single_pattern: bool = False,
) -> dict[str, dict[str, Union[str, dict[str, str]]]]:
    """Parse the first three fields of the form and the intervention type.
    This is the second kind of form.
    We assume that the form always starts with `init_text`

    TODO: parse "4.- GRADO NUCLEAR DE ...", "5.- NECROSIS", "6.- PATRÓN ARQUITECTURAL", "7.- MÁRGENES DE LA BIOPSIA" and "8.- MICROCALCIFICACIONES:"

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
    TIPO DE INTERVENCIÓN:  Exéresis

    LOCALIZACIÓN:  Región inframamaria

    1. TIPO TUMORAL
    Melanoma maligno extensión superficial

    2. NIVEL Y PROFUNDIDAD
    Nivel de Clark  II
    Espesor de Breslow  0,8 mm.

    ...
    ```
    """
    form_text = form_text.lower().strip()
    result = get_intervention_text(form_text, manual_annotated_labels, init_text)

    if not single_pattern:
        # capture each of the fields with a different regex
        dic_individual_patterns = prepare_regex()
        dic_tumor_features = {}
        for title, pattern in dic_individual_patterns.items():
            match = re.search(
                pattern,
                form_text,
            )
            if match is not None:
                dic_tumor_features[title] = match.group(title).strip()

    else:
        # single pattern for the whole form
        full_pattern = prepare_single_regex()
        match = re.search(full_pattern, form_text)
        if match is not None:
            dic_tumor_features = {
                k: v.strip() for k, v in match.groupdict().items() if v is not None
            }

    # only add sub-dict if it is not empty
    if len(dic_tumor_features) > 0:
        result["tumor_features"] = dic_tumor_features

    return result
