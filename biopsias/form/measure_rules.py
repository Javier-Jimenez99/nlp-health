from pathlib import Path
from typing import List, Optional, Union

import jsonlines
import pandas as pd
import spacy
import typer


def add_entity_measure_ruler(nlp: spacy.Language) -> List[dict[str, str]]:
    """Add rules to `spacy.language` to extract measurements.

    Parameters
    ----------
    nlp : spacy.Language
        `spacy.language`.

    Returns
    -------
    List[dict[str, str]]
        List of patterns used to identifies measurements.
    """
    pattern_distance_unit = r"(?i)[mc]?m"
    pattern_number = r"\d+([\,\.]\d+)?"
    pattern_x = r"[(por)x]"
    # entity_ruler = EntityRuler(nlp, phrase_matcher_attr="LOWER")

    entity_ruler = nlp.add_pipe("entity_ruler", last=True)  # spaCy v3 syntax

    # 4.1 cm x 5.4 cm
    pattern_area = [
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": pattern_distance_unit}},
        {"TEXT": {"REGEX": rf"(?i){pattern_x}"}},
        {"LIKE_NUM": True},
    ]
    # 3.4x1 cm
    pattern_area1 = [
        {"TEXT": {"REGEX": rf"(?i){pattern_number}{pattern_x}{pattern_number}"}},
        {"TEXT": {"REGEX": pattern_distance_unit}},
    ]

    # 3.4 x 1 cm
    pattern_area2 = [
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": rf"(?i){pattern_x}"}},
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": pattern_distance_unit}},
    ]

    # 3.5 cm x 5
    pattern_area3 = [
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": pattern_distance_unit}},
        {"TEXT": {"REGEX": rf"(?i){pattern_x}"}},
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": pattern_distance_unit}},
    ]

    pattern_distance = [{"LIKE_NUM": True}, {"TEXT": {"REGEX": pattern_distance_unit}}]

    all_patterns = [
        {
            "label": "AREA",
            "pattern": pattern_area,
        },
        {
            "label": "AREA",
            "pattern": pattern_area1,
        },
        {
            "label": "AREA",
            "pattern": pattern_area2,
        },
        {
            "label": "AREA",
            "pattern": pattern_area3,
        },
        {
            "label": "DISTANCIA",
            "pattern": pattern_distance,
        },
    ]

    entity_ruler.add_patterns(all_patterns)

    return all_patterns


def define_nlp_deprecated(
    save_patterns: Optional[Union[Path, str]] = None
) -> spacy.Language:
    """Define a `spacy.language` to extract measurements inside a text.

    Parameters
    ----------
    save_patterns : Optional[Union[Path, str]], optional
        Path to save language patterns, by default None.
        It is usefull for prodigy.
        If it is None, the patterns are not saved.

    Returns
    -------
    spacy.Language
        `spacy.language` prepared to identify measurements inside a text.
    """
    nlp = spacy.blank("es")
    all_patterns = add_entity_measure_ruler(nlp)

    if save_patterns is not None:
        with jsonlines.open(Path(save_patterns), mode="w") as writer:
            for p in all_patterns:
                writer.write(p)

    return nlp


def define_nlp() -> spacy.Language:
    """Define a `spacy.language` to extract measurements inside a text.
    The measurements are extracted using a regex at sentence level with SpaczzRuler library.

    Returns
    -------
    spacy.Language
        `spacy.language` prepared to identify measurements inside a text.
    """
    nlp = spacy.blank("es")
    ruler = nlp.add_pipe("spaczz_ruler")

    number_pattern = r"\d+([\.,\']\d+)?\s*"
    x_pattern = r"[(por)xy]\s*"

    patterns = [
        {
            "label": "MEASURE",
            "pattern": rf"(?i)(?P<number1>{number_pattern})({x_pattern}(?P<number2>{number_pattern}))?({x_pattern}(?P<number3>{number_pattern}))?(?P<unit>[mc]?m)",
            "type": "regex",
        },
    ]

    ruler.add_patterns(patterns)

    return nlp


def get_entities_text(
    text: str, nlp: spacy.Language, drop_duplicates=False
) -> tuple[str]:
    """Get the text from the entities extract from text.

    Parameters
    ----------
    text : str
        Text to extract measurements from.
    nlp : spacy.Language
        `spacy.language` used to extract entities from the text.
    drop_duplicates : bool, optional
        Determine whether to remove duplicates measurements or not, by default False

    Returns
    -------
    tuple[str]
        All of the measurements texts found in the text.
    """
    doc = nlp(text)
    entities = (str(ent) for ent in doc.ents)
    if drop_duplicates:
        entities = set(entities)
    return sorted(entities)


def get_series_entities(
    series: pd.Series, return_nlp: bool = False
) -> Union[tuple[pd.DataFrame, spacy.Language], pd.DataFrame]:
    """Extract all the data related with the measurements entities found in a `pd.Series`.

    Parameters
    ----------
    series : pd.Series
        pd.Series to extract information from.
    return_nlp : bool
        Determine whether to return the `spacy.Language` used or not, by default False.

    Returns
    -------
    pd.DataFrame
        `pd.DataFrame` that contains all the data related with the measurements entities found.
        If `return_nlp` is `True`,`spacy.Language` used is also returned.
    """
    nlp = define_nlp()
    entities = series.map(nlp)

    result_df = pd.DataFrame()
    result_df["ent_text"] = entities.map(lambda x: [ent.text for ent in x.ents])
    # `start` and `end` attributes are token level indices, not chat level ones
    result_df["ent_start_token"] = entities.map(lambda x: [ent.start for ent in x.ents])
    result_df["ent_end_token"] = entities.map(lambda x: [ent.end for ent in x.ents])

    if return_nlp:
        return result_df, nlp
    else:
        return result_df


def main(text: str = "Esto mide 3 m y lo otro 4.0x3,7 cm"):
    nlp = define_nlp()
    print(get_entities_text(text, nlp))


if __name__ == "__main__":
    typer.run(main)
