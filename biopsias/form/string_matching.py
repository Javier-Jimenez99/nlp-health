import unicodedata
import warnings
from typing import Callable, Iterable
from typing import Union
import numpy as np
from fuzzywuzzy import process
from Levenshtein import distance
from collections import namedtuple

from nltk.corpus import stopwords
import nltk

try:
    STOPWORDS = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("spanish"))

REPLACE_BY_SPACE = "[/(){}\[\]\|@,;]:+"


def strip_accents(text: str) -> str:
    """Strip accents from input string

    Parameters
    ----------
    text : str
        input string

    Returns
    -------
    str
        string without accents
    """
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore")
    text = text.decode("utf-8")
    return str(text)


def clean_name_hard(text: str) -> str:
    """Clean a text deleting ".-", replacing by space ",", replacing by " + ", " mas " and " y ",
    and add spaces to  "/()+"

    Parameters
    ----------
    text : str
        String that will be cleaned.

    Returns
    -------
    str
        String cleaned.
    """
    text = text.lower()

    text = strip_accents(text)

    CHARS_DELETE = ".-"
    for c in CHARS_DELETE:
        text = text.replace(c, "")

    CHARS_SPACE = ","
    for c in CHARS_SPACE:
        text = text.replace(c, " ")

    REPLACE_PLUS = [" mas ", " y "]
    for c in REPLACE_PLUS:
        text = text.replace(c, " + ")

    ADD_SPACES = "/()+"
    for c in ADD_SPACES:
        text = text.replace(c, " " + c + " ")

    # DELETE SPACES
    text = " ".join(text.split())

    return text


def clean_x_no_deep_learning(text: str) -> str:
    """Clean text removing not desired characters and stopwords,
    to work with non deep learnings algorithms.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned text.
    """

    # 1. Apply clean_name_hard (lower,remove,replace)
    # 2. Remove not desired characters, that don't manage clean_name_hard.
    # 3. Delete spanish stopwords

    text = clean_name_hard(text)

    for c in REPLACE_BY_SPACE:
        text = text.replace(c, " ")

    text = " ".join([word for word in text.split() if word not in STOPWORDS])

    return text


def distance_names(s1: str, s2: str) -> float:
    """Claculate a variation of Levenstein distance between two strings.
    Result is 0 if they are very similar.

    Parameters
    ----------
    s1 : str
    s2 : str

    Returns
    -------
    float
        Distance between strings in [0,1] interval. 0 if they are very similar.
    """

    n1, n2 = map(clean_name_hard, (s1, s2))

    distances = np.array(
        [
            [distance(w1, w2) / max(len(w1), len(w2)) for w1 in n1.split()]
            for w2 in n2.split()
        ]
    )

    final_distance = min(distances.min(axis=0).mean(), distances.T.min(axis=0).mean())

    return final_distance


def scorer_token_set(s1: str, s2: str) -> int:
    """Wrap around `distance_names`. Translate distance into score.
    Result in [0,100] interval. 100 if they are very similar.

    Note: It assumes that the original distance is in the [0,1] range.

    Parameters
    ----------
    s1 : str
    s2 : str

    Returns
    -------
    int
        Score similarity in [0,100] interval. 100 if they are very similar.
    """
    distance = distance_names(s1, s2)
    assert (
        0 <= distance <= 1
    ), f"Distance between {s1} and {s2} is not in the [0,1] range: {distance:.3f}"

    return int(100 * (1 - distance))


def word_n_grams(string: str, n: Union[int, Iterable[int]]) -> list[str]:
    """Generetas n-grams for a given string.

    Parameters
    ----------
    string : str
        String to generate n-grams from.
    n : Union[int, Iterable[int]]
        Posible sizes of n-grams generated.

    Returns
    -------
    list[str]
        List that includes n-grams
    """
    words = string.split()
    ngrams = []

    if isinstance(n, int):
        n = (n,)

    for m in n:
        assert m > 0, f"The number of n-grams must be positive. {n} is not valid."
        ngrams.extend(" ".join(words[i : i + m]) for i in range(len(words) - m + 1))

    return ngrams


def create_dict_fuzzymatch(
    keys: Iterable[str],
    choices: Iterable[str],
    scorer: Callable = scorer_token_set,
    score_cutoff: int = 90,
) -> dict[str, tuple[Union[str, int]]]:
    """Create a dictionary relating each of the names in `names_values` with a name in `names_keys`.
    It drops entries with distance greater than `threshold`.

    Parameters
    ----------
    keys : Iterable[str]
        names comming from the assistance emails
    choices : Iterable[str]
        Names comming from the assistance spreadsheets
    scorer : Callable, optional
        Similarity score function, it should return an integer [0,100]. 100 being an exact match.
        By default, `scorer_token_set`
    score_cutoff : int, optional
        Distance threshold to consider a case as a false positive, by default 90

    Returns
    -------
    dict[str, tuple[Union[str,int]]]
        Dictionary relating each name from `keys` with one element from `choices` called `match` and the `score`.
        It is sorted by `score`.
    """
    keys = set(keys)
    choices = set(choices)

    result_tuple = namedtuple("result_tuple", "match score")

    dic = dict()
    for key in keys:
        result = process.extractOne(
            key, choices, scorer=scorer, score_cutoff=score_cutoff
        )
        if result is None:
            match, score = max(
                [(choice, scorer(key, choice)) for choice in choices],
                key=lambda x: x[1],
            )
            warnings.warn(
                f"Element '{key}' has no match with threshold {score_cutoff}. "
                f"You should lower `score_cutoff` to obtain a match. "
                f"Best match of '{key}' is '{match}' with score {score}."
            )
        else:
            match, score = result
            dic[key] = result_tuple(match=match, score=score)

    dic = dict(sorted(dic.items(), key=lambda x: x[1].score, reverse=True))
    return dic
