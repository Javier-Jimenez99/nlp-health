import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from biopsias import binary_classification, multilabel_classification
from biopsias.config import (
    all_biopsias_path,
    histological_degree_model_path,
    histological_type_model_path,
    intervention_type_model_path,
    is_cdi_model_path,
    parsed_forms_path,
    tumor_size_automatic_model_path,
    tumor_size_manual_model_path,
    all_snomed_data_path,
    morf_save_path,
    top_save_path,
)
from biopsias.form.generate_dataset import (
    generate_dataset,
    generate_histological_type_with_annotations,
    generate_annotations_tumor_size_dataset,
)
from biopsias.form.string_matching import clean_x_no_deep_learning
from biopsias.ner_huggingface import HuggingfaceNerModel
from biopsias.snomed.generate_snomed_dataset import generate_snomed_dataset, get_topk_XY
from biopsias.train import train
from biopsias.snomed.train_snomed import train_sklearn
from biopsias.split_tokenizer import split_tokenizer

warnings.filterwarnings("ignore")


def train_intervention_type(save_path: Union[str, Path]):
    """Train a model to solve the intervention type problem
    and save it.

    Parameters
    ----------
    save_path : Union[str, Path]
        Path to save the model
    """
    dataset = generate_dataset(all_biopsias_path, parsed_forms_path,)[
        ["x_diagnostico", "y_intervention"]
    ].rename(columns={"x_diagnostico": "x", "y_intervention": "y"})

    value_counts = pd.Series(
        [intervention for row in dataset["y"] for intervention in row]
    ).value_counts()

    k = 5
    top_k = list(value_counts.index)[:k]

    dataset["y"] = dataset["y"].apply(
        lambda x: [
            intervention if intervention in top_k else "otro" for intervention in x
        ]
    )

    X = dataset["x"].apply(clean_x_no_deep_learning).to_numpy()
    Y = dataset["y"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.15, random_state=0
    )

    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        tokenizer=split_tokenizer,
        strip_accents="unicode",
    )

    base_model = LogisticRegressionCV(
        penalty="l1",
        max_iter=1000,
        solver="liblinear",
        cv=5,
        scoring="f1",
        class_weight="balanced",
    )

    multilabel_classification.train_model(
        x_train,
        x_test,
        y_train,
        y_test,
        vectorizer=vectorizer,
        base_model=base_model,
        save_path=save_path,
    )


def train_histological_degree(save_path: Union[str, Path]):
    """Train a model to solve the histological degree problem
    and save it.

    Parameters
    ----------
    save_path : Union[str, Path]
        Path to save the model
    """

    train(
        "x_diagnostico",
        "y_histological_degree",
        "multiclass",
        "traditional",
        all_biopsias_path,
        parsed_forms_path,
        val_size=0.15,
        save_path=save_path,
    )


def train_histological_type(save_path: Union[str, Path], top_k: int):
    """Train a model to solve the histological type problem
    and save it. This value should be lower than 8.

    Parameters
    ----------
    save_path : Union[str, Path]
        Path to save the model
    """
    assert top_k <= 7, f"Number of classes must be lower than 7, it is {top_k}."

    dataset = generate_histological_type_with_annotations(top_k)

    X = dataset["x"].apply(clean_x_no_deep_learning).to_numpy()
    Y = dataset["y"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.15, random_state=0, stratify=Y
    )

    # Create vectorizer, base model and the entire model and validate the results
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        tokenizer=split_tokenizer,
        strip_accents="unicode",
    )

    base_model = LogisticRegressionCV(
        penalty="l1",
        max_iter=500,
        solver="liblinear",
        cv=10,
        scoring="f1",
        class_weight="balanced",
    )

    binary_classification.train_model(
        x_train,
        y_train,
        vectorizer,
        base_model=base_model,
        save_path=save_path,
    )


def train_tumor_size_automatic(save_path: Union[str, Path]):
    """Train a model to solve the tumor size problem
    and save it.

    Parameters
    ----------
    save_path : Union[str, Path]
        Path to save the model
    """

    data = generate_dataset(all_biopsias_path, parsed_forms_path)
    data = data.dropna(subset=["y_size_representation"])

    labels = sorted(
        set(
            [
                label
                for annotation in data["y_size_representation"]
                for label in annotation
            ]
        )
    )

    model = HuggingfaceNerModel(labels, max_text_length=128)

    train_data, val_data = train_test_split(data, test_size=0.15, random_state=0)

    model.train(
        train_data,
        val_data,
        labels,
        "checkpoints",
        x_column_name="x_diagnostico",
        y_column_ner_annotation="y_size_representation",
        y_column_tokenization="y_size_text_tokenized",
        save_dir=save_path,
    )

    model.validate(
        val_data, "x_diagnostico", "y_size_representation", "y_size_text_tokenized"
    )


def train_tumor_size_manual(save_path: Union[str, Path]):
    """Train a model to solve the tumor size problem
    and save it.

    Parameters
    ----------
    save_path : Union[str, Path]
        Path to save the model
    """

    data = generate_annotations_tumor_size_dataset(
        "data/clean/prodigy/annotations/tumor_size_annotations.jsonl",
        "Diagnóstico",
    )

    labels = sorted(
        set([label for annotation in data["y_labels"] for label in annotation])
    )

    model = HuggingfaceNerModel(labels, max_text_length=256)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=0)

    model.train(
        train_data,
        val_data,
        labels,
        "checkpoints",
        x_column_name="x_text",
        y_column_ner_annotation="y_labels",
        y_column_tokenization="y_tokens",
        save_dir=save_path,
        epochs=10,
    )

    model.validate(val_data, "x_text", "y_labels", "y_tokens")


def train_snomed_models(
    morf_save_path: Union[str, Path], top_save_path: Union[str, Path]
):
    """Train a model to solve the SNOMED morphology and topology problems
    and save it.

    Parameters
    ----------
    morf_save_path : Union[str, Path]
        Path to save the morphology model.
    top_save_path : Union[str, Path]
        Path to save the topology model.
    """

    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        tokenizer=split_tokenizer,
        strip_accents="unicode",
    )

    base_model = LogisticRegression(solver="sag", n_jobs=-1)

    snomed_dataset = generate_snomed_dataset(all_snomed_data_path)

    # MORFPHOLOGY
    morf_dataset = get_topk_XY(
        snomed_dataset,
        "x_Diagnostico",
        "y_ConceptID_Morphologic",
        translate_labels_path="data/clean/id_desc_morf.json",
    )

    train_sklearn(
        morf_dataset,
        base_model,
        vectorizer,
        val_size=0.15,
        save_model_path=morf_save_path,
    )

    # TOPOLOGY

    topo_dataset = get_topk_XY(
        snomed_dataset,
        "x_Diagnostico",
        "y_CodTopografico",
        translate_labels_path="data/clean/id_desc_top.json",
    )
    train_sklearn(
        topo_dataset,
        base_model,
        vectorizer,
        val_size=0.15,
        save_model_path=top_save_path,
    )


def main():
    print("TRAINING HISTOLOGICAL TYPE ...")
    train_histological_type(histological_type_model_path, 4)

    print("TRAINING IS CDI ...")
    train_histological_type(is_cdi_model_path, 1)

    print("TRAINING INTERVENTION TYPE ...")
    train_intervention_type(intervention_type_model_path)

    print("TRAINING TUMOR SIZE AUTOMATIC ...")
    # train_tumor_size_automatic(tumor_size_automatic_model_path)

    print("TRAINING TUMOR SIZE MANUAL ...")
    # train_tumor_size_manual(tumor_size_manual_model_path)

    print("TRAINING HISTOLOGICAL DEGREE ...")
    train_histological_degree(histological_degree_model_path)

    print("TRAINING SNOMED MODELS ...")
    train_snomed_models(morf_save_path, top_save_path)


if __name__ == "__main__":
    typer.run(main)
