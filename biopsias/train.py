import logging
import os
import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from biopsias import (
    binary_classification,
    multiclass_classification_huggingface,
    multilabel_classification,
)
from biopsias import config
from biopsias.form.generate_dataset import generate_dataset
from biopsias.form.string_matching import clean_x_no_deep_learning


def split_tokenizer(text: str) -> str:
    """Text tokenizer.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text tokenized.
    """
    # Habrá que sustituirlo en un futuro
    return text.split()


def train(
    x_column_name: str = "x_diagnostico",
    y_column_name: str = "y_intervention",
    problem_type: str = "multilabel",
    model_type: str = "traditional",
    biopsias_data_path: Union[str, Path] = "./data/clean/Biopsias_HUPM_all.parquet",
    parsed_form_json_path: Union[str, Path] = "./data/clean/parsed_forms_raw.json",
    vectorizer_str: str = "tf_idf",
    val_size: float = 0.15,
    report_path: Union[str, Path] = None,
    save_path: Union[str, Path] = None,
):
    """Train a model for the desired dataset field.

    Parameters
    ----------
    x_column_name : str, optional
        Name of the column use as input of the model, by default "x_diagnostico"
    y_column_name : str, optional
        Name of the column use as outputput of the model, by default "y_intervention"
    problem_type : str, optional
        Type of the problem that will be solve, by default "multilabel".
        It must be "binary", "multiclass" or "multilabel".
    model_type : str, optional
        Type of model that will be used to solve the problem, by default "traditional".
        It must be "traditional" or "transformer".
    biopsias_data_path : str, optional
        Path to parquet file that contains all the information about biopsias diagnosis,
        by default "./data/clean/Biopsias_HUPM_all.parquet"
    parsed_form_json_path : str, optional
        Path to JSON file that contains all the forms from the biopsias diagnosis parsed,
        by default "./data/clean/parsed_forms_raw.json"
    vectorizer_str : str, optional
        Type of the vectorizer used to generate a representation of the text, by default "tf_idf".
        It is only used if `model_type` is "traditional".
    val_size : float, optional
        Size of the validation set, by default 0.15
    report_path: Union[str, Path], optional
        Path to save reports plots.
    save_path: Union[str, Path], optional
        Path to save the model trained,
        it must contain the extension name.
    """
    model = None

    if problem_type not in ["binary", "multiclass", "multilabel"]:
        raise ValueError(
            f"{problem_type} is not a valid problem type value. It must be 'binary', 'multiclass' or 'multilabel'"
        )

    if model_type not in ["traditional", "transformer"]:
        raise ValueError(
            f"{model_type} is not a valid model type value. It must be 'traditional' or 'transformer'"
        )

    biopsias_data_path = Path(biopsias_data_path)
    parsed_form_json_path = Path(parsed_form_json_path)

    data = generate_dataset(biopsias_data_path, parsed_form_json_path)

    data = data.dropna(subset=[y_column_name])

    if report_path is not None:
        save_dir = Path(report_path) / y_column_name

        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = None

    if model_type == "traditional":
        vectorizers_available = {
            "bag_of_words": CountVectorizer(
                max_df=0.9,
                min_df=5,
                tokenizer=split_tokenizer,
                strip_accents="unicode",
            ),
            "tf_idf": TfidfVectorizer(
                max_df=0.9,
                min_df=5,
                tokenizer=split_tokenizer,
                strip_accents="unicode",
            ),
        }

        assert (
            vectorizer_str in vectorizers_available.keys()
        ), f"The representation type must be one of the following:{','.join(vectorizers_available.keys())}"

        x, y = (
            data[x_column_name].apply(clean_x_no_deep_learning).to_numpy(),
            data[y_column_name].to_numpy(),
        )

        # If there is a class with just one element cant be done the stratify,
        # because the data cant be divided.
        unique, counts = np.unique(y, return_counts=True)
        if 1 in counts:
            logging.warn(
                "There is a class with only one element, so it can't be stratified."
            )
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=val_size, random_state=0
            )
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=val_size, random_state=0, stratify=y
            )

        if problem_type == "multilabel":
            training_output = multilabel_classification.train_model(
                x_train,
                x_val,
                y_train,
                y_val,
                vectorizers_available[vectorizer_str],
                save_path=save_path,
            )

            x_val = training_output.x_val
            y_val = training_output.y_val
            model = training_output.model
            labels = training_output.labels

            y_pred = model.predict(x_val)
            y_score = model.predict_proba(x_val)

            print(
                classification_report(
                    y_val, y_pred, target_names=list(map(str, labels))
                )
            )

            multilabel_classification.plot_roc_multilabel(
                y_val, y_score, labels, save_dir
            )

            multilabel_classification.plot_confusion_matrix_per_label(
                y_val,
                y_pred,
                normalize="all",
                labels=labels,
                save_path=save_dir,
            )

        elif problem_type in ["binary", "multiclass"]:
            labels = sorted(data[y_column_name].unique())

            model = binary_classification.train_model(
                x_train,
                y_train,
                vectorizers_available[vectorizer_str],
                save_path=save_path,
            )

            y_pred = model.predict(x_val)
            y_score = model.predict_proba(x_val)

            print(
                classification_report(
                    y_val, y_pred, target_names=list(map(str, labels))
                )
            )

            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.show()

            if save_dir is not None:
                plt.savefig(Path(save_dir) / "confusion_matrix.png")

    elif model_type == "transformer":
        # If there is a class with just one element cant be done the stratify,
        # because the data cant be divided.

        if 1 in data[y_column_name].value_counts():
            logging.warn(
                "There is a class with just one element, so it can't be stratified."
            )
            train_data, val_data = train_test_split(
                data, test_size=val_size, random_state=0
            )
        else:
            train_data, val_data = train_test_split(
                data, test_size=val_size, random_state=0, stratify=data[y_column_name]
            )

        if problem_type == "multilabel":
            # TODO: implement multilabel solver using transformers
            raise ValueError(
                "Transformers to solve multilabel problems are not implemented yet."
            )
        elif problem_type in ["binary", "multiclass"]:
            labels = sorted(data[y_column_name].unique())

            model = (
                multiclass_classification_huggingface.HuggingfaceMulticlassClassifier(
                    len(labels),
                )
            )

            model.train(
                train_data,
                val_data,
                labels,
                "checkpoints",
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                save_dir=save_path,
            )

            model.validate(val_data, labels, x_column_name, y_column_name)
    else:
        raise ValueError(
            f"{model_type} is not a valid model type value. It must be 'traditional' or 'transformer'"
        )


def main(
    x_column_name: str = "x_diagnostico",
    y_column_name: str = "y_histological_degree",
    problem_type: str = "multiclass",
    model_type: str = "transformer",
    vectorizer_str: str = "tf_idf",
    val_size: float = 0.15,
):
    train(
        x_column_name,
        y_column_name,
        problem_type,
        model_type,
        config.all_biopsias_path,
        config.parsed_forms_path,
        vectorizer_str,
        val_size,
        config.figures_path,
    )


if __name__ == "__main__":
    typer.run(main)
