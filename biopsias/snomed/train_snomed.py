import json
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import typer
from biopsias import binary_classification
from biopsias.config import all_snomed_data_path
from biopsias.form.interventions import clean_x_no_deep_learning
from biopsias.snomed.generate_snomed_dataset import generate_snomed_dataset, get_topk_XY
from biopsias.train import split_tokenizer
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Iterable


def prepare_entry(row) -> dict:
    return (
        {
            "id": row.index.tolist()[0],
            "input": {"text": row["x"]},
            "output": {"text": row["y_true"]},
        },
        {
            "id": row.index.tolist()[0],
            "input": {"text": row["x"]},
            "output": {"text": row["y_pred"]},
        },
    )


def prepare_errors_annotations(
    dataset: pd.DataFrame,
    save_true_path: Union[str, Path],
    save_predictions_path: Union[str, Path],
):
    """Prepare errors between annotations and predictions to compare them.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing the data predicted and annotated.
        The columns must be:
        - "x": Input text
        - "y_true": Annotation
        - "y_pred": Prediction

    save_true_path : Union[str, Path]
        Path to Jsonl where the annotations will be ready to compare.
    save_predictions_path : Union[str, Path]
        Path to Jsonl where the predictions will be ready to compare.
    """

    jsonl_list = dataset.apply(prepare_entry, axis=1).tolist()

    true_file = open(save_true_path, "w")
    predictions_file = open(save_predictions_path, "w")

    for line in jsonl_list:
        true_file.write(json.dumps(line[0]) + "\n")
        predictions_file.write(json.dumps(line[1]) + "\n")


def get_metrics(y_true: np.array, y_pred: np.array, labels: Iterable[str]):
    """Calculate and show the metrics obtained after training.

    Parameters
    ----------
    y_true : np.array
        Labels annotated.
    y_pred : np.array
        Labels predicted.
    labels : Iterable[str]
        Available labels in string format.
    """

    # METRICS
    print(f"Total support: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro')}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted')}")

    CM = confusion_matrix(y_true, y_pred, labels=labels)

    classes_n_elements = CM.sum(axis=1)
    classes_accuracy = CM.diagonal() / classes_n_elements

    # TOP ACCURACY LABELS
    print("\nTop accuracy classes:")
    top_classes_index = list(np.argsort(classes_accuracy))
    top_classes_index.reverse()
    for class_index in top_classes_index[:5]:
        print(f"\t- {labels[class_index]}")
        print(
            f"\t\tAccuracy:{classes_accuracy[class_index]}\tSupport:{classes_n_elements[class_index]}\n"
        )

    # CORRELATION BETWEEN LABELS
    sim_CM = (CM + CM.T) / 2
    labels_cor = []
    for i in range(len(labels)):
        for j in range(i):
            labels_cor.append(
                {"label1": labels[i], "label2": labels[j], "corr": sim_CM[i, j]}
            )

    labels_cor_df = pd.DataFrame(labels_cor)
    labels_cor_df_sort = labels_cor_df.sort_values("corr", ascending=False)
    print("\nCorrelation between labels:")
    labels_cor_df_sort.head(5).apply(
        lambda x: print(f"{x['label1']} \tx\t {x['label2']}: {x['corr']}"), axis=1
    )

    # NUMBER OF ELEMENTS VS ACCURACY
    plt.scatter(classes_n_elements, classes_accuracy)
    plt.xlabel("Number of elements")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of each class")
    # plt.xlim([0, 1000])
    plt.savefig("Figures/snomed_elements_vs_accuracy.png")


def train_sklearn(
    data: pd.DataFrame,
    model: any,
    vectorizer: any,
    val_size: float = 0.15,
    save_model_path: Union[str, Path] = None,
    save_true_path: Union[str, Path] = None,
    save_pred_path: Union[str, Path] = None,
):
    """Train a sklearn model to solve snomed problems.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used to solve the problem.
        It must have "x" and "y" columns.
    model : any
        Sklearn model that will be trained.
    vectorizer : any
        Vectorizer that will be used to train.
    val_size : float, optional
        Split proportion into train and validation data, by default 0.15
    save_model_path : Union[str, Path], optional
        Path to save model, by default None
    save_true_path : Union[str, Path], optional
        Path to Jsonl where the annotations will be ready to compare, by default None
    save_pred_path : Union[str, Path], optional
        Path to Jsonl where the predictions will be ready to compare, by default None
    """

    x, y = (
        data["x"].apply(clean_x_no_deep_learning).to_numpy(),
        data["y"].to_numpy(),
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, random_state=0, stratify=y
    )

    model = binary_classification.train_model(
        x_train, y_train, vectorizer, base_model=model
    )

    y_pred = model.predict(x_val)

    get_metrics(y_val, y_pred, model.classes_)

    if save_true_path is not None and save_pred_path is not None:
        compare_dataset = pd.DataFrame(x_val, columns=["x"])
        compare_dataset["y_true"] = y_val
        compare_dataset["y_pred"] = y_pred

        compare_dataset = compare_dataset[
            compare_dataset["y_true"] != compare_dataset["y_pred"]
        ]

        prepare_errors_annotations(compare_dataset, save_true_path, save_pred_path)

    if save_model_path is not None:
        dump(model, save_model_path)


def main(type="top"):
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        tokenizer=split_tokenizer,
        strip_accents="unicode",
    )

    base_model = LogisticRegression(solver="sag", n_jobs=-1)

    snomed_dataset = generate_snomed_dataset(all_snomed_data_path)

    if type == "top":
        dataset = get_topk_XY(
            snomed_dataset,
            "x_Diagnostico",
            "y_CodTopografico",
            translate_labels_path="data/clean/id_desc_top.json",
        )

        predictions_path = Path(
            "data/clean/prodigy/to_annotate/snomed_top_predictions.jsonl"
        )
        true_path = Path("data/clean/prodigy/to_annotate/snomed_top_true.jsonl")

    elif type == "morf":
        dataset = get_topk_XY(
            snomed_dataset,
            "x_Diagnostico",
            "y_ConceptID_Morphologic",
            translate_labels_path="data/clean/id_desc_morf.json",
        )

        predictions_path = Path(
            "data/clean/prodigy/to_annotate/snomed_morf_predictions.jsonl"
        )
        true_path = Path("data/clean/prodigy/to_annotate/snomed_morf_true.jsonl")

    train_sklearn(
        dataset,
        base_model,
        vectorizer,
        0.4,
        save_true_path=true_path,
        save_pred_path=predictions_path,
    )


if __name__ == "__main__":
    typer.run(main)
