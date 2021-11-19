import logging
import os
from collections import namedtuple
from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.multiclass import unique_labels

from joblib import dump


def plot_roc_multilabel(
    y_true: np.array,
    y_score: np.array,
    labels: Iterable[str],
    save_path: Optional[Path] = None,
):
    """Plots ROC curve for micro and macro averaging for each of the labels in a multilabel classification problem.

    Parameters
    ----------
    y_true : np.array
        Array of shape (n_samples,n_labels) that represents the classification made on each of the samples.
    y_score : np.array
        Array of shape (n_samples,n_labels) that represents the probability estimates.
    labels : Iterable[str]
        List of available labels to classify each sample
    save_path : OptionalUnion[[str,Path]]
        Path to save figures, by default None.
        If `None` figures are not saved.
    """

    # Can get the number of labels from y_true shape
    n_labels = len(labels)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_labels)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_labels):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_labels
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "salmon",
            "mediumseagreen",
            "gold",
            "maroon",
            "darkslateblue",
            "orchid",
        ]
    )
    for i, color in zip(range(0, n_labels), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of ROC to multi-class")
    plt.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(Path(save_path) / "roc_curve.png")

    plt.show()


def plot_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    *,
    labels: Iterable[str] = None,
    sample_weight: np.array = None,
    normalize: str = None,
    display_labels: np.array = None,
    include_values: bool = True,
    xticks_rotation: Union[str, float] = "horizontal",
    values_format: str = None,
    cmap: Union[
        str, tuple[str, Iterable[str]]
    ] = "viridis",  # Creo que tuple[str,Iterable[str]] es un colormap,
    # porque no se si existe una clases específica para eso en matplotlib
    ax: plt.axes = None,
    colorbar: bool = True,
):
    """Plot Confusion Matrix of a single label.

    Parameters
    ----------
    y_true : np.array
        {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y_pred : np.array
        array-like of shape (n_samples,)
        Target values.
    labels : Iterable[str], optional
        array-like of shape (n_classes,), by default None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    sample_weight : np.array, optional
        array-like of shape (n_samples,), by default None
        Sample weights.
    normalize : str, optional
        {'true', 'pred', 'all'}, by default None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : np.array, optional
        array-like of shape (n_classes,), by default None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    include_values : bool, optional
        Includes values in confusion matrix., by default True
    xticks_rotation : Union[str, float], optional
        Rotation of xtick labels., by default "horizontal"
    values_format : str, optional
        Format specification for values in confusion matrix., by default None.
        If `None`, the format specification is 'd' or '.2g' whichever is shorter.
    cmap : Union[ str, tuple[str, Iterable[str]] ], optional
        Colormap recognized by matplotlib., by default "viridis"
    ax : plt.axes
        Axes object to plot on, by default None.
        If `None`, a new figure and axes is created.
    colorbar : bool, optional
        Whether or not to add a colorbar to the plot, by default True
    """

    cm = confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=labels, normalize=normalize
    )

    if display_labels is None:
        if labels is None:
            display_labels = unique_labels(y_true, y_pred)
        else:
            display_labels = labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        values_format=values_format,
        colorbar=colorbar,
    )


def plot_confusion_matrix_per_label(
    y_true: np.array,
    y_pred: np.array,
    *,
    labels: Iterable[str] = None,
    sample_weight: np.array = None,
    normalize: str = None,
    display_labels: np.array = None,
    include_values: bool = True,
    xticks_rotation: Union[str, float] = "horizontal",
    values_format: str = None,
    cmap: Union[
        str, tuple[str, Iterable[str]]
    ] = "viridis",  # Creo que tuple[str,Iterable[str]] es un colormap,
    # porque no se si existe una clases específica para eso en matplotlib
    ax: plt.axes = None,
    colorbar: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot Confusion Matrix of each label, in a multilabel classification problem.
    The number of plots is equal to the number of labels.

    Parameters
    ----------
    y_true : np.array
        {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y_pred : np.array
        array-like of shape (n_samples,)
        Target values.
    labels : Iterable[str], optional
        array-like of shape (n_classes,), by default None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    sample_weight : np.array, optional
        array-like of shape (n_samples,), by default None
        Sample weights.
    normalize : str, optional
        {'true', 'pred', 'all'}, by default None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : np.array, optional
        array-like of shape (n_classes,), by default None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    include_values : bool, optional
        Includes values in confusion matrix., by default True
    xticks_rotation : Union[str, float], optional
        Rotation of xtick labels., by default "horizontal"
    values_format : str, optional
        Format specification for values in confusion matrix., by default None.
        If `None`, the format specification is 'd' or '.2g' whichever is shorter.
    cmap : Union[ str, tuple[str, Iterable[str]] ], optional
        Colormap recognized by matplotlib., by default "viridis"
    ax : plt.axes
        Axes object to plot on, by default None.
        If `None`, a new figure and axes is created.
    colorbar : bool, optional
        Whether or not to add a colorbar to the plot, by default True
    save_path : OptionalUnion[[str,Path]]
        Path to save figures, by default None.
        If `None` figures are not saved.
    """

    if labels is None:
        labels = map(lambda x: f"Label {x}", range(y_true.shape[1]))

    for idx, label in enumerate(labels):
        plot_confusion_matrix(
            y_true[:, idx],
            y_pred[:, idx],
            labels=None,
            sample_weight=sample_weight,
            normalize=normalize,
            display_labels=display_labels,
            include_values=include_values,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            cmap=cmap,
            ax=ax,
            colorbar=colorbar,
        )
        title = f"{label}_confusion_matrix"
        plt.title(title)

        if save_path is not None:
            plt.savefig(Path(save_path) / f"{title}.png")

    plt.show()


def plot_multiclass_confusion_matrix_from_multilabel(
    labels: np.array,
    y_true: np.array,
    y_pred: np.array,
    classes: np.array = None,
    **plot_confusion_matrix_kwargs,
):
    """Plot a Confusion Matrix os a multi-class problem with a subgroup of labels.
    These classes must be exclusive, i.e., one example cannot belong to more than one class.

    Parameters
    ----------
    labels : np.array
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    y_true : np.array
        {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y_pred : np.array
        array-like of shape (n_samples,)
        Target values.
    classes : np.array, optional
        Labels selected to plot in the confusion matrix, by default None
    """

    if classes is None:
        classes = labels

    # Find indices of the desired labels
    indices_classes = np.where(np.in1d(labels, classes))[0]

    # Select desired labels
    # TODO: no es esto lo lismo que `classes`?
    m_labels = labels[indices_classes]

    y_true_sub = y_true[:, indices_classes]
    assert np.all(
        y_true_sub.sum(axis=1) == 1
    ), "To be a multiclass problem, there must always appear exactly one class in each example in `y_true`."
    y_pred_sub = y_pred[:, indices_classes]
    logging.warning(
        np.all(y_pred_sub.sum(axis=1) == 1),
        "To be a multiclass problem, there must always appear exactly one class in each example in `y_pred`. If there are several, one will be forced.",
    )
    # TODO: qué pasa si no hay ninguna etiqueta? el argmax aun así devuelve algo. Hay que crear una última opción de no existe entre las classes.
    idx_true = y_true_sub.argmax(axis=1)
    idx_pred = y_pred_sub.argmax(axis=1)

    plot_confusion_matrix(
        m_labels[idx_true],
        m_labels[idx_pred],
        labels=m_labels,
        **plot_confusion_matrix_kwargs,
    )
    plt.show()


def train_model(
    x_train: np.array,
    x_val: np.array,
    y_labels_train: np.array,
    y_labels_val: np.array,
    vectorizer: sklearn.feature_extraction.text,
    base_model: sklearn.linear_model = None,
    save_path: Union[str, Path] = None,
) -> namedtuple:
    """Preprocess, split the data and train the model.

    Parameters
    ----------
    x_train : np.array
        Numpy matrix containing train inputs.
    x_val: np.array
        Numpy matrix containing validation inputs.
    y_labels_train : np.array
        Numpy matrix containing train outputs.
    y_labels_val: np.array
        Numpy matrix containing validation outputs.
    vectorizer : sklearn.feature_extraction.text
        Estimator to vectorize input data.
    base_model : sklearn.linear_model, optional
        Base model to perform the train, by default None.
        If it is None a LogisticRegressor model will be used.
    save_path: Union[str, Path], optional
        Path to save the model trained,
        it must contain the extension name.

    Returns
    -------
    namedtuple
        Named tuple that contains: data used to train (`x_train`, `y_train`), data used to validate (`x_val`, `y_val`),
        model fitted (`model`) and labels generated (`label`).
    """

    mlb = MultiLabelBinarizer()

    y_train = mlb.fit_transform(y_labels_train)
    y_val = mlb.transform(y_labels_val)
    labels = mlb.classes_

    if base_model is None:
        base_model = LogisticRegression(
            penalty="l1", C=1.0, max_iter=500, solver="liblinear"
        )

    model = Pipeline(
        [
            ("vectorizer", vectorizer),
            (
                "clf",
                OneVsRestClassifier(base_model, n_jobs=-1),
            ),
        ]
    )

    model.fit(x_train, y_train)

    if save_path is not None:
        dump((model, labels), save_path)

    result = namedtuple("Result", "x_train y_train x_val y_val model labels")
    return result(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model=model,
        labels=labels,
    )
