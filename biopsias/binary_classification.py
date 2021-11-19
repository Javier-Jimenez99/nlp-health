import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump
from pathlib import Path
from typing import Union


def train_model(
    x_train: np.array,
    y_train: np.array,
    vectorizer: any,
    base_model: sklearn.linear_model = None,
    save_path: Union[str, Path] = None,
) -> sklearn.linear_model:
    """Preprocess, split the data and train the model.

    Parameters
    ----------
    x_train : np.array
        Numpy matrix containing train inputs.
    y_train : np.array
        Numpy matrix containing train outputs.
    vectorizer : sklearn.feature_extraction.text
        Estimator to vectorize input data.
    base_model : sklearn.linear_model, optional
        Base model to perform the train, by default None.
        If it is None a LogisticRegressor model will be used.
    train_size : float, optional
        Size of the validation set with respect to the total data, by default 0.15
    save_path: Union[str, Path], optional
        Path to save the model trained,
        it must contain the extension name.

    Returns
    -------
    sklearn.linear_model
        Model fitted.
    """

    if base_model is None:
        base_model = LogisticRegression(
            penalty="l1", C=1.0, max_iter=500, solver="liblinear"
        )

    model = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("clf", base_model),
        ]
    )

    model.fit(x_train, y_train)

    if save_path is not None:
        dump(model, save_path)

    return model
