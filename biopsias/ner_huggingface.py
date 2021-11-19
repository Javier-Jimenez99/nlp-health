import os
from pathlib import Path
from typing import Iterable, Optional, Union

from pandas.core.frame import DataFrame

import tokenizations
import numpy as np
import pandas as pd
import torch
import transformers
import typer
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from collections import namedtuple

from seqeval import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

from biopsias import config
from biopsias.form.generate_dataset import generate_dataset

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

Prediction = namedtuple("Prediction", "labels logits tokens")


def decode_predictions(
    encoded_tokens: Iterable[str],
    encoded_predictions: Iterable[str],
    decoded_tokens: Iterable[str],
) -> list[str]:
    """Using the tokens from two different tokenizers and the prediction of the first one,
    generate the predictions of the second one.

    Parameters
    ----------
    encoded_tokens : list
        Tokens generated using the first tokenizer.
    encoded_predictions : list
        Predictions generated using the first tokenizer.
    decoded_tokens : list
        Tokens generated using the second tokenizer.

    Returns
    -------
    list
        Predictions generated using the second tokenizer.
    """
    s2h, h2s = tokenizations.get_alignments(decoded_tokens, encoded_tokens)
    predictions_spacy = [list() for t in decoded_tokens]
    for index, pred in zip(h2s, encoded_predictions):
        # assert len(index) <= 1
        if len(index) != 0:
            predictions_spacy[index[0]].append(pred)

    return [aggregate_bio_labels(pred) for pred in predictions_spacy]


def aggregate_bio_labels(labels: Iterable[str]) -> str:
    """Select a valid option over posible labels.
    If it can't be possible a bad token "E" is returned.

    Parameters
    ----------
    labels : Iterable[str]
        Options to select the label

    Returns
    -------
    str
        Selected label.
    """
    # assert len(labels) > 0
    bad_token = "E"
    if len(labels) == 0:
        return "O"

    # Si solo hay una label
    if len(labels) == 1:
        label = labels[0]
    # Si todas son iguales pero no son Bs
    elif len(set(labels)) == 1 and not labels[0].startswith("B"):
        label = labels[0]
    # Caso B, I, I...
    elif (
        labels[0].startswith("B")
        and (len(set(labels[1:])) == 1 and labels[1].startswith("I"))
        and labels[0][1:] == labels[1][1:]
    ):
        label = labels[0]
    else:
        label = bad_token

    return label


class NerDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: Iterable[str],
        tokenizer: AutoTokenizer,
        x_column_name: str = "x_diagnostico",
        y_column_ner_annotation: str = "y_size_representation",
        y_column_tokenization: str = "y_size_text_tokenized",
    ):
        """Initialize the dataset used to train a Huggingface model.
        Parameters
        ----------
        data : pd.DataFrame
            Data used to compound the dataset.
        labels : Iterable[str]
            Labels to which each element belongs.
        tokenizer : AutoTokenizer
            Tokenizer object used to preprocess the text.
        x_column_name : str, optional
            Name of the column use as input of the model, by default "x_diagnostico"
        y_column_ner_annotation : str, optional
            Name of the column that contains the ner labels, by default "y_column_ner_annotation"
        y_column_tokenization: str, optional
            Name of the column that corresponds with the text tokenized, by default
        """
        # Process X values
        self.tokenizer = tokenizer

        self.labels = labels

        self.x_column_name = x_column_name
        self.y_column_ner_annotation = y_column_ner_annotation
        self.y_column_tokenization = y_column_tokenization

        self.batchencodings_df = data.apply(
            self._process_text,
            axis=1,
        )

    def _process_text(
        self,
        row: pd.DataFrame,
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        """Process the text and generate representations to train transformers models.
        Parameters
        ----------
        row: pd.DataFrame
            Row of the dataset.
        Returns
        -------
        transformers.tokenization_utils_base.BatchEncoding
            Object that contains the data used to train transformers models.
        """
        text = row[self.x_column_name]

        hugginface_tokenization = self.tokenizer.tokenize(
            text, add_special_tokens=True, truncation=True, padding="max_length"
        )
        original_tokenization = row[self.y_column_tokenization]

        o2h, h2o = tokenizations.get_alignments(
            original_tokenization, hugginface_tokenization
        )

        y_labels = [self.labels.index("O")] * len(hugginface_tokenization)

        for i, tok_list in enumerate(o2h):
            label = row[self.y_column_ner_annotation][i]
            for j, tok in enumerate(tok_list):
                if "B-" in label and j != 0:
                    y_labels[tok] = self.labels.index(label.replace("B-", "I-"))
                else:
                    y_labels[tok] = self.labels.index(label)

        batchencoding_out = self.tokenizer(text, truncation=True, padding="max_length")
        batchencoding_out["labels"] = y_labels

        return batchencoding_out

    def __len__(self) -> int:
        """Calculate the number of elements inside the dataset.
        Returns
        -------
        int
            Number of elements inside the dataset.
        """
        return len(self.batchencodings_df)

    def __getitem__(
        self, index: int
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        """Get one processed element from the dataset, given it index.
        Parameters
        ----------
        index : int
            Index of the desired element.
        Returns
        -------
        transformers.tokenization_utils_base.BatchEncoding
            Object that contains the data used to train transformers models.
        """
        return self.batchencodings_df.iloc[index]


class HuggingfaceNerModel:
    def __init__(
        self,
        labels: Iterable[str],
        model_path: Union[Path, str] = "dccuchile/bert-base-spanish-wwm-uncased",
        max_text_length: int = 128,
        try_gpu: bool = True,
    ):
        """Initialize a Multiclass Classifier that uses models based on transformers.
        Parameters
        ----------
        labels : Iterable[str]
            Labels to which each element belongs.
        model_path : Union[Path, str], optional
            Path of pretrained model, by default "dccuchile/bert-base-spanish-wwm-uncased"
        max_text_length : int, optional
            Maximum number of tokens(words) that can has a model input, by default 128
        try_gpu : bool, optional
            Search a GPU device to execute operations, by default True
        """
        model_path = Path(model_path)

        # Test if a GPU can be used
        self.device = (
            torch.device("cuda")
            if try_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.max_length = max_text_length

        self.labels = labels
        self.n_labels = len(labels)

        self.load_model(model_path)

    def compute_metrics(
        self, eval_prediction: transformers.EvalPrediction
    ) -> dict[str, float]:
        """Metrics calculated during the training.
        Parameters
        ----------
        eval_prediction : transformers.EvalPrediction
            Prediction given gy the model.
        Returns
        -------
        dict[str, float]
            Results of the metrics.
        """
        y_pred = []
        for tokens in eval_prediction.predictions.argmax(2).tolist():
            y_pred.append([self.labels[t] for t in tokens])

        y_true = []
        for tokens in eval_prediction.label_ids.tolist():
            y_true.append([self.labels[t] for t in tokens])

        results = {
            "precision": metrics.precision_score(y_true, y_pred),
            "recall": metrics.recall_score(y_true, y_pred),
            "F1": metrics.f1_score(y_true, y_pred),
            "accuracy": metrics.accuracy_score(y_true, y_pred),
        }

        return results

    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        labels: list,
        checkpoint_dir: Optional[Union[Path, str]],
        x_column_name: str = "x_diagnostico",
        y_column_ner_annotation: str = "y_size_representation",
        y_column_tokenization: str = "y_size_text_tokenized",
        save_dir: Optional[Union[Path, str]] = None,
        learning_rate: float = 1e-5,
        epochs: int = 3,
        batch_size: int = 16,
        weight_decay: float = 0,
        dropout: float = 0.1,
    ):
        """Train the model.
        Parameters
        ----------
        training_data : pd.DataFrame
            Data used to train.
        validation_data : pd.DataFrame
            Data used to validate.
        labels : list
            Labels to which each element belongs.
        checkpoint_dir : Optional[Union[Path,str]]
            Path to folder where training checkpoints are saved.
        x_column_name : str, optional
            Name of the column use as input of the model, by default "x_diagnostico"
        y_column_name : str, optional
            Name of the column use as outputput of the model, by default "y_intervention"
        save_dir : Optional[Union[Path,str]], optional
            Path to folder where the model weights are saved, by default None.
            If it is `None` the model won't be saved.
        learning_rate : float, optional
            Learning rate used by the optimizer (AdamW), by default 1e-5
        epochs : int, optional
            Number of training epochs, by default 3
        batch_size : int, optional
            Size of the batch used as input of the model in each step, by default 16
        weight_decay : float, optional
            Weight decay value, by default 0
        dropout : float, optional
            Dropout probability, by default 0.1
        """
        checkpoint_dir = Path(checkpoint_dir)

        train_dataset = NerDataset(
            training_data,
            labels,
            self._tokenizer,
            x_column_name,
            y_column_ner_annotation,
            y_column_tokenization,
        )
        validation_dataset = NerDataset(
            validation_data,
            labels,
            self._tokenizer,
            x_column_name,
            y_column_ner_annotation,
            y_column_tokenization,
        )

        args = TrainingArguments(
            output_dir=checkpoint_dir,
            do_train=True,
            do_eval=True,
            save_steps=0,
            eval_steps=100,
            evaluation_strategy="steps",
            logging_first_step=True,
            disable_tqdm=False,
            dataloader_num_workers=6,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            weight_decay=weight_decay,
            warmup_steps=0,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            no_cuda=self.device == "cpu",
            seed=0,
        )

        # Dropout is used here because it must be defined when model is created
        self._model = self._model_init(dropout)

        trainer = Trainer(
            self._model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self._tokenizer,
            eval_dataset=validation_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        if save_dir is not None:
            save_dir = Path(save_dir)
            trainer.save_model(save_dir)

    def validate(
        self,
        validation_data: pd.DataFrame,
        x_column_name: str = "x_diagnostico",
        y_column_ner_annotation: str = "y_size_representation",
        y_column_tokenization: str = "y_size_text_tokenized",
    ):
        """Validate the model using new input data.
        Parameters
        ----------
        validation_data : pd.DataFrame
            Data used to validate.
        x_column_name : str, optional
            Name of the column use as input of the model, by default "x_diagnostico"
        y_column_name : str, optional
            Name of the column use as outputput of the model, by default "y_intervention"
        """
        validation_dataset = NerDataset(
            validation_data,
            self.labels,
            self._tokenizer,
            x_column_name,
            y_column_ner_annotation,
            y_column_tokenization,
        )

        self._model.eval()

        y_pred = []
        y_true = []
        for batch_encoding in validation_dataset:
            out = self._model(
                **(batch_encoding.convert_to_tensors("pt", True).to(self.device)),
                return_dict=True,
            )

            pred = out.logits.squeeze().argmax(1).cpu().numpy()
            true = batch_encoding["labels"].squeeze().cpu().numpy()

            pred = pred[
                batch_encoding["attention_mask"].squeeze().cpu().numpy().astype(bool)
            ]
            true = true[
                batch_encoding["attention_mask"].squeeze().cpu().numpy().astype(bool)
            ]

            y_pred.append([self.labels[v] for v in pred])
            y_true.append([self.labels[v] for v in true])

        # y_pred = np.array(y_pred)
        # y_true = np.array(y_true)

        print(metrics.classification_report(y_true, y_pred))

        # cm = confusion_matrix(
        #   np.array(y_true).flatten(), np.array(y_pred).flatten(), labels=self.labels
        # )
        # disp = ConfusionMatrixDisplay(cm, display_labels=self.labels)
        # disp.plot()
        # plt.show()

    def _model_init(self, dropout: float = 0.1) -> AutoModelForTokenClassification:
        """Initialize the model.
        Parameters
        ----------
        dropout : float, optional
            Dropout probability, by default 0.1
        Returns
        -------
        AutoModelForTokenClassification
            Huggingface model used to solve multiclass classification problems.
        """
        config = AutoConfig.from_pretrained(
            self._model_path,
            num_labels=self.n_labels,
            return_dict=True,
            hidden_dropout_prob=dropout,
        )

        return AutoModelForTokenClassification.from_pretrained(
            self._model_path, config=config
        ).to(self.device)

    def load_model(self, model_path: Union[Path, str]):
        """Load pretrained model.
        Parameters
        ----------
        model_path : Union[Path,str]
            Path to pretrained model.
            It can be a path to Huggingface Model Hub (https://huggingface.co/models) or
            a local folder containing wights and tokenizer of the model.
        """
        model_path = Path(model_path)
        self._model_path = model_path
        self._model = self._model_init()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, model_max_length=self.max_length, use_fast=True
        )

    def save_model(self, save_dir: Union[Path, str]):
        """Save model wights and tokenizer.
        Parameters
        ----------
        save_dir : Union[Path,str]
            Path to folder where model will be saved.
        """
        save_dir = Path(save_dir)
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)

    def predict(self, text_to_predict: str) -> namedtuple:
        """predicts the ner labels that correspond
        to each token of the given text.

        Parameters
        ----------
        text_to_predict : str
            Text to obtain predictions from.

        Returns
        -------
        namedtuple
            List containing the predicted ner labels.
        """

        self._model.eval()

        batch_encoding = self._tokenizer(
            text_to_predict,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        out = self._model(
            **batch_encoding,
            return_dict=True,
        )

        mask = batch_encoding["attention_mask"].cpu().squeeze().numpy().astype(bool)

        predictions = out.logits.squeeze().argmax(1).cpu().numpy()[mask]

        ner_labels = [self.labels[v] for v in predictions]

        # Check when there is an 'I-...' label that is not connected with a 'B-...'
        for i in range(len(ner_labels)):
            if ner_labels[i].startswith("I") and (
                i == 0
                or (
                    not ner_labels[i - 1].startswith("B")
                    and not ner_labels[i - 1].startswith("I")
                )
            ):
                ner_labels[i] = "O"

        logits = out.logits.squeeze().cpu().detach().numpy()[mask]
        tokens = self._tokenizer.convert_ids_to_tokens(
            batch_encoding["input_ids"].squeeze().cpu().numpy()[mask].tolist()
        )

        return Prediction(ner_labels, logits, tokens)


def main(
    x_column_name: str = "x_diagnostico",
    y_column_ner_annotation: str = "y_size_representation",
    y_column_tokenization: str = "y_size_text_tokenized",
    val_size: float = 0.15,
):
    biopsias_data_path = config.all_biopsias_path
    parsed_form_json_path = config.parsed_forms_path
    data = generate_dataset(biopsias_data_path, parsed_form_json_path)
    data = data.dropna(subset=[y_column_ner_annotation])

    labels = sorted(
        set(
            [
                label
                for annotation in data[y_column_ner_annotation]
                for label in annotation
            ]
        )
    )

    model = HuggingfaceNerModel(labels)

    train_data, val_data = train_test_split(data, test_size=val_size, random_state=0)

    model.train(
        train_data,
        val_data,
        labels,
        "checkpoints",
        x_column_name=x_column_name,
        y_column_ner_annotation=y_column_ner_annotation,
        y_column_tokenization=y_column_tokenization,
    )

    model.validate(
        val_data, x_column_name, y_column_ner_annotation, y_column_tokenization
    )


if __name__ == "__main__":
    typer.run(main)
