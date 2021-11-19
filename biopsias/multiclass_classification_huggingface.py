import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import transformers
import typer
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

from biopsias import config
from biopsias.form.generate_dataset import generate_dataset

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class PytorchDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: list,
        tokenizer: AutoTokenizer,
        x_column_name: str = "x_diagnostico",
        y_column_name: str = "y_intervention",
    ):
        """Initialize the dataset used to train a Huggingface model.

        Parameters
        ----------
        data : pd.DataFrame
            Data used to compound the dataset.
        labels : list
            Labels to which each element belongs.
        tokenizer : AutoTokenizer
            Tokenizer object used to preprocess the text.
        x_column_name : str, optional
            Name of the column use as input of the model, by default "x_diagnostico"
        y_column_name : str, optional
            Name of the column use as outputput of the model, by default "y_intervention"
        """
        # Process X values
        self.tokenizer = tokenizer
        self.X = data[x_column_name].map(self._process_text)

        # Process Y values
        self.Y = data[y_column_name].map(lambda x: labels.index(x))

    def _process_text(
        self, text: str
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        """Process the text and generate representations to train transformers models.

        Parameters
        ----------
        text : str
            Text to be processed.

        Returns
        -------
        transformers.tokenization_utils_base.BatchEncoding
            Object that contains the data used to train transformers models.
        """
        text = text.lower().replace("\n", " ")

        tokenized = self.tokenizer(text, truncation=True)

        return tokenized

    def __len__(self) -> int:
        """Calculate the number of elements inside the dataset.

        Returns
        -------
        int
            Number of elements inside the dataset.
        """
        return len(self.X.index)

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
        item = self.X.iloc[index]
        item["labels"] = self.Y.iloc[index]
        return item


class HuggingfaceMulticlassClassifier:
    def __init__(
        self,
        n_labels: int,
        model_path: Union[Path, str] = "dccuchile/bert-base-spanish-wwm-uncased",
        max_text_length: int = 128,
        try_gpu: bool = True,
    ):
        """Initialize a Multiclass Classifier that uses models based on transformers.

        Parameters
        ----------
        n_labels : int
            Number of labels that output the model.
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

        print(f"Device: {self.device}")

        self.max_length = max_text_length

        self.n_labels = n_labels

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
        y_pred = eval_prediction.predictions.argmax(1)
        y_true = eval_prediction.label_ids

        if self.n_labels > 2:
            precision, recall, f1score, support = precision_recall_fscore_support(
                y_true, y_pred, average="macro"
            )
        else:
            precision, recall, f1score, support = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )

        results = {
            "precision": precision,
            "recall": recall,
            "F1": f1score,
            "accuracy": (y_pred == y_true).mean(),
        }

        return results

    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        labels: list,
        checkpoint_dir: Optional[Union[Path, str]],
        x_column_name: str = "x_diagnostico",
        y_column_name: str = "y_intervention",
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

        train_dataset = PytorchDataset(
            training_data, labels, self._tokenizer, x_column_name, y_column_name
        )
        validation_dataset = PytorchDataset(
            validation_data, labels, self._tokenizer, x_column_name, y_column_name
        )

        args = TrainingArguments(
            output_dir=checkpoint_dir,
            do_train=True,
            do_eval=True,
            save_steps=0,
            eval_steps=200,
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
        self._model = self.model_init(dropout)

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
        labels: list,
        x_column_name: str = "x_diagnostico",
        y_column_name: str = "y_intervention",
    ):
        """Validate the model using new input data.

        Parameters
        ----------
        validation_data : pd.DataFrame
            Data used to validate.
        labels : list
            Labels to which each element belongs.
        x_column_name : str, optional
            Name of the column use as input of the model, by default "x_diagnostico"
        y_column_name : str, optional
            Name of the column use as outputput of the model, by default "y_intervention"
        """
        validation_dataset = PytorchDataset(
            validation_data, labels, self._tokenizer, x_column_name, y_column_name
        )

        self._model.eval()

        y_pred = []
        y_true = []
        for batch_encoding in validation_dataset:
            out = self._model(
                **(batch_encoding.convert_to_tensors("pt", True).to(self.device)),
                return_dict=True,
            )

            pred = out.logits.squeeze().argmax().cpu().numpy()

            y_pred.append(pred)
            y_true.append(batch_encoding["labels"].cpu().numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        print(
            classification_report(y_true, y_pred, target_names=list(map(str, labels)))
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()

    def model_init(self, dropout: float = 0.1) -> AutoModelForSequenceClassification:
        """Initialize the model.

        Parameters
        ----------
        dropout : float, optional
            Dropout probability, by default 0.1

        Returns
        -------
        AutoModelForSequenceClassification
            Huggingface model used to solve multiclass classification problems.
        """
        config = AutoConfig.from_pretrained(
            self._model_path,
            num_labels=self.n_labels,
            return_dict=True,
            hidden_dropout_prob=dropout,
        )

        return AutoModelForSequenceClassification.from_pretrained(
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
        self._model = self.model_init()
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


def main(
    x_column_name: str = "x_diagnostico",
    y_column_name: str = "y_histological_degree",
    val_size: float = 0.15,
):
    biopsias_data_path = config.all_biopsias_path
    parsed_form_json_path = config.parsed_forms_path
    data = generate_dataset(biopsias_data_path, parsed_form_json_path)
    data = data.dropna(subset=[y_column_name])

    labels = sorted(data[y_column_name].unique())

    model = HuggingfaceMulticlassClassifier(
        len(labels),
    )

    train_data, val_data = train_test_split(data, test_size=val_size, random_state=0)

    model.train(
        train_data,
        val_data,
        labels,
        "checkpoints",
        x_column_name=x_column_name,
        y_column_name=y_column_name,
    )

    model.validate(val_data, labels, x_column_name, y_column_name)


if __name__ == "__main__":
    typer.run(main)
