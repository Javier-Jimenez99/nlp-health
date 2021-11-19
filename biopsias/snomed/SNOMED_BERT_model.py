from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset, DataLoader
import torch
import random


class SNOMED_dataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def processText(self, text):
        text = text.lower().replace("\n", " ")

        tokenized = self.tokenizer(text, truncation=True)

        return tokenized

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        x = self.processText(row["x"]).data

        try:
            y = self.labels.index(row["y"])
        except:
            y = len(self.labels) - 1

        x["labels"] = y
        return x

    def randomItem(self):
        return self.__getitem__(random.randint(0, self.__len__()))


class Model:
    def __init__(
        self,
        modelPath="dccuchile/bert-base-spanish-wwm-uncased",
        nLabels=401,
        try_gpu=True,
    ):
        self.device = (
            torch.device("cuda")
            if try_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.maxLength = 128

        self.nLabels = nLabels

        self.load_model(modelPath)

    def computeMetrics(self, evalPrediction):
        yPred = evalPrediction.predictions.argmax(1)
        yTrue = evalPrediction.label_ids

        metrics = {}

        metrics["accuracy"] = (yPred == yTrue).mean()

        return metrics

    def train(
        self, saveDir, checkpointDir, trainingData, validationData, labels, config
    ):
        trainDataset = SNOMED_dataset(trainingData, labels, self.TOKENIZER)
        validationDataset = SNOMED_dataset(validationData, labels, self.TOKENIZER)

        args = TrainingArguments(
            output_dir=checkpointDir,
            do_train=True,
            do_eval=True,
            save_steps=0,
            eval_steps=200,
            evaluation_strategy="steps",
            logging_first_step=True,
            disable_tqdm=False,
            dataloader_num_workers=6,
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=16,
            weight_decay=config["weight_decay"],
            warmup_steps=0,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
        )

        model = self.model_init(config["dropout"])

        trainer = Trainer(
            model,
            args=args,
            train_dataset=trainDataset,
            tokenizer=self.TOKENIZER,
            eval_dataset=validationDataset,
            compute_metrics=self.computeMetrics,
        )

        trainer.train()

        trainer.save_model(saveDir)

    def predict(self, text, topNumber):
        dataset = SNOMED_dataset(None, None, self.TOKENIZER)
        batchEncoding = (
            dataset.processText(text).convert_to_tensors("pt", True).to(self.device)
        )

        self.MODEL.eval()
        out = self.MODEL(**batchEncoding, return_dict=True)

        values, indices = torch.topk(out.logits, topNumber)

        return indices.tolist()[0], values.tolist()[0]

    def model_init(self, dropout=0.1):
        config = AutoConfig.from_pretrained(
            self.MODEL_PATH,
            num_labels=self.nLabels,
            return_dict=True,
            hidden_dropout_prob=dropout,
        )

        return AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_PATH, config=config
        ).to(self.device)

    def load_model(self, modelPath):
        self.MODEL_PATH = modelPath
        self.MODEL = self.model_init()
        self.TOKENIZER = AutoTokenizer.from_pretrained(
            self.MODEL_PATH, model_max_length=self.maxLength, use_fast=True
        )

    def save_model(self, saveDir):
        self.MODEL.save_pretrained(saveDir)
        self.TOKENIZER.save_pretrained(saveDir)
