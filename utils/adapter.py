import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import torch
from transformers import TrainingArguments, AdapterTrainer, logging
from transformers.adapters import BertAdapterModel

class BERTAdapter():
  # This class implements our BERT+Adapter predictor
  def __init__(self, dataset, num_epochs, batch_size, learning_rate, category, group, device):
    self.dataset = dataset
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.category = category
    self.group = group
    self.adapter_name = self.category + "_" + self.group
    self.num_labels = 2 if self.category == "C" else 1
    self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()
    self.model = BertAdapterModel.from_pretrained("bert-base-uncased").to(self.device)

  def fit(self, X, y, X_val=None, y_val=None):
    # Obtain train/eval datasets
    subset_train = self.dataset.select(X)
    subset_train = subset_train.rename_column(self.group, "labels")
    subset_train.reset_format()
    subset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    if X_val is not None:
      subset_eval = self.dataset.select(X_val)
      subset_eval = subset_eval.rename_column(self.group, "labels")
      subset_eval.reset_format()
      subset_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Add a classification head and adapter
    self.model.add_classification_head(self.adapter_name, num_labels=self.num_labels)
    self.model.add_adapter(self.adapter_name)

    # Enable adapter and prediction head training
    self.model.train_adapter(self.adapter_name)
    self.model.set_active_adapters(self.adapter_name)

    # Train
    training_args = TrainingArguments(
      output_dir="output_dir",
      overwrite_output_dir=True,
      remove_unused_columns=False,
      dataloader_drop_last=False,
      save_total_limit=1,
      per_device_train_batch_size=self.batch_size,
      # per_device_eval_batch_size=self.batch_size,
      # evaluation_strategy="epoch",
      logging_strategy="epoch",
      learning_rate=self.learning_rate,
      num_train_epochs=self.num_epochs
    )

    trainer = AdapterTrainer(
      model=self.model,
      args=training_args,
      train_dataset=subset_train,
      # eval_dataset=subset_eval,
    )
    trainer.train()


  def predict(self, X):
    # Obtain predict subset
    subset = self.dataset.select(X)
    subset.reset_format()
    subset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(subset, batch_size = self.batch_size)

    # print(f"Adapter Summary AFTER {self.model.adapter_summary()}")
    preds = []
    self.model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.no_grad():
          # Forward pass to compute predictions
          outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
          if self.category == "C":
            outputs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
          preds.append(outputs.detach().cpu().numpy())
    torch.cuda.empty_cache()
    return np.concatenate(preds).reshape(-1, 1)
  
  def predict_proba(self, X):
    predictions = self.predict(X)
    difference = np.ones(predictions.shape) - predictions
    return np.concatenate((difference, predictions), axis=1)
  
  