import torch
import pickle
import warnings

import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AnnotationClassifier(nn.Module):
    def __init__(self, model_name:str, max_len:int, n_classes:int, device):
        super(AnnotationClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.out_features = self.model.config.hidden_size
        self.n_classes = n_classes
        self.device = device
        if self.n_classes == 1:
            self.model.classifier = nn.Sequential(
                nn.Linear(self.out_features, self.n_classes), nn.Sigmoid()
            )
        else:
            self.model.classifier = nn.Linear(self.out_features, self.n_classes)
        self.model = self.model.to(self.device)

    def fit(self):
        self.model = self.model.train()

    def eval(self):
        self.model = self.model.eval()

    def __call__(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.n_classes == 1:
            outputs = outputs.logits.sigmoid()
            outputs = outputs.view(-1).view(-1)
        else:
            outputs = outputs.logits
        return outputs

    def __str__(self):
        return str(self.model)

    def save(self, filename:str):
        torch.save(self.state_dict(), ('models/' + filename + '.pt'))

    def load(self, filename:str):
        self.load_state_dict(torch.load(filename, map_location=self._device))

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        return torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
