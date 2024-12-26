import torch
import pickle
import warnings

import numpy as np

from torch import nn

from Cleaner import Cleaner
from BookRegressor import BookRegressor
from AnnotationClassifier import AnnotationClassifier

class MicroService:
    def __init__(self, cleaner, annotationclassifier, bookregressor, device):
        self._cleaner = cleaner
        self._annotationclassifier = annotationclassifier
        self._bookregressor = bookregressor
        self._device = device

    def _str_transform(self, string, size):
        for trash in ['\n', '\t']:
            string = string.replace(trash, '')
        need_grow = 0 if len(string) >= size else size - len(string)
        for _ in range(need_grow):
            string += ' '
        return string[:512]

    def __call__(self, dataframe: pd.DataFrame):
        annotation = [self._str_transform(text) for text in tqdm(df['annotation'])]
        rate = [ 10 if self._annotationclassifier.predict(text) else 0 for text in tqdm(annotation)]
        rateSize = [ 9 if r == 10 else 0 for r in tqdm(rate) ]

        dataframe['rate'] = rate
        dataframe['rateSize'] = rateSize

        clean_dataframe = self._cleaner(dataframe)
        X_poly = self._cleaner.transform(clean_dataframe)
        X = torch.tensor(X_poly, dtype=torch.float32, device=self._device)

        pred = self._bookregressor(X)
        return pred
