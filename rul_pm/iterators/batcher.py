import math
from typing import Optional

import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.transformation.transformers import Transformer, simple_pipeline
from rul_pm.transformation.utils import PandasToNumpy
from tqdm.auto import tqdm


class Batcher:
    def __init__(self,
                 iterator: WindowedDatasetIterator,
                 batch_size: int,
                 restart_at_end: bool = True):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False

    def __len__(self):
        return math.ceil(len(self.iterator) / self.batch_size)

    def __iter__(self):
        self.iterator.__iter__()
        return self

    def __next__(self):
        X = []
        y = []
        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()
            else:
                raise StopIteration
        try:
            for _ in range(self.batch_size):
                X_t, y_t = next(self.iterator)
                X.append(np.expand_dims(X_t, axis=0))
                y.append(np.expand_dims(y_t, axis=0))

        except StopIteration:
            pass
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X.astype(np.float32), y.astype(np.float32)


def get_batcher(dataset: AbstractLivesDataset,
                window: int,
                batch_size: int,
                transformer: Transformer,
                step: int,
                output_size: int = 1,
                shuffle: bool = False,
                restart_at_end: bool = True,
                cache_size: int = 20,
                evenly_spaced_points: Optional[int] = None) -> Batcher:
    iterator = WindowedDatasetIterator(dataset,
                                       window,
                                       transformer,
                                       step=step,
                                       output_size=output_size,
                                       shuffle=shuffle,
                                       cache_size=cache_size,
                                       evenly_spaced_points=evenly_spaced_points)
    b = Batcher(iterator, batch_size, restart_at_end)
    return b


def dataset_map(fun, dataset, step, transformer, window):
    batcher = get_batcher(dataset,
                          window,
                          512,
                          transformer,
                          step,
                          shuffle=False,
                          restart_at_end=False)
    for X, y in tqdm(batcher):
        fun(X, y)


def get_features(dataset, step, window, features):
    t = simple_pipeline(features)
    data = {f: [] for f in features}

    def populate_data(X, y):
        for i, f in enumerate(features):
            data[f].extend(np.squeeze(y[:, i]).tolist())
    t = Transformer(
        features,
        t,
        transformerY=PandasToNumpy()
    )
    dataset_map(populate_data, dataset, step, t, window)
    return data
