from typing import List

from sklearn.model_selection import train_test_split
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.file import read_json


@register('conv_reader')
class ConvReader(DatasetReader):
    """Class to read training datasets"""

    def read(self, data_path: str):
        if str(data_path).endswith(".pickle"):
            dataset = load_pickle(data_path)
        elif str(data_path).endswith(".json"):
            dataset = read_json(data_path)
        else:
            raise TypeError(f'Unsupported file type: {data_path}')

        processed_samples = []
        for element in dataset:
            processed_samples.append([element, element])

        train_data, test_data = train_test_split(processed_samples, test_size=0.05)
        dataset = {"train": train_data, "valid": test_data, "test": test_data}

        return dataset
