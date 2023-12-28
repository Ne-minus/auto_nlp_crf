import os
from abc import ABC, abstractmethod
from typing import Union

from preprocessing import ABSADataset


class ABSAPipeline(ABC):
    @abstractmethod
    def train(self,
              train_dataset: ABSADataset,
              train_configs: Union[os.PathLike, str]):
        """
        Train the aspect-based sentiment analysis pipeline.
        """
        ...

    @abstractmethod
    def initialize_models(self):
        """
        Initialize the models and other necessary components for inference using trained artifacts.
        """
        ...

    @abstractmethod
    def from_checkpoints(self, config_path):
        ...

    @abstractmethod
    def predict(self,
                test_dataset: ABSADataset):
        """
        Make predictions for aspect extraction and sentiment classification on a single raw review.

        Parameters:
        - raw_review (str): Raw review text for inference.

        Returns:
        - dict: A dictionary containing extracted aspects and their corresponding sentiment predictions.
        """
        ...


class ABSAComponent(ABC):
    @abstractmethod
    def fit(self,
            train_dataset: ABSADataset):
        ...

    @staticmethod
    @abstractmethod
    def from_pretrained(**kwargs):
        ...

    @abstractmethod
    def predict(self,
                test_dataset: ABSADataset):
        ...