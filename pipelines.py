import os
from pathlib import Path
from typing import Union, Iterable, Tuple

from abstract_pipelines import ABSAPipeline
from preprocessing import ABSADataset


class ThreeShotPipeline(ABSAPipeline):
    def __init__(self,
                 pipeline_name: str,
                 pipeline_components: Tuple[str, str, str],
                 components_configs: dict,
                 save_checkpoints: Union[str, os.PathLike] = None,
                 save_intermediate_data: Union[str, os.PathLike] = None):

        self.name = pipeline_name
        self.components = pipeline_components
        self.components_configs = components_configs

        self.models = None

        self.save_checkpoints = save_checkpoints
        self.save_intermediate_data = save_intermediate_data

    def initialize_models(self):
        models = []
        for comp in self.components:
            model = globals()[self.components_configs[comp]['class_name']](
            self.components_configs[comp]['default_params'])
            models.append(model)
        return models

    def train(self,
              train_dataset: ABSADataset,
              configs: dict):
        self.models = self.initialize_models()
        for model in self.models:
            model.fit(train_dataset)

    def from_checkpoints(self,
                         checkpoint_config: dict):
        models = []
        for comp in self.components:
            model = globals()[self.components_configs[comp]['class_name']].from_pretrained(
                **checkpoint_config[comp]
            )
            models.append(model)
        return models

    def predict(self,
                test_dataset: ABSADataset):
        ...