import os
from pathlib import Path
from typing import Union, Iterable, Tuple, Dict

from abstract_pipelines import ABSAPipeline
from preprocessing import ABSADataset
from crf_model import CRFModel
from ats_bert_model import ATSBert


class ThreeShotPipeline:
    def __init__(self,
                 components: Iterable[str],
                 components_config: Dict[str, dict]
                 ):
        self.components = components
        self.components_config = components_config
        self.models = []

    # def initialize_models(self):
    #     models = []
    #     for comp in self.components:
    #         model = globals()[self.components_configs[comp]['class_name']](
    #         self.components_configs[comp]['default_params'])
    #         models.append(model)
    #     return models

    def train(self,
              train_dataset: ABSADataset):
        ...
        # self.models = self.initialize_models()
        # for model in self.models:
        #     model.fit(train_dataset)

    @staticmethod
    def from_checkpoints(components_config: Dict[str, dict],
                         checkpoints_config: dict):
        pipeline = ThreeShotPipeline(list(components_config.keys()), components_config)

        models = []
        for comp, config in components_config.items():
            model = globals()[config['class_name']].from_pretrained(
                **checkpoints_config[comp]
            )
            models.append(model)
        pipeline.models = models
        return pipeline

    def predict(self,
                test_dataset: ABSADataset):
        for model in self.models:
            pred = model.predict(test_dataset)
            setattr(test_dataset, model.output_type, pred)
        return test_dataset

    @staticmethod
    def write_res_files(test_dataset: ABSADataset,
                        res_folder: Union[str, os.PathLike] = './data/res/'):
        res_folder = Path(res_folder).resolve()
        res_folder.mkdir(parents=True, exist_ok=True)

        aspects = test_dataset.ats_input()[['text_id', 'category', 'aspect', 'start', 'end', 'sentiment']]
        aspects.to_csv(Path(res_folder, f'aspects_pred.csv'), sep='\t', header=False, index=False)


if __name__ == '__main__':
    import yaml
    import pandas as pd

    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    pd.set_option('display.max_columns', None)

    pl_config = config['pipelines']['atc+ats+acs']
    components_config = {comp: config for comp, config in config['components'].items() if comp in pl_config['components']}
    checkpoints_config = config['checkpoints']

    pipeline = globals()[pl_config['class_name']].from_checkpoints(components_config, checkpoints_config)

    test_dataset = ABSADataset(config['dataset'], 'test')
    res_dataset = pipeline.predict(test_dataset)
    save = input('Save result? y/n: ')
    if save:
        pipeline.write_res_files(res_dataset)