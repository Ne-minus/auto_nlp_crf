from typing import List

import pandas as pd

from preprocessing import ABSADataset
from abstract_pipelines import ABSAComponent


class ACSAlgo(ABSAComponent):
    output_type = 'categories'
    oblig_attrs = 'ats_input_'

    def __init__(self,
                 categories: List[str]):
        self.categories = categories
        ...

    def fit(self,
            train_dataset: ABSADataset):
        ...

    @staticmethod
    def from_pretrained(**kwargs):
        model = ACSAlgo(**kwargs)
        return model

    def _validate_dataset(self,
                         dataset: ABSADataset,
                         part: str):
        missing = self.oblig_attrs[part] - set(dir(dataset))
        if missing:
            raise ValueError(f'Dataset missing the following obligatory attrs: {missing}')

    def predict(self,
                test_dataset: ABSADataset):
        print("Predicting category sentiment...")
        reviews = test_dataset.reviews
        ref_table = pd.DataFrame([(i, cat) for i in reviews['text_id'] for cat in self.categories],
                                 columns=['text_id', 'category'])
        aspects = test_dataset.ats_input()
        pred_sent = aspects.groupby(['text_id', 'category'])['sentiment'].agg(
            lambda x: list(pd.Series.mode(x))[-1]
        ).reset_index()
        cat_table = pd.merge(ref_table, pred_sent, how='left', on=['text_id', 'category']).fillna('absence')
        return cat_table


if __name__ == '__main__':
    import yaml
    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    import pandas as pd
    pd.set_option('display.max_columns', None)

    model_config = config['components']['algo_acs']
    checkpoints = config['checkpoints']
    ats_model = globals()[model_config['class_name']].from_pretrained(**checkpoints['algo_acs'])

    dev_dataset = ABSADataset(config['dataset'], 'dev')
    print(ats_model.predict(dev_dataset))
