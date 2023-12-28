from typing import Dict


import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from abstract_pipelines import ABSAComponent, ABSADataset


rubert_sent_map = {'NEGATIVE': 'negative',
                   'NEUTRAL': 'neutral',
                   'POSITIVE': 'positive'}

multi_sent_map = {'1 star': 'negative',
                  '2 stars': 'negative',
                  '3 stars': 'neutral',
                  '4 stars': 'positive',
                  '5 stars': 'positive'}


class ATSBert(ABSAComponent):

    oblig_attrs = {
        'train': {'reviews', 'aspects'},
        'inference': {'reviews', 'parsed_aspects_'}
    }

    cats_transl = {
        'Whole': {'ru': 'впечатление в целом', 'en': 'Whole'},
        'Service': {'ru': 'сервис', 'en': 'Service'},
        'Food': {'ru': 'еда', 'en': 'Food'},
        'Interior': {'ru': 'интерьер', 'en': 'Interior'},
        'Price': {'ru': 'цена', 'en': 'Price'}
    }

    def __init__(self,
                 model_name: str,
                 sentiment_map: Dict[str, str],
                 prompt_type: int = 1,
                 lang: str = 'ru'
                 ):
        self.model_name = model_name
        self.sentiment_map = sentiment_map
        self.prompt_type = prompt_type
        self.lang = lang
        self.load_model()

    def load_model(self):
        sentiment_analysis = pipeline('text-classification', model=self.model_name)
        self.model = sentiment_analysis

    def fit(self,
            train_dataset: ABSADataset):
        ...

    @staticmethod
    def from_pretrained(model_name: str,
                        sentiment_map: Dict[str, str],
                        prompt_type: int = 1,
                        lang: str = 'ru'):
        model = ATSBert(model_name, sentiment_map, prompt_type, lang)
        return model

    @staticmethod
    def fill_prompt(prompt_type, sent, text=None, aspect=None, category=None):
        if prompt_type == 1:
            prompt = f'{sent} [SEP] {aspect}, {category}'
        elif prompt_type == 2:
            prompt = f"Определи тональность аспекта '{aspect}' в следующем тексте: '{sent}'"
        elif prompt_type == 3:
            prompt = sent
        return prompt

    def translate_category(self,
                           data: pd.DataFrame):
        data['category'] = data.apply(lambda x: self.cats_transl[x['category']][self.lang], axis=1)
        return data

    def _predict(self,
                 ats_input: pd.DataFrame):
        data = []
        print('Predicting sentiments...')
        for _, row in tqdm(ats_input.iterrows(), total=ats_input.shape[0]):
            result = self.model(self.fill_prompt(self.prompt_type, row['sent'], row['aspect'], row['category']))
            data.append(self.sentiment_map[result[0]['label']])
        ats_input['sentiment'] = data
        return ats_input

    def _validate_dataset(self,
                         dataset: ABSADataset,
                         part: str):
        missing = self.oblig_attrs[part] - set(dir(dataset))
        if missing:
            raise ValueError(f'Dataset missing the following obligatory attrs: {missing}')

    def predict(self,
                test_dataset: ABSADataset):
        self._validate_dataset(test_dataset, part='inference')
        ats_input = test_dataset.ats_input()
        ats_input['category'] = self.translate_category(ats_input)['category']
        return self._predict(ats_input)


if __name__ == '__main__':
    import yaml

    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    bert_config = config['components']['bert_ats']
    checkpoints = config['checkpoints']
    ats_model = globals()[bert_config['class_name']].from_pretrained(**checkpoints['bert_ats'])

    train_dataset = ABSADataset(config['dataset'], 'dev')

    print(ats_model.predict(train_dataset))