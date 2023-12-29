from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

    output_type = 'ats_input_'

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
                 model_type: str = 'model',
                 prompt_type: int = 1,
                 lang: str = 'ru'
                 ):
        self.model_name = model_name
        self.model_type = model_type
        self.sentiment_map = sentiment_map
        self.prompt_type = prompt_type
        self.lang = lang
        self.load_model()

    def load_model(self):
        if self.model_type == 'pipeline':
            sentiment_analysis = pipeline('text-classification', model=self.model_name)
            self.model = sentiment_analysis
        elif self.model_type == 'model':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to('cpu')

    def fit(self,
            train_dataset: ABSADataset):
        ...

    @staticmethod
    def from_pretrained(model_name: str,
                        model_type: str,
                        sentiment_map: Dict[str, str],
                        prompt_type: int = 1,
                        lang: str = 'ru'):
        model = ATSBert(model_name, sentiment_map, model_type, prompt_type, lang)
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
        data['category_tr'] = data.apply(lambda x: self.cats_transl[x['category']][self.lang], axis=1)
        return data

    def _predict_p(self,
                   inp: str) -> str:
        result = self.model(inp)
        label = self.sentiment_map[result[0]['label']]
        return label

    def _predict_m(self,
                   inp: str) -> str:
        inp = self.tokenizer(inp, padding='max_length', max_length=128, truncation=True, return_tensors='pt').to('cpu')
        with torch.no_grad():
            logits = self.model(**inp).logits.cpu()[0]
            pred = self.sentiment_map[int(np.argmax(logits))]
        return pred

    def _validate_dataset(self,
                         dataset: ABSADataset,
                         part: str):
        missing = self.oblig_attrs[part] - set(dir(dataset))
        if missing:
            raise ValueError(f'Dataset missing the following obligatory attrs: {missing}')

    def predict(self,
                test_dataset: ABSADataset):
        self._validate_dataset(test_dataset, part='inference')
        print('Predicting sentiments...')
        ats_input = test_dataset.ats_input()
        ats_input = self.translate_category(ats_input)
        data = []
        idx = []
        for i, row in tqdm(ats_input.iterrows(), total=ats_input.shape[0]):
            inp = self.fill_prompt(self.prompt_type, row['sent'], row['aspect'], row['category_tr'])
            if self.model_type == 'pipeline':
                result = self._predict_p(inp)
            elif self.model_type == 'model':
                result = self._predict_m(inp)
            data.append(result)
            idx.append(i)
        ats_input['sentiment'] = pd.Series(data, index=idx)
        return ats_input


if __name__ == '__main__':
    import yaml

    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    pd.set_option('display.max_columns', None)

    bert_config = config['components']['bert_ats_finetuned']
    checkpoints = config['checkpoints']
    ats_model = globals()[bert_config['class_name']].from_pretrained(**checkpoints['bert_ats_finetuned'])

    train_dataset = ABSADataset(config['dataset'], 'dev')

    print(ats_model.predict(train_dataset))