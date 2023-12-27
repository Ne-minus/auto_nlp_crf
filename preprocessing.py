import os
from collections import Counter
from typing import Callable, Union, Iterable, Tuple
from pathlib import Path

import pandas as pd
import spacy
from spacy import Language


nlp = spacy.load("ru_core_news_sm")


def split_data(data: pd.DataFrame, train_idx, test_idx, split_col: str = 'text_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = data[data[split_col].isin(train_idx)]
    test_data = data[data[split_col].isin(test_idx)]
    return train_data, test_data


class SpacyTokenizer:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self,
                 text: str,
                 split_sents: bool = True) -> Iterable[Iterable[Tuple[str, str, int, int]]]:
        parsed = self.nlp(text)
        tokenized = []
        for sent in parsed.sents:
            tokens = [(t.text, t.pos_, t.idx, t.idx + len(t.text)) for t in sent]
            if split_sents:
                tokenized.append(tokens)
            else:
                tokenized.extend(tokens)
        if not split_sents:
            tokenized = [tokenized]
        return tokenized


class ABSADataset:

    spacy_tokenizer = SpacyTokenizer(nlp)

    def __init__(self,
                 dataset_configs: dict,
                 part: str = 'train',
                 save_data: Union[str, os.PathLike] = None):

        self.dataset_configs = dataset_configs[part]
        self.part = part
        self.save_data = save_data

        self.reviews = pd.read_csv(self.dataset_configs['reviews'],
                                   sep='\t', header=None, index_col=None, names=['text_id', 'text'])

        if part == 'train' or part == 'dev':
            self.aspects = pd.read_csv(self.dataset_configs['aspects'], sep='\t', header=None, index_col=None,
                                       names=['text_id', 'category', 'aspect', 'start', 'end', 'sentiment'])
            self.categories = pd.read_csv(self.dataset_configs['categories'], sep='\t', header=None, index_col=None,
                                          names=['text_id', 'category', 'sentiment'])

    def parse_reviews(self):
        parsed = []
        for _, text in self.reviews.iterrows():
            tokenized = ABSADataset.spacy_tokenizer(text.text, True)
            for sent_id, sent in enumerate(tokenized):
                for token in sent:
                    parsed.append((
                        text.text_id,
                        sent_id,
                        *token,
                    ))
        parsed = pd.DataFrame(parsed, columns=['text_id', 'sent_id', 'token', 'POS', 'char_start', 'char_end'])

        if self.save_data:
            parsed.to_csv(Path(self.save_data, f'parsed_{self.part}.csv').resolve(),
                          sep='\t', header=True, index=False)
        return parsed

    def to_crf_bio(self,
                   parsed: pd.DataFrame):
        if not isinstance(self.aspects, pd.DataFrame):
            raise AttributeError('To convert to BIO you should init with aspect_file')

        bio = []
        token_idx = []
        for text_id, token_ids in parsed.groupby(by='text_id').groups.items():
            text_aspects = self.aspects[self.aspects.text_id == text_id].reset_index()

            aspect_idx = 0
            prev_is_ent = False  # whether the previous token is an entity or not
            for i in token_ids:
                token = parsed.loc[i]

                if aspect_idx <= text_aspects.index[-1] and \
                        token.char_start >= text_aspects.loc[aspect_idx, 'start'] and \
                        token.char_end <= text_aspects.loc[aspect_idx, 'end']:
                    bio_tag = 'B' if not prev_is_ent else 'I'
                    bio_tag = bio_tag + '-' + text_aspects.loc[aspect_idx, 'category']
                    prev_is_ent = True
                else:
                    bio_tag = 'O'
                    if prev_is_ent:
                        aspect_idx += 1
                    prev_is_ent = False
                bio.append(bio_tag)
                token_idx.append(i)

        parsed['BIO'] = pd.Series(bio, token_idx)

        if self.save_data:
            parsed.to_csv(Path(self.save_data, f'bio_{self.part}.csv').resolve(),
                          sep='\t', header=True, index=False)
        return parsed

    def bio2aspects(self, bio_annot: pd.DataFrame):
        aspects = []
        aspect_tokens = []
        for idx, token in bio_annot.iterrows():
            token_bio = token['BIO'].split('-')
            if token_bio[0] == 'B':
                if aspect_tokens:
                    aspects.append((
                        ' '.join([t['token'] for t in aspect_tokens]),
                        Counter([t['BIO'].split('-')[1] for t in aspect_tokens]).most_common(1)[0][0],
                        aspect_tokens[0]['char_start'],
                        aspect_tokens[-1]['char_end'],
                        *token.values.tolist()
                    ))
                aspect_tokens = [token]
            elif token_bio[0] == 'I':
                aspect_tokens.append(token)
            elif token_bio[0] == 'O' and aspect_tokens:
                aspects.append((
                    ' '.join([t['token'] for t in aspect_tokens]),
                    Counter([t['BIO'].split('-')[1] for t in aspect_tokens]).most_common(1)[0][0],
                    aspect_tokens[0]['char_start'],
                    aspect_tokens[-1]['char_end'],
                    *token.values.tolist()))
                aspect_tokens = []
        aspects = pd.DataFrame(aspects, columns=['aspect', 'category', 'aspect_start', 'aspect_end', *bio_annot.columns])
        aspects = aspects.drop(['BIO', 'char_start', 'char_end', 'token', 'POS'], axis=1)
        return aspects

if __name__ == '__main__':
    import yaml
    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    part = input('Dataset part (train, dev): ')

    dataset = ABSADataset(config['dataset'], part)

    # parsed = dataset.parse_reviews()
    # save_parsed = input('Save parsed dataset? y/n: ')
    # if save_parsed == 'y':
    #     parsed.to_csv(f'./data/parsed_{part}.csv', sep='\t', header=True, index=False)
    #
    # bio = dataset.to_crf_bio(parsed)
    # save_bio = input('Save bio annotation? y/n: ')
    # if save_bio == 'y':
    #     bio.to_csv(f'./data/bio_{part}.csv', sep='\t', header=True, index=False)

    bio = pd.read_csv(config['dataset'][part]['bio'], sep='\t', header=0)
    res = dataset.bio2aspects(bio)
    print(res.head(30))



