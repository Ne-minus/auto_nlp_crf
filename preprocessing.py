import os
from bisect import bisect_left
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

        self._load_preprocessed()

    def _load_preprocessed(self):
        preprocessed = self.dataset_configs.get('preprocessed', [])
        for data_type in preprocessed:
            df = pd.read_csv(preprocessed[data_type], sep='\t', header=0)
            setattr(self, data_type + '_', df)

    def save_preprocessed(self,
                          data_path: Union[str, os.PathLike]):
        preprocessed = self.preprocessed_attrs()
        print(F'Saving the following data: {preprocessed}')
        for attr_name in preprocessed:
            attr_data = getattr(self, attr_name)
            attr_data.to_csv(Path(data_path, f'{attr_name}{self.part}.csv').resolve(),
                             sep='\t', header=True, index=False)

    def preprocessed_attrs(self):
        return [attr for attr in dir(self) if attr[-1] == '_' and attr[-2] != '_']

    def parsed_reviews(self):
        parsed_ = getattr(self, 'parsed_', None)
        if parsed_ is None:
            return self._parse_reviews()
        return parsed_

    def _parse_reviews(self):
        print('Parsing reviews...')

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
        setattr(self, 'parsed_', parsed)
        return parsed

    def crf_bio(self):
        bio_ = getattr(self, 'bio_', None)
        if bio_ is None:
            parsed = self.parsed_reviews()
            return self._to_crf_bio(parsed)
        return bio_

    def _to_crf_bio(self,
                    parsed: pd.DataFrame):
        if not isinstance(self.aspects, pd.DataFrame):
            raise AttributeError('To convert to BIO you should init with aspect_file')

        print('Converting to BIO...')
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
        setattr(self, 'bio_', parsed)
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

    def _find_sent_offset(self, parsed: pd.DataFrame):
        sent_offsets = []
        sents = parsed.groupby(['text_id', 'sent_id']).groups
        for (text_id, sent_id), token_idxs in sents.items():
            sent_start = parsed.loc[token_idxs[0], 'char_start']
            sent_end = parsed.loc[token_idxs[-1], 'char_end']
            sent_offsets.append((text_id, sent_id, sent_start, sent_end))
        return pd.DataFrame(sent_offsets, columns=['text_id', 'sent_id', 'sent_start', 'sent_end'])

    def parsed2bertinput(self, parsed: pd.DataFrame):
        bert_input = []
        sent_offset = self._find_sent_offset(parsed)

        for text_id, aspect_ids in self.aspects.groupby('text_id').groups.items():
            text_sents = sent_offset[sent_offset['text_id'] == text_id]
            review_text = self.reviews[self.reviews['text_id'] == text_id]['text'].values[0]
            sent_ends = text_sents['sent_end'].tolist()
            for asp_id in aspect_ids:
                aspect = self.aspects.loc[asp_id]
                aspect_sent = bisect_left(sent_ends, aspect['start'])
                aspect_sent_id = text_sents[text_sents['sent_id'] == aspect_sent].index.tolist()[0]
                sent_info = text_sents.loc[aspect_sent_id]
                bert_input.append((
                        aspect_sent,
                        review_text[sent_info['sent_start']: sent_info['sent_end']],
                        *aspect.tolist()
                ))
        return pd.DataFrame(bert_input, columns=['sent_id', 'sentence', *self.aspects.columns])


if __name__ == '__main__':
    import yaml
    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    part = input('Dataset part (train, dev, text): ')

    dataset = ABSADataset(config['dataset'], part)

    # parsed
    parsed = dataset.parsed_reviews()

    if part == 'train' or part == 'dev':
        # bio
        bio = dataset.crf_bio()

        # # bert input1
        # bert_input = dataset.parsed2bertinput(parsed)
        # print(bert_input.head())
        # save_bert_input = input('Save bert input annotation? y/n: ')
        # if save_bert_input == 'y':
        #     bert_input.to_csv(f'./data/bert_input_{part}.csv', sep='\t', header=True, index=False)

    save_data = input('Save all preprocessed files? y/n: ')
    if save_data == 'y':
        dataset.save_preprocessed('./data')


