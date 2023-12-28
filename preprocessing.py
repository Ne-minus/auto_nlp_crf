import os
from bisect import bisect_left
from collections import Counter
from typing import Callable, Union, Iterable, Tuple
from pathlib import Path

import pandas as pd
import spacy
from spacy import Language


nlp = spacy.load("ru_core_news_sm")

TRUE_ASPECTS = ['text_id', 'aspect', 'category', 'start', 'end', 'sentiment']
TRUE_REVIEWS = ['text_id', 'text']
TRUE_CATEGORIES = ['text_id', 'category', 'sentiment']

# TRUE_REVIEWS -> PARSED_REVIEWS +
PARSED_REVIEWS = ['text_id', 'sent_id', 'token', 'POS', 'start', 'end']
# TRUE_REVIEWS -> SENT_INFO  +
SENT_INFO = ['text_id', 'sent_id', 'sent', 'start', 'end']
# (SENT_INFO, TRUE_ASPECTS) -> PARSED_ASPECTS  +                               # optional
PARSED_ASPECTS = ['text_id', 'sent_id', 'aspect', 'category', 'start', 'end', 'sentiment']
# (SENT_INFO, PARSED_ASPECTS) -> ATS_FORMAT  +                               # optional   # new
ATS_FORMAT = ['text_id', 'sent_id', 'aspect', 'category', 'start', 'end', 'sentiment', 'sent']

# (PARSED_REVIEWS, TRUE_ASPECTS) -> BIO +
BIO = ['text_id', 'sent_id', 'token', 'POS', 'start', 'end', 'BIO']

# (PARSED_REVIEWS, TRUE_CATEGORIES) -> TEXT_SENTIMENT
TEXT_SENTIMENT = ['text_id', 'text', 'category', 'sentiment']

# CRF
# (PARSED_REVIEWS, TRUE_ASPECTS) -> BIO -> TRAIN
# REVIEWS -> PARSED_REVIEWS -> predict -> PARSED_ASPECTS (without sentiment)

# ATS
# ATS_FORMAT (with sentiment) -> TRAIN
# (SENT_INFO, PARSED_ASPECTS (without sentiment)) -> ATS_FORMAT (without sentiment) -> PREDICT -> ATS_FORMAT (with sentiment)

# ACS_ALGO
# ... -> TRAIN
# PARSED_ASPECTS (with sentiment) -> PREDICT -> TRUE_CATEGORIES

# ACS_BERT
# (PARSED_REVIEWS, TRUE_CATEGORIES) -> TEXT_SENTIMENT -> TRAIN
# PARSED_ASPECT_SENTIMENTS -> PREDICT -> TRUE_CATEGORIES


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
        preprocessed = self.dataset_configs.get('preprocessed', None)
        if preprocessed:
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
            return self.parse_reviews()
        return parsed_

    def parse_reviews(self):
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
        parsed = pd.DataFrame(parsed, columns=['text_id', 'sent_id', 'token', 'POS', 'start', 'end'])
        setattr(self, 'parsed_', parsed)
        print('"parsed_" attribute created')
        return parsed

    def sent_info(self):
        sents_ = getattr(self, 'sents_', None)
        if sents_ is None:
            return self.group_sent_info()
        return sents_

    def group_sent_info(self):
        print("Creating sentence info... ")
        parsed = self.parsed_reviews()
        sent_info = []
        for text_id, sent_idx in parsed.groupby(['text_id']).groups.items():
            text_info = parsed.loc[sent_idx]
            text = self.reviews[self.reviews['text_id'] == text_id]['text'].tolist()[0]
            for sent_id, token_idx in text_info.groupby(['sent_id']).groups.items():
                sent_start = parsed.loc[token_idx[0], 'start']
                sent_end = parsed.loc[token_idx[-1], 'end']
                sent = text[sent_start: sent_end]
                sent_info.append((
                    text_id,
                    sent_id,
                    sent,
                    sent_start,
                    sent_end
                ))
        sent_info = pd.DataFrame(sent_info, columns=['text_id', 'sent_id', 'sent', 'start', 'end'])
        setattr(self, 'sents_', sent_info)
        print('"sents_" attribute created')
        return sent_info

    def crf_bio(self):
        bio_ = getattr(self, 'bio_', None)
        if bio_ is None:
            parsed = self.parsed_reviews()
            return self._to_crf_bio(parsed)
        return bio_

    def _to_crf_bio(self,
                    parsed: pd.DataFrame):
        # (PARSED_REVIEWS, PARSED_ASPECTS) -> BIO
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
        print('"bio_ attribute created')
        return parsed

    def _find_aspect_sent(self,
                          aspects: pd.DataFrame,
                          sents: pd.DataFrame):
        print('Matching aspects with sents... ')
        aspect_sents = []
        sent_ends = {text_id: sents.loc[idx]['end'].tolist() for text_id, idx in sents.groupby('text_id').groups.items()}
        for _, row in aspects.iterrows():
            sent_id = bisect_left(sent_ends[row['text_id']], row['start'])
            aspect_sents.append(sent_id)
        aspects['sent_id'] = aspect_sents
        setattr(self, 'parsed_aspects_', aspects)
        print('"parsed_aspects_" attribute created')
        return aspects

    def parsed_aspects(self):
        parsed_aspects_ = getattr(self, 'parsed_aspects_', None)
        if parsed_aspects_ is None:
            return self._find_aspect_sent(self.aspects, self.sent_info())
        return parsed_aspects_

    def _add_sent_text(self,
                       parsed_aspects: pd.DataFrame,
                       sents: pd.DataFrame):
        print("Adding sentence text to aspects...")
        merged = pd.merge(parsed_aspects, sents[['text_id', 'sent_id', 'sent']])
        setattr(self, 'ats_input_', merged)
        print('"ats_input_" attribute created')
        return merged

    def ats_input(self):
        ats_input_ = getattr(self, 'ats_input_', None)
        if ats_input_ is None:
            return self._add_sent_text(self.parsed_aspects(), self.sent_info())
        return ats_input_
    #
    # def bio2aspects(self, bio_annot: pd.DataFrame):
    #     aspects = []
    #     aspect_tokens = []
    #     for idx, token in bio_annot.iterrows():
    #         token_bio = token['BIO'].split('-')
    #         if token_bio[0] == 'B':
    #             if aspect_tokens:
    #                 aspects.append((
    #                     ' '.join([t['token'] for t in aspect_tokens]),
    #                     Counter([t['BIO'].split('-')[1] for t in aspect_tokens]).most_common(1)[0][0],
    #                     aspect_tokens[0]['char_start'],
    #                     aspect_tokens[-1]['char_end'],
    #                     *token.values.tolist()
    #                 ))
    #             aspect_tokens = [token]
    #         elif token_bio[0] == 'I':
    #             aspect_tokens.append(token)
    #         elif token_bio[0] == 'O' and aspect_tokens:
    #             aspects.append((
    #                 ' '.join([t['token'] for t in aspect_tokens]),
    #                 Counter([t['BIO'].split('-')[1] for t in aspect_tokens]).most_common(1)[0][0],
    #                 aspect_tokens[0]['char_start'],
    #                 aspect_tokens[-1]['char_end'],
    #                 *token.values.tolist()))
    #             aspect_tokens = []
    #     aspects = pd.DataFrame(aspects, columns=['aspect', 'category', 'aspect_start', 'aspect_end', *bio_annot.columns])
    #     aspects = aspects.drop(['BIO', 'char_start', 'char_end', 'token', 'POS'], axis=1)
    #     return aspects

    # def _find_sent_offset(self, parsed: pd.DataFrame):
    #     sent_offsets = []
    #     sents = parsed.groupby(['text_id', 'sent_id']).groups
    #     for (text_id, sent_id), token_idxs in sents.items():
    #         sent_start = parsed.loc[token_idxs[0], 'char_start']
    #         sent_end = parsed.loc[token_idxs[-1], 'char_end']
    #         sent_offsets.append((text_id, sent_id, sent_start, sent_end))
    #     return pd.DataFrame(sent_offsets, columns=['text_id', 'sent_id', 'sent_start', 'sent_end'])

    # def parsed2bertinput(self, parsed: pd.DataFrame):
    #     bert_input = []
    #     sent_offset = self._find_sent_offset(parsed)
    #
    #     for text_id, aspect_ids in self.aspects.groupby('text_id').groups.items():
    #         text_sents = sent_offset[sent_offset['text_id'] == text_id]
    #         review_text = self.reviews[self.reviews['text_id'] == text_id]['text'].values[0]
    #         sent_ends = text_sents['sent_end'].tolist()
    #         for asp_id in aspect_ids:
    #             aspect = self.aspects.loc[asp_id]
    #             aspect_sent = bisect_left(sent_ends, aspect['start'])
    #             aspect_sent_id = text_sents[text_sents['sent_id'] == aspect_sent].index.tolist()[0]
    #             sent_info = text_sents.loc[aspect_sent_id]
    #             bert_input.append((
    #                     aspect_sent,
    #                     review_text[sent_info['sent_start']: sent_info['sent_end']],
    #                     *aspect.tolist()
    #             ))
    #     return pd.DataFrame(bert_input, columns=['sent_id', 'sentence', *self.aspects.columns])


if __name__ == '__main__':
    import yaml
    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)
    pd.set_option('display.max_columns', None)

    part = input('Dataset part (train, dev, text): ')

    dataset = ABSADataset(config['dataset'], part)

    # print('parsed_reviews')
    # print(dataset.parsed_reviews().head())
    #
    # print()
    # print('parsed_aspects')
    # print(dataset.parsed_aspects().head())
    #
    # print()
    # print('sentence_info')
    # print(dataset.sent_info().head())

    if part == 'train' or part == 'dev':
        # print()
        # print('bio')
        # print(dataset.crf_bio().head())

        print()
        print('ats_input')
        print(dataset.ats_input().head())

    print(dataset.preprocessed_attrs())

    save_data = input('Save all preprocessed files? y/n: ')
    if save_data == 'y':
        dataset.save_preprocessed('./data/preprocessed/')


