import os
from typing import Callable, Union, Iterable, Tuple

import pandas as pd
from spacy import Language


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


class TrainDataset:

    def __init__(self,
                 reviews_file: Union[str, os.PathLike],
                 aspect_file: Union[str, os.PathLike]):

        self.reviews = pd.read_csv(reviews_file, sep='\t', header=None, index_col=None, names=['text_id', 'text'])
        self.aspects = pd.read_csv(aspect_file, sep='\t', header=None, index_col=None,
                                   names=['text_id', 'category', 'aspect', 'start', 'end', 'sentiment'])

    def convert_to_bio(self,
                       tokenizer: Callable[[str, bool], Iterable[Iterable[Tuple[str, str, int, int]]]],
                       split_sents: bool = True):
        """
        Converts original datasets to BIO annotation Args: tokenizer: tokenizer that takes text string and
        split_sents param and returns [[(token, pos, char_start, char_end]]
        split_sents: whether to split texts into sentence or not

        Returns: [[(token, pos, bio-tag, char_start, char_end)]]
        """

        full_bio = []
        for _, text in self.reviews.iterrows():
            tokenized = tokenizer(text.text, split_sents)
            text_aspects = self.aspects[self.aspects.text_id == text.text_id].reset_index()

            aspect_idx = 0
            text_bio = []
            prev_is_ent = False  # whether the previous token is an entity or not

            for span_idx, span in enumerate(tokenized):
                span_bio = []

                for token in span:
                    if aspect_idx <= text_aspects.index[-1] and \
                            token[2] >= text_aspects.loc[aspect_idx, 'start'] and \
                            token[3] <= text_aspects.loc[aspect_idx, 'end']:
                        bio_tag = 'B' if not prev_is_ent else 'I'
                        bio_tag = bio_tag + '-' + text_aspects.loc[aspect_idx, 'category']
                        prev_is_ent = True
                    else:
                        bio_tag = 'O'
                        if prev_is_ent:
                            aspect_idx += 1
                        prev_is_ent = False

                    span_bio.append((
                        (text.text_id, span_idx),
                        token[0],
                        token[1],
                        bio_tag,
                        token[2],
                        token[3]))

                if split_sents:
                    text_bio.append(span_bio)
                else:
                    text_bio.extend(span_bio)

            full_bio.extend(text_bio)
        return full_bio