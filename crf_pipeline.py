import os
import pickle
import csv
from typing import Union, Iterable, Tuple, List

from tqdm import tqdm
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics


from abstract_pipelines import ABSAComponent
from preprocessing import ABSADataset


class CRFModel(ABSAComponent):

    oblig_attrs = {
        'train': {'reviews'},
        'inference': {'reviews'}
    }

    def __init__(self,
                 algorithm='lbfgs',
                 max_iterations=100,
                 all_possible_transitions=True,
                 c1=0.3,
                 c2=0.2):

        self.crf_class = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            max_iterations=max_iterations,
            c1=c1,
            c2=c2,
            all_possible_transitions=all_possible_transitions
        )

    @staticmethod
    def from_pretrained(path_to_pretrained: Union[os.PathLike, str]):
        with open(path_to_pretrained, 'rb') as f:
            crf_class = pickle.load(f)

        crf_model = CRFModel()
        crf_model.crf_class = crf_class
        return crf_model

    def get_word_features(self,
                          sentence: List[Tuple[str, str]],
                          i: int) -> dict:
        """
        Generate a dictionary of features for a given word in a sentence.

        Args:
        - sentence (List[Tuple[str, str]]): The input sentence represented as a list of tuples
          where each tuple contains a word and its corresponding part-of-speech tag.
        - i (int): The index of the target word in the sentence.

        Returns:
        dict: A dictionary containing various features for the target word and its context.
        """

        word = sentence[i][0]
        postag = sentence[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }

        if i > 0:
            word1 = sentence[i - 1][0]
            postag1 = sentence[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][0]
            postag1 = sentence[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        return features

    def extract_info(self, text_id, bios, tokens, starts, ends):
        lists = []

        for id, tag in enumerate(bios):
            if tag.startswith('B-') or (tag.startswith('I-') and bios[id-1] == 'O'):
                holder = {}
                holder['text_id'] = text_id
                holder['aspect'] = tag[2:]
                holder['token'] = tokens[id]
                holder['start'] = starts[id]
                holder['end'] = ends[id]

                lists.append(holder)
            elif tag.startswith('I-'):
                if tokens[id] in ',./-:!?':
                    lists[-1]['token'] += tokens[id]
                else:
                    lists[-1]['token'] += f' {tokens[id]}'
                lists[-1]['end'] = ends[id]
                
        return [lst for lst in lists if lst]


    def get_X_features(self,
                       sentence: List[Tuple[str, str]]) -> List[dict]:
        """
        Generate a list of feature dictionaries for each word in a given sentence.
        Args:
        - sentence (List[Tuple[str, str]]): The input sentence represented as a list of tuples
          where each tuple contains a word and its corresponding part-of-speech tag.

        Returns:
        List[dict]: A list of feature dictionaries, where each dictionary corresponds to the features
          of a word in the input sentence.
        """
        return [self.get_word_features(sentence, word) for word in range(len(sentence))]

    def construct_features(self,
                           X_dataset: pd.DataFrame,
                           train: bool = True) -> Tuple[List[List[dict]], List[List[str]] or None]:
        """
        Construct features for CRF from a dataset.

        Args:
        - X_dataset (pd.DataFrame): A DataFrame containing dataset information with columns
          'text_id', 'sent_id', 'token', 'POS' (and 'BIO' if train = True).
        - train (bool): If True, the function includes labels ('BIO') in the output. If False,
          labels are not included in the output.

        Returns:
        Tuple[List[List[dict]], List[List[str]] or None]: A tuple containing two elements:
          - X (List[List[dict]]): A list of feature dictionaries for each word in the input sentences.
          - y (List[list] or None): If `train` is True, a list of BIO labels for each word in
            the input sentences. If `train` is False, this element is None.

        """

        X = []
        y = []
        sentences = X_dataset.groupby(by=['text_id', 'sent_id']).groups
        for _, tokens in tqdm(sentences.items(), position=0, leave=True):
            X.append(self.get_X_features(X_dataset.loc[tokens, ['token', 'POS']].values.tolist()))
            if train:
                y.append(X_dataset.loc[tokens]['BIO'].values.tolist())
        return X, y

    def _validate_dataset(self,
                         dataset: ABSADataset,
                         part: str):
        missing = self.oblig_attrs[part] - set(dir(dataset))
        if missing:
            raise ValueError(f'Dataset missing the following obligatory attrs: {missing}')

    def fit(self,
            train_dataset: ABSADataset) -> None:
        self._validate_dataset(train_dataset, 'train')
        print('Reprocessing your train data...')
        bio_annotation = train_dataset.crf_bio()
        X_train_features, y_train_labels = self.construct_features(bio_annotation)
        self.crf_class.fit(X_train_features, y_train_labels)

    def save_checkpoint(self,
                        to_path: Union[str, os.PathLike]):
        with open(to_path, 'wb') as f:
            pickle.dump(self.crf_class, f)

    def evaluate(self,
                 eval_dataset: ABSADataset) -> Tuple[float, float]:
        print('Reprocessing your eval data...')
        self._validate_dataset(eval_dataset, 'train')
        bio_annotation = eval_dataset.crf_bio()
        X_test_features, y_test_labels = self.construct_features(bio_annotation)
        y_eval = self.crf_class.predict(X_test_features)
        labels = self.crf_class.classes_

        f1 = metrics.flat_f1_score(y_test_labels, y_eval,
                                   average='weighted', labels=labels)
        accuracy = metrics.flat_accuracy_score(y_test_labels, y_eval)
        return f1, accuracy

    def predict(self,
                test_data: ABSADataset):
        print('Reprocessing your test data...')
        self._validate_dataset(test_data, 'inference')
        parsed_test_data = test_data.parsed_reviews()

        featured_test_data, _ = self.construct_features(parsed_test_data, train=False)
        preds = self.crf_class.predict(featured_test_data)

        parsed_test_data = parsed_test_data.groupby(by=['text_id', 'sent_id']).agg(list)
        parsed_test_data['predictions'] = preds
        parsed_test_data.reset_index(level='text_id', inplace=True)

        parsed_test_data['results'] = parsed_test_data.apply(lambda x: self.extract_info(x.text_id, x.predictions, x.token, x.char_start, x.char_end), axis=1)

        results = list(parsed_test_data['results'])

        final = []
        for sent in results:
            if sent:
                final.extend(sent)

        pd.DataFrame(final).to_csv('aspects_predicted.tsv', header=False, index=False, sep='\t')

        return pd.DataFrame(final)


if __name__ == '__main__':

    import yaml
    with open('configs.yml', 'r') as file:
        config = yaml.safe_load(file)

    mode = input('Select mode (train or inference): ')
    crf_config = config['components']['crf']
    crf_model_class = globals()[crf_config['class_name']]

    if mode == 'train':
        # train
        train_dataset = ABSADataset(config['dataset'], 'train', './data/')
        crf = crf_model_class(**crf_config['default_params'])
        crf.fit(train_dataset)

        checkpoint_file = input('Name checkpoint file: ')
        if checkpoint_file:
            crf.save_checkpoint(f'./checkpoints/{checkpoint_file}.sav')

        # eval
        eval_dataset = ABSADataset(config['dataset'], 'dev', './data/')
        f1_eval, acc_eval = crf.evaluate(eval_dataset)
        print(f'F-score on evaluation set: {f1_eval, acc_eval}')

    else:
        # prediction
        pretrained = config['checkpoints']['crf']['path']
        crf = crf_model_class.from_pretrained(pretrained)

        test_dataset = ABSADataset(config['dataset'], 'test')
        print(test_dataset.preprocessed_attrs())
        pred = crf.predict(test_dataset)
        print(pred)