import os
import pickle
from typing import Union, Iterable, Tuple, List

from tqdm import tqdm
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics


from abstract_pipelines import ATCDetection
from preprocessing import ABSADataset


class CRFModel(ATCDetection):

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


    def extract_aspects(self, label: list) -> list:
        """Extracts B- and I- tags and their positions in a sentence.

        Args:
            label (list): sentence in BIO-tagging

        Returns:
            list: list of dictionaries, one dictionary for each aspect with their position in a sentence
        """

        aspects = []
        current_aspect = None
        for idx, label in enumerate(label):
            if label.startswith('B-'):
                current_aspect = label[2:]
                # print(current_aspect)
                aspects.append({'aspect': current_aspect, 'start': idx, 'end': idx})
            elif label.startswith('I-') and current_aspect:
                if aspects and current_aspect == aspects[-1]['aspect']:
                    aspects[-1]['end'] = idx
        return aspects

    def find_aspect_positions(self, sentence: list, aspects: list) -> list:
        """Maps BIO-tagged sentence with an aactual test set.

        Args:
            sentence (list): sentence from processed test set (similar to in line 47)
            aspects (list): extracted from extract_aspects

        Returns:
            list: list of dictionaries, one dictionary for each aspect
        """

        for aspect in aspects:
            start = aspect['start']
            end = aspect['end']
            if start is not None and end is not None:
                tuples = sentence[start:end + 1]
                asp_string = [i[1] for i in tuples]
                text_id = [i[0][0] for i in tuples]
                sent_id = [i[0][1] for i in tuples]
                aspect['string'] = ' '.join(asp_string)
                aspect['text_id'] = text_id[0]
                aspect['sent_id'] = sent_id[0]
                aspect['start'] = sentence[start][-2]
        return aspects

    def extract_position(self) -> list:
        """Exctract start and end positions from raw text for each aspect (not token-, but character-wise)

        Returns:
            list: in a format of test set (without sentimenr)
        """

        results = []
        for id in range(len(self.y_pred)):
            aspects = self.extract_aspects(self.y_pred[id])
            aspects = self.find_aspect_positions(self.test_data[id], aspects)

            for aspect in aspects:
                # text = initial_test[str(aspect['text_id'])]
                start = aspect['start']
                end = start + len(aspect['string'])
                results.append([aspect['text_id'], aspect['aspect'], aspect['string'], start, end])

        # pickle.dump(results, open('./aspects_pred.pkl', 'wb'))
        #
        # with open('./aspects_pred.tsv', 'w', newline='') as file:
        #     writer = csv.writer(file, delimiter='\t')
        #     writer.writerows(results)

        return results

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

    # def get_y_labels(self,
    #                  sentence: Iterable[str]) -> list:
    #     """Reformats dats for testing.
    #
    #     Args:
    #         sentence (list): sentence tokens bio tags
    #
    #     Returns:
    #         list: list of sentences in BIO-tags
    #     """
    #     return [tag for tag in sentence]

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

    def _to_parsed(self,
                   dataset: ABSADataset) -> pd.DataFrame:

        parsed_file = dataset.dataset_configs.get('parsed', None)
        if not parsed_file:
            parsed = dataset.parse_reviews()
        else:
            parsed = pd.read_csv(parsed_file, sep='\t', header=0)
        return parsed

    def _to_bio(self,
                dataset: ABSADataset) -> pd.DataFrame:
        dataset_configs = dataset.dataset_configs
        bio_file = dataset_configs.get('bio', None)
        if not bio_file:
            parsed = self._to_parsed(dataset)
            bio = dataset.to_crf_bio(parsed)
        else:
            bio = pd.read_csv(bio_file, sep='\t', header=0)
        return bio

    def fit(self,
            train_dataset: ABSADataset) -> None:
        print('Reprocessing your train data...')
        bio_annotation = self._to_bio(train_dataset)
        X_train_features, y_train_labels = self.construct_features(bio_annotation)
        self.crf_class.fit(X_train_features, y_train_labels)

    def save_checkpoint(self,
                        to_path: Union[str, os.PathLike]):
        with open(to_path, 'wb') as f:
            pickle.dump(self.crf_class, f)

    def evaluate(self,
                 eval_dataset: ABSADataset) -> Tuple[float, float]:
        print('Reprocessing your eval data...')
        bio_annotation = self._to_bio(eval_dataset)
        X_test_features, y_test_labels = self.construct_features(bio_annotation)
        y_eval = self.crf_class.predict(X_test_features)
        labels = self.crf_class.classes_

        f1 = metrics.flat_f1_score(y_test_labels, y_eval,
                                   average='weighted', labels=labels)
        accuracy = metrics.flat_accuracy_score(y_test_labels, y_eval)
        return f1, accuracy

    def predict_label(self, test_data: ABSADataset) -> list:
        """Makes predictions and reformats them.

        Args:
            test_data (list): test set
            initial_test (list): list of raw texts of a type -- [text_id, raw_text]

        Returns:
            list: predictions
        """
        print('Reprocessing your test data...')
        parsed_test_data = self._to_parsed(test_data)
        print(parsed_test_data.head())
        # self.test_data = test_data
        # X_test_features = [self.get_X_features(s) for s in tqdm(self.test_data, position=0, leave=True)]
        # # y_test_labels = [self.get_y_labels(s) for s in tqdm(self.test_data, position=0, leave=True)]
        #
        # try:
        #     if os.path.exists(self.path) and self.path != '':
        #         classifier = self.precomp_classifier
        #     else:
        #         classifier = self.crf_class
        # except Exception as e:
        #     print(e)
        #     print('You need to provide path to models\' weights or train the model with crf.fit')
        #
        # self.y_pred = classifier.predict(X_test_features)
        #
        # return self.extract_position()

    def predict(self,
                tokenized_test_dataset: pd.DataFrame):
        ...


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
        pretrained = config['checkpoints']['crf']['default']['path']
        crf = crf_model_class.from_pretrained(pretrained)

        test_dataset = ABSADataset(config['dataset'], 'test')
        crf.predict_label(test_dataset)




