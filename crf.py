from itertools import chain

import nltk
import csv
import json
import tqdm
import os
from tqdm import tqdm
import pickle
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

nltk.download('averaged_perceptron_tagger_ru')
# nltk.download('universal_tagset')



class CRF_model:
    def __init__(self, algorithm, path_to_weights=None):
        self.algo = algorithm
        if path_to_weights:
            self.path = path_to_weights
            self.precomp_classifier = pickle.load(open(self.path, 'rb'))
        # else:
        #     self.path = ''

        # self.precomp_classifier = pickle.load(open(self.path, 'rb'))
     
    def get_pos_tag(self, word: str) -> str:
        """Generates pos tag if exists.

        Args:
            word (str): word itself
        Returns:
            str: UD pos tag or empty
        """        
        
        try:
            tag = nltk.pos_tag([word], lang='rus')[0][1]
            # print(tag)
        except Exception as e:
            tag = ''
            # print(tag)
        return tag
    
    def get_word_features(self, sentence: list, i: int) -> dict:
        """Extracts features of a word and its neighbours.

        Args:
            sentence (list): includes text_id, sentence_id, word, BIO-tag, start, end
            i (int): word id

        Returns:
            dict: ditctionary of features for a given word
        """      

        word = sentence[i][1]
        # postag = self.get_pos_tag(word)
        postag = sentence[i][2]

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
            word1 = sentence[i-1][1]
            # postag1 =  self.get_pos_tag(word1)
            postag1 = sentence[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence)-1:
            word1 = sentence[i+1][1]
            # postag1 = self.get_pos_tag(word1)
            postag1 = sentence[i+1][1]
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
                asp_string = [i[1]for i in tuples]
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
                end = start+len(aspect['string'])
                results.append([aspect['text_id'], aspect['aspect'], aspect['string'], start, end])
            
        pickle.dump(results, open('./aspects_pred.pkl', 'wb'))

        with open('./aspects_pred.tsv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(results)

        return results
    
    def get_X_features(self, sentence: list) -> list:
        """Refromats data for training.

        Args:
            sentence (list): sentence from a test set

        Returns:
            list: list of dicts with features
        """        

        return [self.get_word_features(sentence, word) for word in range(len(sentence))]
    
    def get_y_labels(self, sentence: list) -> list:
        """Reformats dats for testing.

        Args:
            sentence (list): sentence from a test set

        Returns:
            list: list of sentences in BIO-tags
        """        
        # print(sentence[0][2].split('-')[0] )
        return [word[3] for word in sentence]   
    
    def fit(self, train_data: list) -> None:

        print('Reprocessing your train data...')
        self.X_train_features = [self.get_X_features(s) for s in tqdm(train_data, position=0, leave=True)]
        self.y_train_labels = [self.get_y_labels(s) for s in tqdm(train_data, position=0, leave=True)]
        
        self.crf_class = sklearn_crfsuite.CRF(
            algorithm=self.algo,
            max_iterations=100,
            c1=0.3,
            c2=0.2,
            all_possible_transitions=True
        )
        self.crf_class.fit(self.X_train_features, self.y_train_labels)
        pickle.dump(self.crf_class, open('./crf_weights_ud+positions.sav', 'wb'))
        self.labels = list(self.crf_class.classes_)
        # self.labels.remove('O')  
        # print(self.labels)
        
    def evaluate(self, eval_data: list) -> float:
        print('Reprocessing your eval data...')
        X_test_features = [self.get_X_features(s) for s in tqdm(eval_data, position=0, leave=True)]
        y_test_labels = [self.get_y_labels(s) for s in tqdm(eval_data, position=0, leave=True)]
        y_eval = self.precomp_classifier.predict(X_test_features)

        f1 = metrics.flat_f1_score(y_test_labels, y_eval,
                              average='weighted', labels=self.labels)
        accuracy = metrics.flat_accuracy_score(y_test_labels, y_eval)
        return f1, accuracy
        
    def predict_label(self, test_data:list) -> list:
        """Makes predictions and reformats them.

        Args:
            test_data (list): test set
            initial_test (list): list of raw texts of a type -- [text_id, raw_text]

        Returns:
            list: predictions
        """
        print('Reprocessing your test data...')
        self.test_data = test_data
        X_test_features = [self.get_X_features(s) for s in tqdm(self.test_data, position=0, leave=True)]
        y_test_labels = [self.get_y_labels(s) for s in tqdm(self.test_data, position=0, leave=True)]
        
        try:
            if os.path.exists(self.path) and self.path != '':
                classifier = self.precomp_classifier
            else:
                classifier = self.crf_class
        except Exception as e:
            print(e)
            print('You need to provide path to models\' weights or train the model with crf.fit')
            
        self.y_pred = classifier.predict(X_test_features)

        return self.extract_position()


if __name__ == '__main__':
        train_set = json.load(open('./data/bio_train.json', 'rb'))
        eval_set = json.load(open('./data/bio_dev.json', 'rb'))
        test_set = json.load(open('./data/bio_dev.json', 'rb'))
        
        path_to_weights = './crf_weights.sav'
        
        crf = CRF_model('lbfgs', path_to_weights=path_to_weights)
        crf.fit(train_set)
        f1_eval, acc_eval = crf.evaluate(eval_set)
        resultsiks = crf.predict_label(test_set)

        print(f'F-score on evaluation set: {f1_eval, acc_eval}')