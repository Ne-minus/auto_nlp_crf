from itertools import chain

import nltk
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



class CRF_model:
    def __init__(self, algorithm, path_to_weights=None):
        self.algo = algorithm
        if path_to_weights:
            self.path = path_to_weights
        else:
            self.path = ''
     
    def get_pos_tag(self, word):
        try:
            tag = nltk.pos_tag([word])[0][1]
        except:
            tag = ''
        return tag
    
    def get_word_features(self, sentence, i):
        word = sentence[i][1]
            
        postag = self.get_pos_tag(word)

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
            postag1 =  self.get_pos_tag(word1)
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
            postag1 = self.get_pos_tag(word1)
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
        
    def extract_aspects(self, label):
        aspects = []
        current_aspect = None
        for idx, label in enumerate(label):
            if label.startswith('B-'):
                current_aspect = label[2:]
                aspects.append({'aspect': current_aspect, 'start': idx, 'end': idx})
            elif label.startswith('I-') and current_aspect:
                if aspects and current_aspect == aspects[-1]['aspect']:
                    aspects[-1]['end'] = idx
        return aspects   

    def find_aspect_positions(self, sentence, aspects):
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
        return aspects
 
    def extract_position(self, initial_test):
        results = []
        for id in range(len(self.y_pred)):
            aspects = self.extract_aspects(self.y_pred[id])
            aspects = self.find_aspect_positions(self.test_data[id], aspects)
            
            for aspect in aspects:
                text = initial_test[str(aspect['text_id'])]
                start = text.find(aspect['string'])                
                end = start+len(aspect['string'])
                results.append([aspect['text_id'], aspect['aspect'], aspect['string'], start, end])
            
        pickle.dump(results, open('./aspects.pkl', 'wb'))
        return results
    
    def get_X_features(self, sentence):
        return [self.get_word_features(sentence, word) for word in range(len(sentence))]
    
    def get_y_labels(self, sentence):
        return [word[2] for word in sentence]   
    
    def fit(self, train_data):
        print('Reprocessing your train data...')
        self.X_train_features = [self.get_X_features(s) for s in tqdm(train_data, position=0, leave=True)]
        self.y_train_labels = [self.get_y_labels(s) for s in tqdm(train_data, position=0, leave=True)]
        
        self.crf_class = sklearn_crfsuite.CRF(
            algorithm=self.algo,
            max_iterations=100,
            c1=0.1,
            c2=0.1,
            all_possible_transitions=True
        )
        self.crf_class.fit(self.X_train_features, self.y_train_labels)
        pickle.dump(self.crf_class, open('./crf_weights.sav', 'wb'))
        self.labels = list(self.crf_class.classes_).remove('O')  
        
    def evaluate(self, eval_data):
        print('Reprocessing your eval data...')
        X_test_features = [self.get_X_features(s) for s in tqdm(eval_data, position=0, leave=True)]
        y_test_labels = [self.get_y_labels(s) for s in tqdm(eval_data, position=0, leave=True)]
        y_eval = self.crf_class.predict(X_test_features)
        f1 = metrics.flat_f1_score(y_test_labels, self.y_pred,
                              average='weighted', labels=self.labels)
        return f1
        
    def predict_label(self, test_data, initial_test):
        print('Reprocessing your test data...')
        self.test_data = test_data
        X_test_features = [self.get_X_features(s) for s in tqdm(self.test_data, position=0, leave=True)]
        y_test_labels = [self.get_y_labels(s) for s in tqdm(self.test_data, position=0, leave=True)]
        
        try:
            if os.path.exists(self.path) and self.path != '':
                classifier = pickle.load(open(self.path, 'rb'))
            else:
                classifier = self.crf_class
        except Exception as e:
            print(e)
            print('You need to provide path to models\' weights or train the model with crf.fit')
            
        self.y_pred = classifier.predict(X_test_features)

        return self.extract_position(initial_test)


if __name__ == '__main__':
        train_set = pickle.load(open('./bio_corpus_train.pkl', 'rb'))
        eval_set = pickle.load(open('./bio_corpus_dev.pkl', 'rb'))
        test_set = pickle.load(open('./bio_corpus_dev.pkl', 'rb'))
        initial_test = pickle.load(open('./dev_id_text.pkl', 'rb'))
        
        path_to_weights = './crf_weights.sav'
        
        crf = CRF_model('lbfgs', path_to_weights=path_to_weights)
        resultsiks = crf.predict_label(test_set, initial_test)

        print(resultsiks[0:3])