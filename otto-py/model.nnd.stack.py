import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction
from sklearn.cross_validation import StratifiedKFold
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh, linear
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
from nolearn.lasagne import NeuralNet
from joblib import Parallel, delayed
#from multiprocessing import Pool
import sys
import getopt
import theano
import random
import os
from sklearn.metrics import log_loss

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        setattr(nn, self.name, new_value)
        #getattr(nn, self.name).set_value(new_value)
        
def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.seed(seed=2015)
    np.random.shuffle(X)
    X, labels, ids = X[:, 1:-1].astype(np.float32), X[:, -1], X[:, 0].astype(str)
    tfidf = feature_extraction.text.TfidfTransformer()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    X = tfidf.fit_transform(X).toarray()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, ids, encoder, scaler, tfidf
    
def load_test_data(path, scaler, tfidf):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = tfidf.transform(X).toarray()
    X = scaler.fit_transform(X)
    return X, ids
    
def make_submission(clf, X_test, ids, encoder, name='predictions/nn2.cv.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

def NeuralNetConstructor():
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('dense2', DenseLayer),
               ('dropout3', DropoutLayer),           
               ('output', DenseLayer)]


    net0 = NeuralNet(layers=layers0,
                 
                     input_shape=(None, num_features),
                     #dense100_num_units=50,
                     #dense100_nonlinearity=linear,
                     dense0_num_units=500,
                     dropout1_p=0.5,
                     dense1_num_units=300,
                     dropout2_p=0.1,
                     dense2_num_units=100,
                     dropout3_p=0.1,   
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=nesterov_momentum,
                     update_learning_rate=0.01,
                     update_momentum=0.9,
                     #on_epoch_finished=[
                     #    AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                     #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     #    ],
                 
                     eval_size=0.01,
                     verbose=0,
                     max_epochs=70)
    return net0

def compute_fold(train_index, valid_index, X, y, X_test, ids_train, ids_test):
    net0 = NeuralNetConstructor()
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    index_shuffle = [i for i in range(X_train.shape[0])]
    random.shuffle(index_shuffle)
    net0.fit(X_train[index_shuffle,:], y_train[index_shuffle])
            
    # prediction on valid
    y_pred = net0.predict_proba(X_valid)
    preds_train = pd.DataFrame(y_pred, columns=['Class_'+str(i+1) for i in range(num_classes)])
    preds_train['id'] = ids_train[valid_index]
    preds_train['set'] = 1

    y_pred = net0.predict_proba(X_test)
    preds_test = pd.DataFrame(y_pred, columns=['Class_'+str(i+1) for i in range(num_classes)])
    preds_test['id'] = ids_test
    preds_test['set'] = 0
    
    preds = preds_train.append(preds_test, ignore_index=True)
    return preds
    

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:e:c:f:", ["train=", "test=", "pred=", "epoch=", "cv=", "folds="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
pred_file = opts['--pred']
epoch = int(opts['--epoch'])
cv = int(opts['--cv'])
nfolds = int(opts['--folds'])

if cv == 0: 
    nfolds = 2

X, y, ids_train, encoder, scaler, tfidf = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file, scaler, tfidf)
num_classes = len(encoder.classes_)
num_features = X.shape[1]
skf = StratifiedKFold(y, nfolds, random_state=2015)
ids_train_folds = np.empty(0)
for train_index, valid_index in skf:
    ids_train_folds = np.append(ids_train_folds, ids_train[valid_index])
#pool = Pool()

for e in range(epoch):
    print "processing iteration", e

    if cv == 0:
        net0 = NeuralNetConstructor()
        index_shuffle = [i for i in range(X.shape[0])]
        random.shuffle(index_shuffle)
        net0.fit(X[index_shuffle,:], y[index_shuffle])
        preds = pd.DataFrame(net0.predict_proba(X_test), columns=['Class_'+str(i+1) for i in range(num_classes)])
        preds['id'] = ids_test
        preds.to_csv('../data/output-py/test_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
    else:
        count = 0
        preds_epoch = pd.DataFrame()
        actual = np.empty(0)
        list_result = Parallel(n_jobs=6)(delayed(compute_fold)(train_index, valid_index, X, y, X_test, ids_train, ids_test) for train_index, valid_index in skf)
        preds = pd.concat(list_result, axis = 0)
        preds.to_csv('../data/output-py/train_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
        #list_result = [pool.apply_async(compute_fold, args = (train_index, valid_index, X, y, ids_train)) for train_index, valid_index in skf]
