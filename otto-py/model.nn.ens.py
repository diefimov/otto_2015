import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh, linear
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
from nolearn.lasagne import NeuralNet
from joblib import Parallel, delayed
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
    #X[X<0.0000001] = 0.0000001
    #X[X>0.9999999] = 0.9999999
    #X = -np.log(np.divide(1-X, X))
    #X = -np.log(X)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, ids, encoder, scaler
    
def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    #X[X<0.0000001] = 0.0000001
    #X[X>0.9999999] = 0.9999999
    #X = -np.log(np.divide(1-X, X))
    #X = -np.log(X)
    X = scaler.transform(X)
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
               #('dense2', DenseLayer),
               #('dropout3', DropoutLayer),           
               ('output', DenseLayer)]


    net0 = NeuralNet(layers=layers0,
                 
                     input_shape=(None, num_features),
                     #dense100_num_units=50,
                     #dense100_nonlinearity=linear,
                     dense0_num_units=700,
                     dropout1_p=0.7,
                     dense1_num_units=100,
                     dropout2_p=0.1,
                     #dense2_num_units=50,
                     #dropout3_p=0.01,   
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=nesterov_momentum,
                     update_learning_rate=0.007,
                     update_momentum=0.9,
                     #on_epoch_finished=[
                     #    AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                     #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     #    ],
                 
                     eval_size=0.01,
                     verbose=0,
                     max_epochs=50)
    return net0

def train_neural_net(X, y, X_test, e):
    net0 = NeuralNetConstructor()
    np.random.seed(seed=2015 + 100*e)
    index_shuffle = [i for i in range(X.shape[0])]
    random.shuffle(index_shuffle)
    net0.fit(X[index_shuffle,:], y[index_shuffle])
    preds_epoch = pd.DataFrame(net0.predict_proba(X_test), columns=['Class_'+str(i+1) for i in range(num_classes)])
    return preds_epoch


opts, args = getopt.getopt(sys.argv[1:], "t:v:p:e:", ["train=", "test=", "pred=", "epoch="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
epoch = int(opts['--epoch'])
pred_file = opts['--pred']


X, y, ids_train, encoder, scaler = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file, scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

list_result = Parallel(n_jobs=6)(delayed(train_neural_net)(X, y, X_test, e) for e in range(epoch))
preds = sum(list_result)

# check the result
preds = preds.divide(epoch)
preds['id'] = ids_test.astype(float).astype(int)
preds.to_csv('../data/output-py/ens_1level/' + os.path.splitext(pred_file)[0] + '.csv', index=False)
