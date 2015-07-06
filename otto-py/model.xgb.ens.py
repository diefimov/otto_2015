import pandas as pd
import numpy as np
import sys
import getopt
from sklearn import feature_extraction
from sklearn.preprocessing import LabelEncoder
sys.path.append('/Users/ef/xgboost/wrapper')
import xgboost as xgb
import random
from sklearn.metrics import log_loss
import os

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.seed(seed=2015)
    np.random.shuffle(X)
    X, labels, ids = X[:, 1:-1].astype(np.float32), X[:, -1], X[:, 0].astype(str)
    #X[X<0.0000001] = 0.0000001
    #X[X>0.9999999] = 0.9999999
    #X = -np.log(np.divide(1-X, X))
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    return X, y, ids, encoder
    
def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    #X[X<0.0000001] = 0.0000001
    #X[X>0.9999999] = 0.9999999
    #X = -np.log(np.divide(1-X, X))
    return X, ids

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:", ["train=", "test=", "pred="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
pred_file = opts['--pred']
target_col = 'target'

X, y, ids_train, encoder = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.125
param['max_depth'] = 4
param['eval_metric'] = 'mlogloss'
param['silent'] = 1
param['nthread'] = 6
param['num_class'] = 9
param['subsample'] = 0.7
param['colsample_bylevel'] = 0.12
param['colsample_bytree'] = 1.0
num_round = 110
pars = [1, 5, 10, 15, 20, 25, 30, 35, 40]

for par in pars:
    print "epoch for min_child_weight", par
    param['min_child_weight'] = par
    param['seed'] = random.randint(10, 1000000) + par
    plst = list(param.items())

    index_shuffle = [i for i in range(X.shape[0])]
    random.shuffle(index_shuffle)
    xgmat_train = xgb.DMatrix( X[index_shuffle,:], label=y[index_shuffle], missing = -999.0)
    bst = xgb.train( plst, xgmat_train, num_round );
    xgmat_test = xgb.DMatrix( X_test, missing = -999.0 )
    preds_epoch = pd.DataFrame(bst.predict( xgmat_test ), columns=['Class_'+str(i+1) for i in range(num_classes)])
    if par == 1:
        preds = preds_epoch.copy()
    else:
        preds = preds.add(preds_epoch, fill_value=0)
        
preds = preds.divide(len(pars))
preds['id'] = ids_test.astype(float).astype(int)
preds.to_csv('../data/output-py/ens_1level/' + os.path.splitext(pred_file)[0] + '.csv', index=False)
