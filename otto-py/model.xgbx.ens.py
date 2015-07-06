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
    X, y, ids = X[:, 1:-1].astype(np.float32), X[:, -1], X[:, 0]
    return X, y, np.unique(ids).astype(str)
    
def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0]
    return X, np.unique(ids).astype(str)

def extract_predictions(X_test, y_pred):
    num_col = X_test.shape[1]
    for i in range(num_classes):
        indicator = X_test[:,num_col-num_classes+i]
        yi_pred = np.array([[y_pred[jx]] for jx, j in enumerate(indicator) if j == 1.])
        if i == 0:
            y_pred_matrix = yi_pred
        else:
            y_pred_matrix = np.append(y_pred_matrix, yi_pred, 1)
    return y_pred_matrix

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:", ["train=", "test=", "pred="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
pred_file = opts['--pred']
target_col = 'target'

X, y, ids_train = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file)
num_classes = 9
num_features = X.shape[1]

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.03
param['max_depth'] = 6
param['eval_metric'] = 'logloss'
param['silent'] = 1
param['nthread'] = 6
#param['gamma'] = 0.9
param['subsample'] = 0.9
param['colsample_bytree'] = 0.7
num_round = 400
#pars = [5, 10, 15, 20, 25, 30, 35]
pars = [5, 10, 15]

epoch = len(pars)
for e in range(epoch):
    print "iteration", e
    param['min_child_weight'] = pars[e]
    #param['seed'] = random.randint(10, 1000000) + pars[e]
    param['seed'] = 33090 + 450*e
    plst = list(param.items())

    index_shuffle = [i for i in range(X.shape[0])]
    random.shuffle(index_shuffle)
    xgmat_train = xgb.DMatrix( X[index_shuffle,:], label=y[index_shuffle], missing = -999.0)
    bst = xgb.train( plst, xgmat_train, num_round );
    xgmat_test = xgb.DMatrix( X_test, missing = -999.0 )
    preds_epoch = pd.DataFrame(extract_predictions(X_test, bst.predict( xgmat_test )), columns=['Class_'+str(i+1) for i in range(num_classes)])
    if e == 0:
        preds = preds_epoch.copy()
    else:
        preds = preds.add(preds_epoch, fill_value=0)
        
preds = preds.divide(epoch)
preds['id'] = ids_test.astype(float).astype(int)
preds.to_csv('../data/output-py/ens_1level/' + os.path.splitext(pred_file)[0] + '.csv', index=False)
