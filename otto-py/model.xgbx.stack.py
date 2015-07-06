import pandas as pd
import numpy as np
import sys
import getopt
from sklearn import feature_extraction
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
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
    encoder = LabelEncoder()
    y_coded = encoder.fit_transform(labels).astype(np.int32)
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(labels)
    return X, y, y_coded, ids, encoder
    
def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    return X, ids

def extend_datasets(X, y, X_test):
    y_ext = []
    for i in range(num_classes):
        yi = [t[i] for t in y]
        y_ext = y_ext + yi
        X_class = X
        X_test_class = X_test
        for j in range(num_classes):
            if j == i:
                X_class = np.append(X_class, np.ones((X_class.shape[0],1)), 1)
                X_test_class = np.append(X_test_class, np.ones((X_test_class.shape[0],1)), 1)
            else:
                X_class = np.append(X_class, np.zeros((X_class.shape[0],1)), 1)
                X_test_class = np.append(X_test_class, np.zeros((X_test_class.shape[0],1)), 1)
        if i == 0:
            X_ext = X_class
            X_test_ext = X_test_class
        else:
            X_ext = np.append(X_ext, X_class, 0)
            X_test_ext = np.append(X_test_ext, X_test_class, 0)
    return X_ext, y_ext, X_test_ext

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


def compute_fold(train_index, valid_index, X, y, X_test, ids_train, ids_test):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    X_train_ext, y_train_ext, X_valid_ext = extend_datasets(X_train, y_train, X_valid)
    
    index_shuffle = [i for i in range(X_train_ext.shape[0])]
    random.shuffle(index_shuffle)
    y_train_ext = [y_train_ext[t] for t in index_shuffle]
    xgmat_train = xgb.DMatrix( X_train_ext[index_shuffle,:], label=y_train_ext, missing = -999.0)
    bst = xgb.train( plst, xgmat_train, num_round );

    #prediction on valid
    xgmat_valid = xgb.DMatrix( X_valid_ext, missing = -999.0 )
    y_pred = extract_predictions(X_valid_ext, bst.predict( xgmat_valid ))
    preds_train = pd.DataFrame(y_pred, columns=['Class_'+str(i+1) for i in range(num_classes)])
    preds_train['id'] = ids_train[valid_index]
            
    return preds_train
        

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:e:c:f:", ["train=", "test=", "pred=", "epoch=", "cv=", "folds="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
pred_file = opts['--pred']
epoch = int(opts['--epoch'])
cv = int(opts['--cv'])
nfolds = int(opts['--folds'])
target_col = 'target'

if cv == 0: 
    nfolds = 2

X, y, y_coded, ids_train, encoder = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file)
num_classes = len(encoder.classes_)
num_features = X.shape[1]
skf = StratifiedKFold(y_coded, nfolds, random_state=2015)
ids_train_folds = np.empty(0)
for train_index, valid_index in skf:
    ids_train_folds = np.append(ids_train_folds, ids_train[valid_index])

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 15
param['eval_metric'] = 'logloss'
param['silent'] = 1
param['nthread'] = 6
#param['gamma'] = 0.9
#param['min_child_weight'] = 4
param['subsample'] = 0.9
param['colsample_bytree'] = 0.7
#param['colsample_bylevel'] = 0.5
num_round = 300

for e in range(epoch):
    print "processing iteration", e
    param['seed'] = 2015 + 10*e
    plst = list(param.items())

    if cv == 0:
        X_ext, y_ext, X_test_ext = extend_datasets(X, y, X_test)
        index_shuffle = [i for i in range(X_ext.shape[0])]
        random.shuffle(index_shuffle)
        y_ext = [y_ext[t] for t in index_shuffle]
        xgmat_train = xgb.DMatrix( X_ext[index_shuffle,:], label=y_ext, missing = -999.0)
        bst = xgb.train( plst, xgmat_train, num_round );
        xgmat_test = xgb.DMatrix( X_test_ext, missing = -999.0 )
        preds = pd.DataFrame(extract_predictions(X_test_ext, bst.predict( xgmat_test )), columns=['Class_'+str(i+1) for i in range(num_classes)])
        preds['id'] = ids_test
        preds.to_csv('../data/output-py/test_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
    else:
        count = 0
        for train_index, valid_index in skf:
            print "processing fold", count+1
            preds_fold = compute_fold(train_index, valid_index, X, y, X_test, ids_train, ids_test)
            if count == 0:
                preds = preds_fold.copy()
            else:
                preds = preds.append(preds_fold)
            count += 1
        preds.to_csv('../data/output-py/train_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
