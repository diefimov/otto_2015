import pandas as pd
import numpy as np
import sys
import getopt
from sklearn import feature_extraction
from sklearn.cross_validation import StratifiedKFold
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

    log2FloorX = np.floor(np.log2(X + 1))
    #X_col_sums = log2FloorX.sum(axis=0, keepdims=False)
    #ix = [ix for ix, i in enumerate(X_col_sums) if i>0]
    #log2FloorX = log2FloorX[:,ix]
    #X_feats = log2FloorX.copy()
    X_feats = np.append(X, log2FloorX, axis = 1)

    log3FloorX = np.floor(np.divide(np.log(X + 1),np.log(3)))    
    X_feats = np.append(X_feats, log3FloorX, axis = 1)

    log4FloorX = np.floor(np.divide(np.log(X+1),np.log(4)))
    X_feats = np.append(X_feats, log4FloorX, axis = 1)
    
    log5FloorX = np.floor(np.divide(np.log(X+1),np.log(5)))
    X_feats = np.append(X_feats, log5FloorX, axis = 1)
    
    log6FloorX = np.floor(np.divide(np.log(X+1),np.log(6)))
    X_feats = np.append(X_feats, log6FloorX, axis = 1)
    
    log7FloorX = np.floor(np.divide(np.log(X+1),np.log(7)))
    X_feats = np.append(X_feats, log7FloorX, axis = 1)
    
    log8FloorX = np.floor(np.divide(np.log(X+1),np.log(8)))
    X_feats = np.append(X_feats, log8FloorX, axis = 1)
    
    log9FloorX = np.floor(np.divide(np.log(X+1),np.log(9)))
    X_feats = np.append(X_feats, log9FloorX, axis = 1)
    
    log12FloorX = np.floor(np.divide(np.log(X+1),np.log(12)))
    X_feats = np.append(X_feats, log12FloorX, axis = 1)
    
    log13FloorX = np.floor(np.divide(np.log(X+1),np.log(13)))
    X_feats = np.append(X_feats, log13FloorX, axis = 1)
    
    logExpFloorX = np.floor(np.log(X+1))
    X_feats = np.append(X_feats, logExpFloorX, axis = 1)
    
    sqrtFloorX = np.floor(np.sqrt(X+1))
    X_feats = np.append(X_feats, sqrtFloorX, axis = 1)
    
    powX = np.power(X+1,2)
    X_feats = np.append(X_feats, powX, axis = 1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    return X_feats, y, ids, encoder
    
def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    
    log2FloorX = np.floor(np.log2(X + 1))
    X_feats = np.append(X, log2FloorX, axis = 1)

    log3FloorX = np.floor(np.divide(np.log(X + 1),np.log(3)))    
    X_feats = np.append(X_feats, log3FloorX, axis = 1)

    log4FloorX = np.floor(np.divide(np.log(X+1),np.log(4)))
    X_feats = np.append(X_feats, log4FloorX, axis = 1)
    
    log5FloorX = np.floor(np.divide(np.log(X+1),np.log(5)))
    X_feats = np.append(X_feats, log5FloorX, axis = 1)
    
    log6FloorX = np.floor(np.divide(np.log(X+1),np.log(6)))
    X_feats = np.append(X_feats, log6FloorX, axis = 1)
    
    log7FloorX = np.floor(np.divide(np.log(X+1),np.log(7)))
    X_feats = np.append(X_feats, log7FloorX, axis = 1)
    
    log8FloorX = np.floor(np.divide(np.log(X+1),np.log(8)))
    X_feats = np.append(X_feats, log8FloorX, axis = 1)
    
    log9FloorX = np.floor(np.divide(np.log(X+1),np.log(9)))
    X_feats = np.append(X_feats, log9FloorX, axis = 1)
    
    log12FloorX = np.floor(np.divide(np.log(X+1),np.log(12)))
    X_feats = np.append(X_feats, log12FloorX, axis = 1)
    
    log13FloorX = np.floor(np.divide(np.log(X+1),np.log(13)))
    X_feats = np.append(X_feats, log13FloorX, axis = 1)
    
    logExpFloorX = np.floor(np.log(X+1))
    X_feats = np.append(X_feats, logExpFloorX, axis = 1)
    
    sqrtFloorX = np.floor(np.sqrt(X+1))
    X_feats = np.append(X_feats, sqrtFloorX, axis = 1)
    
    powX = np.power(X+1,2)
    X_feats = np.append(X_feats, powX, axis = 1)
    
    return X_feats, ids

def compute_fold(train_index, valid_index, X, y, X_test, ids_train, ids_test):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    index_shuffle = [i for i in range(X_train.shape[0])]
    random.shuffle(index_shuffle)
    xgmat_train = xgb.DMatrix( X_train[index_shuffle,:], label=y_train[index_shuffle], missing = -999.0)
    bst = xgb.train( plst, xgmat_train, num_round );

    #prediction on valid
    xgmat_valid = xgb.DMatrix( X_valid, missing = -999.0 )
    y_pred = bst.predict( xgmat_valid )
    preds_train = pd.DataFrame(y_pred, columns=['Class_'+str(i+1) for i in range(num_classes)])
    preds_train['id'] = ids_train[valid_index]
    preds_train['set'] = 1
            
    #prediction on test
    xgmat_test = xgb.DMatrix( X_test, missing = -999.0 )
    y_pred = bst.predict( xgmat_test )
    preds_test = pd.DataFrame(y_pred, columns=['Class_'+str(i+1) for i in range(num_classes)])
    preds_test['id'] = ids_test
    preds_test['set'] = 0
            
    preds = preds_train.append(preds_test, ignore_index=True)
    return preds

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:c:f:", ["train=", "test=", "pred=", "cv=", "folds="])
opts = {x[0]:x[1] for x in opts}
train_file = opts['--train']
test_file = opts['--test']
pred_file = opts['--pred']
#epoch = int(opts['--epoch'])
epoch = 10
cv = int(opts['--cv'])
nfolds = int(opts['--folds'])
target_col = 'target'

if cv == 0: 
    nfolds = 2

X, y, ids_train, encoder = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file)
num_classes = len(encoder.classes_)
num_features = X.shape[1]
skf = StratifiedKFold(y, nfolds, random_state=2015)
ids_train_folds = np.empty(0)
for train_index, valid_index in skf:
    ids_train_folds = np.append(ids_train_folds, ids_train[valid_index])

param = {}
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'
param['eta'] = 0.05
param['silent'] = 1
param['num_class'] = 9
param['nthread'] = 6

for e in range(epoch):
    print "processing iteration", e
    param['seed'] = 3015 + (10*e)
    
    if e == 0:
        param['max_depth'] = 50
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.012        
        param['colsample_bytree'] = 1.0
        num_round = 500
    
    if e == 1:
        param['max_depth'] = 50
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.024
        param['colsample_bytree'] = 1.0
        num_round = 400
    
    if e == 2:
        param['max_depth'] = 30
        param['min_child_weight'] = 1
        param['colsample_bylevel'] = 0.012        
        param['colsample_bytree'] = 1.0
        num_round = 550
    
    if e == 3:
        param['max_depth'] = 50
        param['min_child_weight'] = 3
        param['colsample_bylevel'] = 0.012        
        param['colsample_bytree'] = 1.0
        num_round = 400

    if e == 4:
        param['max_depth'] = 50
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.018
        param['colsample_bytree'] = 1.0
        num_round = 400

    if e == 5:
        param['max_depth'] = 50
        param['min_child_weight'] = 8
        param['colsample_bylevel'] = 0.012
        param['colsample_bytree'] = 1.0
        num_round = 650

    if e == 6:
        param['max_depth'] = 40
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.012
        param['colsample_bytree'] = 1.0
        num_round = 500
    
    if e == 7:
        param['max_depth'] = 14
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.024
        param['colsample_bytree'] = 1.0
        num_round = 750

    if e == 8:
        param['max_depth'] = 14
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.036
        param['colsample_bytree'] = 1.0
        num_round = 650

    if e == 9:
        param['max_depth'] = 19
        param['min_child_weight'] = 5
        param['colsample_bylevel'] = 0.012
        param['colsample_bytree'] = 1.0
        num_round = 750

    plst = list(param.items())

    if cv == 0:
        index_shuffle = [i for i in range(X.shape[0])]
        random.shuffle(index_shuffle)
        xgmat_train = xgb.DMatrix( X[index_shuffle,:], label=y[index_shuffle], missing = -999.0)
        bst = xgb.train( plst, xgmat_train, num_round );
        xgmat_test = xgb.DMatrix( X_test, missing = -999.0 )
        preds = pd.DataFrame(bst.predict( xgmat_test ), columns=['Class_'+str(i+1) for i in range(num_classes)])
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
