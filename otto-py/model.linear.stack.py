import pandas as pd
import numpy as np
import sys
import getopt
import os
import random
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import statsmodels.api as sm
sys.path.append('/Users/ef/xgboost/wrapper')
import xgboost as xgb

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.seed(seed=2015)
    np.random.shuffle(X)
    X, labels, ids = X[:, 1:-1].astype(np.float32), X[:, -1], X[:, 0].astype(str)
    means = np.array([[1.0] if np.mean(row)==0.0 else [np.mean(row)] for row in X])
    X_feats = np.divide(X, means)
    #X_feats2 = np.floor(np.log10(X + 1.))
    X = np.append(X, X_feats, axis = 1)
    #X = np.append(X, X_feats2, axis = 1)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(labels)
    encoder = LabelEncoder()
    y_coded = encoder.fit_transform(labels).astype(np.int32)
    return X, y, y_coded, ids, scaler
    
def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    means = np.array([[1.0] if np.mean(row)==0.0 else [np.mean(row)] for row in X])
    X_feats = np.divide(X, means)
    #X_feats2 = np.floor(np.log10(X + 1.))
    X = np.append(X, X_feats, axis = 1)
    #X = np.append(X, X_feats2, axis = 1)
    #X = np.log10(X + 1.)
    #X = scaler.fit_transform(X)
    return X, ids

opts, args = getopt.getopt(sys.argv[1:], "t:v:p:e:c:f:s:", ["train=", "test=", "pred=", "epoch=", "cv=", "folds="])
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

X, y, y_coded, ids_train, scaler = load_train_data(train_file)
X_test, ids_test = load_test_data(test_file, scaler)
num_classes = len(y[0])
num_features = X.shape[1]
skf = StratifiedKFold(y_coded, nfolds, random_state=2015)
ids_train_folds = np.empty(0)
for train_index, valid_index in skf:
    ids_train_folds = np.append(ids_train_folds, ids_train[valid_index])

#train = train.reindex(np.random.permutation(train.index))

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['booster'] = 'gblinear'
param['max_depth'] = 12
param['eval_metric'] = 'logloss'
param['silent'] = 1
param['nthread'] = 6
param['min_child_weight'] = 1
param['colsample_bylevel'] = 0.4
param['scale_pos_weight'] = 1
param['seed'] = 10
num_round =200

for e in range(epoch):
    print "processing iteration", e
    #seed = random.randint(10, 1000000) + e
    seed = 3168 + e
    param['seed'] = seed
    plst = list(param.items())

    if cv == 0:
        index_shuffle = [i for i in range(X.shape[0])]
        random.shuffle(index_shuffle)
        preds_epoch = pd.DataFrame()
        for i in range(num_classes):
            print "processing class", i
            yi = [t[i] for t in y[index_shuffle]]
            
            xgmat_train = xgb.DMatrix( X[index_shuffle,:], label=yi, missing = -999.0)
            bst = xgb.train( plst, xgmat_train, num_round );
            xgmat_test = xgb.DMatrix( X_test, missing = -999.0 )
            preds_epoch['Class_'+str(i+1)+'_LinModel1'] = bst.predict( xgmat_test )
            
            glm = sm.GLM(yi,X[index_shuffle,:],sm.families.Binomial())
            results = glm.fit()
            preds_epoch['Class_'+str(i+1)+'_LinModel2'] = results.predict(X_test)
            
            clf=LogisticRegression(C=1, penalty='l1',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel3'] = clf.predict_proba(X_test)[:,1]
            
            clf=LogisticRegression(C=1e-1, penalty='l1',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel4'] = clf.predict_proba(X_test)[:,1]

            clf=LogisticRegression(C=1e-2, penalty='l1',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel5'] = clf.predict_proba(X_test)[:,1]

            clf=LogisticRegression(C=10, penalty='l1',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel6'] = clf.predict_proba(X_test)[:,1]

            clf=LogisticRegression(C=1, penalty='l2',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel7'] = clf.predict_proba(X_test)[:,1]

            clf=LogisticRegression(C=1e-1, penalty='l2',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel8'] = clf.predict_proba(X_test)[:,1]
            
            clf=LogisticRegression(C=1e-2, penalty='l2',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel9'] = clf.predict_proba(X_test)[:,1]

            clf=LogisticRegression(C=1e-3, penalty='l2',random_state=seed)
            clf.fit(X[index_shuffle,:],yi)
            preds_epoch['Class_'+str(i+1)+'_LinModel10'] = clf.predict_proba(X_test)[:,1]

    else:
        count = 0
        for train_index, valid_index in skf:
            print "processing fold", count+1
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            if count == 0:
                actual = y_valid
            else:
                actual = np.append(actual, y_valid, axis=0)
            index_shuffle = [i for i in range(X_train.shape[0])]
            random.shuffle(index_shuffle)
            
            preds_fold = pd.DataFrame()
            for i in range(num_classes):
                print "processing class", i
                yi_train = [t[i] for t in y_train[index_shuffle]]
                
                xgmat_train = xgb.DMatrix( X_train[index_shuffle,:], label=yi_train, missing = -999.0)
                bst = xgb.train( plst, xgmat_train, num_round );
                xgmat_valid = xgb.DMatrix( X_valid, missing = -999.0 )
                preds_fold['Class_'+str(i+1)+'_LinModel1'] = bst.predict( xgmat_valid )
                
                glm = sm.GLM(yi_train,X_train[index_shuffle,:],sm.families.Binomial())
                results = glm.fit()
                preds_fold['Class_'+str(i+1)+'_LinModel2'] = results.predict(X_valid)

                clf=LogisticRegression(C=1, penalty='l1',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel3'] = clf.predict_proba(X_valid)[:,1]
            
                clf=LogisticRegression(C=1e-1, penalty='l1',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel4'] = clf.predict_proba(X_valid)[:,1]

                clf=LogisticRegression(C=1e-2, penalty='l1',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel5'] = clf.predict_proba(X_valid)[:,1]

                clf=LogisticRegression(C=10, penalty='l1',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel6'] = clf.predict_proba(X_valid)[:,1]

                clf=LogisticRegression(C=1, penalty='l2',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel7'] = clf.predict_proba(X_valid)[:,1]

                clf=LogisticRegression(C=1e-1, penalty='l2',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel8'] = clf.predict_proba(X_valid)[:,1]
            
                clf=LogisticRegression(C=1e-2, penalty='l2',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel9'] = clf.predict_proba(X_valid)[:,1]

                clf=LogisticRegression(C=1e-3, penalty='l2',random_state=seed)
                clf.fit(X_train[index_shuffle,:],yi_train)
                preds_fold['Class_'+str(i+1)+'_LinModel10'] = clf.predict_proba(X_valid)[:,1]
              
            if count == 0:
                preds_epoch = preds_fold.copy()
            else:
                preds_epoch = preds_epoch.append(preds_fold, ignore_index=True)

            count += 1
            for i in range(10):
                cols = ['Class_'+str(j+1)+'_LinModel'+str(i+1) for j in range(num_classes)]
                preds_model = preds_epoch[cols].as_matrix()
                print "logloss for model", i+1, log_loss(actual, preds_model)
    if cv == 0:
        preds_epoch['id'] = ids_test.astype(float).astype(int)
        preds_epoch.to_csv('../data/output-py/test_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
        preds_epoch = preds_epoch.drop('id', axis=1)
    else:
        preds_epoch['id'] = ids_train_folds.astype(float).astype(int)
        preds_epoch.to_csv('../data/output-py/train_raw/' + os.path.splitext(pred_file)[0] + '.epoch' + str(e) + '.csv', index=False)
        preds_epoch = preds_epoch.drop('id', axis=1)
    
    if e == 0:
        preds = preds_epoch.copy()
    else:
        preds = preds.add(preds_epoch, fill_value=0)
    if cv == 1:
        preds_epoch = preds.copy()
        preds_epoch = preds_epoch.divide(e+1)
        
        for i in range(10):
            cols = ['Class_'+str(j+1)+'_LinModel'+str(i+1) for j in range(num_classes)]
            preds_model = preds_epoch[cols].as_matrix()
            print "final logloss for model", i+1, log_loss(actual, preds_model)

        
# create submission file
#preds = preds.divide(epoch)
#if cv == 0:
#    preds['id'] = ids_test.astype(float).astype(int)
#    preds.to_csv(os.path.splitext(pred_file)[0] + '.csv', index=False)
#else:
#    preds['id'] = ids_train_folds.astype(float).astype(int)
#    preds.to_csv(os.path.splitext(pred_file)[0] + '.csv', index=False)

