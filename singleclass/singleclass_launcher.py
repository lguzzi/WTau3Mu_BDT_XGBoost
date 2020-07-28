#!/bin/env python2
import sys
import os
import argparse

import pickle
import root_pandas
import numpy            as np
import pandas           as pd
import matplotlib.cm    as cm

import json

from sklearn            import preprocessing
from sklearn.externals  import joblib

sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/datasets')
from W_dataset  import train, test, valid
from features   import features, labels

sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/libs'    )
from trainer    import start_XGBoost_trainer, add_bdt_score
from plotter    import plot_overtraining, plot_ROC, plot_correlation_matrix, plot_features, plot_efficiency_vs_taumass


parser = argparse.ArgumentParser('XGBoost training script')
parser.add_argument('--load'     , default = None        , help = 'load an existing training from the specified file')
parser.add_argument('--label'    , default = 'test'      , help = 'set the label for the training'                   )
parser.add_argument('--save_tree', action  = 'store_true', help = 'save enriched ntuples after training or loading'  )
args = parser.parse_args()

#tag = os.popen('date +%s').readlines()[0].strip('\n')
tag = args.label

## https://xgboost.readthedocs.io/en/latest/parameter.html
hyperpars = {   
                "max_depth"               : 3     , ## crucial to prevent low-score overfitting
                "learning_rate"           : 0.05  , ## (aka eta) step size shrinkage used in update to prevents overfitting. Eta shrinks the feature weights to make the boosting process more conservative.
                "n_estimators"            : 10000 ,
                "subsample"               : 0.7   ,
                "colsample_bytree"        : 0.7   ,
                "min_child_weight"        : 10    , ## sensitive to low-score overtraining (should be as low as 10)
                "gamma"                   : 1     , ## Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be
                "reg_alpha"               : 1     , ## (L1 reg) Increasing this value will make model more conservative
                "reg_lambda"              : 1     , ## (L2 reg) Increasing this value will make model more conservative
                "kfold"                   : 5     ,
                "early_stopping_rounds"   : 10    ,
}

## train a new classifier of load an existing one from a .pck file
if args.load is None:
    classifiers = start_XGBoost_trainer(
        train       = train     ,
        valid       = valid     ,
        tag         = tag       ,
        features    = features  ,   
        **hyperpars
    )

    if not os.path.exists('./pck'): os.mkdir('./pck')

    classifier_file = open('./pck/classifier_%s.pck' %tag, 'w+')
    pickle.dump(classifiers, classifier_file)
    classifier_file.close()
else: 
    classifiers = joblib.load(args.load)

## add the BDT score to the training and test dataset
for jj, clf in enumerate(classifiers):
    add_bdt_score(classifier = clf, sample = train, features = features, scale = len(classifiers), index = jj)
    add_bdt_score(classifier = clf, sample = test , features = features, scale = len(classifiers), index = jj)

## plot ROCs, correlation matrices and overtraining tests
if not os.path.exists('./pdf'):         os.mkdir('./pdf')
if not os.path.exists('./pdf/%s' %tag): os.mkdir('./pdf/%s' %tag)

plot_ROC(y = test ['target'], score = test ['bdt'], title = 'ROC curve (test)' ,                                     color = 'r', label = 'test ROC' , xlab = 'FPR', ylab = 'TPR')
plot_ROC(y = train['target'], score = train['bdt'], title = 'ROC curve (train)', filename = './pdf/%s/roc.pdf' %tag, color = 'b', label = 'train ROC', xlab = 'FPR', ylab = 'TPR', save_file = True)

plot_overtraining(  train  = train, 
                    test   = test , 
                    target = 1, score = 'bdt' ,
                    title  = 'bkg proba distribution' , filename = './pdf/%s/overtraining_bkg_proba.pdf' %tag)

#plot_correlation_matrix( sample = pd.concat([train[train.target == 0], test[test.target == 0]]), features = features + ['bdt', 'cand_refit_tau_mass'], labels = labels, label = '', filename = './pdf/%s/corr_mat_bkg.pdf' %tag)
plot_correlation_matrix( sample = pd.concat([train, test]), features = features + ['bdt', 'cand_refit_tau_mass'], labels = labels, label = '', filename = './pdf/%s/corr_mat_bkg.pdf' %tag)
plot_features(classifiers = classifiers, labels = labels, filename = './pdf/%s/f_score.pdf' %tag)

plot_efficiency_vs_taumass(sample = test[test.target == 1], cut = 0.996, filename = './pdf/%s/efficiency_vs_mass.pdf' %tag) 

## save the enriched ntuples
if args.save_tree:
    #sigW = pd.concat([train[(train.target == 1) & (train.shifted == 0)], test[(test.target == 1) & (test.shifted == 0)]])
    #bkg  = pd.concat([train[(train.target == 0) & (train.shifted == 0)], test[(test.target == 0) & (test.shifted == 0)]])

    sigW = pd.concat([train[(train.target == 1)], test[(test.target == 1)]])
    bkg  = pd.concat([train[(train.target == 0)], test[(test.target == 0)]])

    if not os.path.exists('./ntuples'): os.mkdir('./ntuples')

    #os.remove('./ntuples/signal_{TAG}.root'    .format(TAG = tag))
    #os.remove('./ntuples/background_{TAG}.root'.format(TAG = tag))

    sigW.to_root('./ntuples/signal_{TAG}.root'    .format(TAG = tag), key = 'tree')
    bkg .to_root('./ntuples/background_{TAG}.root'.format(TAG = tag), key = 'tree')

## save the BDT config
json.dump(hyperpars, open('pdf/%s/hyperpars.json' %tag, 'w'), indent = True) 

print 'all done'