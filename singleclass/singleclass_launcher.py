#!/bin/env python2
import sys
import os
import argparse

import pickle
import root_pandas
import numpy            as np
import pandas           as pd
import matplotlib.cm    as cm

from sklearn            import preprocessing
from sklearn.externals  import joblib

sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/datasets')
from W_dataset  import train, test, features, labels

sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/libs'    )
from trainer    import start_XGBoost_trainer, add_bdt_score
from plotter    import plot_overtraining, plot_ROC, plot_correlation_matrix, plot_features


parser = argparse.ArgumentParser('XGBoost training script')
parser.add_argument('--load'     , default = None        , help = 'load an existing training from the specified file')
parser.add_argument('--save_tree', action  = 'store_true', help = 'save enriched ntuples after training or loading'  )
args = parser.parse_args()

#tag = os.popen('date +%s').readlines()[0].strip('\n')
tag = '28jan2020'

## train a new classifier of load an existing one from a .pck file
if args.load is None:
    classifiers = start_XGBoost_trainer(
        train       = train     ,
        test        = test      ,
        tag         = tag       ,
        features    = features  ,
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

plot_ROC(y = test ['target'], score = test ['bdt'], title = 'ROC curve (test)' , filename = './pdf/%s/roc_test.pdf'  %tag, color = 'r', label = 'BDT score', xlab = 'FPR', ylab = 'TPR')
plot_ROC(y = train['target'], score = train['bdt'], title = 'ROC curve (train)', filename = './pdf/%s/roc_train.pdf' %tag, color = 'b', label = 'BDT score', xlab = 'FPR', ylab = 'TPR', save_file = True)

plot_overtraining(  train  = train, 
                    test   = test , 
                    target = 1, score = 'bdt' ,
                    title  = 'bkg proba distribution' , filename = './pdf/%s/overtraining_bkg_proba.pdf' %tag)

#plot_correlation_matrix( sample = pd.concat([train[train.target == 0], test[test.target == 0]]), features = features + ['bdt', 'cand_refit_tau_mass'], labels = labels, label = '', filename = './pdf/%s/corr_mat_bkg.pdf' %tag)
plot_correlation_matrix( sample = pd.concat([train, test]), features = features + ['bdt', 'cand_refit_tau_mass'], labels = labels, label = '', filename = './pdf/%s/corr_mat_bkg.pdf' %tag)
plot_features(classifiers = classifiers, labels = labels, filename = './pdf/%s/f_score.pdf' %tag)

## save the enriched ntuples
if args.save_tree:
    sigW = pd.concat([train[(train.target == 1)], test[(test.target == 1)]])
    bkg  = pd.concat([train[(train.target == 0)], test[(test.target == 0)]])

    if not os.path.exists('./ntuples'): os.mkdir('./ntuples')

    #os.remove('./ntuples/signal_{TAG}.root'    .format(TAG = tag))
    #os.remove('./ntuples/background_{TAG}.root'.format(TAG = tag))

    sigW.to_root('./ntuples/signal_{TAG}.root'    .format(TAG = tag), key = 'tree')
    bkg .to_root('./ntuples/background_{TAG}.root'.format(TAG = tag), key = 'tree')

print 'all done'