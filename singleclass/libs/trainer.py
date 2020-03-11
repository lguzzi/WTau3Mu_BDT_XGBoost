import numpy as np, os, xgboost
from xgboost                    import XGBClassifier
from sklearn.model_selection    import StratifiedKFold
from sklearn.multiclass         import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection    import train_test_split

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

def add_bdt_score(classifier, sample, features, index = 0, scale = 1., label = 'bdt'):
    if not hasattr(sample, label):
        sample.insert(len(sample.columns), label, np.zeros(sample.id.shape))
        
    proba = classifier.predict_proba(sample[features])[:, 1]
    sample.loc[:, label] += proba / scale
    sample.loc[:, label + str(index)] = proba

def start_XGBoost_trainer(  train,
                            test ,
                            tag  ,
                            features,
                            ## HYPERPARAMETERS
                                max_depth               = 3     ,
                                learning_rate           = 0.05  ,
                                n_estimators            = 10000 ,
                                subsample               = 0.7   ,
                                colsample_bytree        = 0.7   ,
                                min_child_weight        = 10    ,
                                gamma                   = 5     ,
                                reg_alpha               = 0.0   ,
                                reg_lambda              = 5     ,
                                kfold                   = 5     ,
                                early_stopping_rounds   = 10    ,
): 
    kf = StratifiedKFold(n_splits = kfold, random_state = 1986, shuffle = True )
    fold_indices = kf.split(train[features].values, train['target'], groups = train['target'])

    #X_train = train[features]
    #y_train = train['target']
    #X_train, X_valid, y_train, y_valid = train_test_split(train[features], train['target'], test_size=0.2, random_state=1986)

    classifiers = []
    for i, (train_index, test_index) in enumerate(fold_indices):
        print 'INFO: evaluating', ordinal(i+1), 'fold'
        X_train, X_valid = train[train.id.isin(train_index)][features], train[train.id.isin(test_index)][features]
        y_train, y_valid = train[train.id.isin(train_index)]['target'], train[train.id.isin(test_index)]['target']

        clf = XGBClassifier(    
            max_depth       = max_depth         ,
            learning_rate   = learning_rate     ,
            n_estimators    = n_estimators      ,
            silent          = True              , 
            subsample       = subsample         ,
            colsample_bytree= colsample_bytree  ,
            #min_child_weight= min_child_weight * np.sum(train[train.id.isin(train_index)].weight),
            min_child_weight= min_child_weight  ,
            gamma           = gamma             ,
            seed            = 1986              ,
            reg_alpha       = reg_alpha         ,
            reg_lambda      = reg_lambda        ,
            n_jobs          = 60                ,
            #nthread         = 60                ,
            #tree_method     = 'gpu_exact'       ,
        )

        clf.fit(    X_train, y_train, 
            eval_set                = [(X_train, y_train), (X_valid, y_valid)]  ,
            early_stopping_rounds   = early_stopping_rounds                     ,
            eval_metric             = 'auc'                                     ,
            sample_weight           = train['weight']                           ,
            verbose                 = False                                     ,
        )

        classifiers.append(clf)
 
    return classifiers

