import pickle
import numpy    as np
import pandas   as pd
import seaborn  as sb ; sb.set(style="white")
import matplotlib
import matplotlib.pyplot as plt

from xgboost            import plot_importance
from collections        import OrderedDict
from itertools          import product
from root_numpy         import root2array
from sklearn.metrics    import roc_curve
from scipy.stats        import ks_2samp
from sklearn.metrics    import confusion_matrix

def plot_overtraining(train, test, score = 'score', target = 1, title = '', filename = 'overtraining.pdf'):
        ## true positive
        hist, bins = np.histogram(test[test['target'] == target][score], range = (0, 1), bins = 50, density = True)
        width  = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        scale  = len(test) / sum(hist)
        err    = np.sqrt(hist*scale) / scale
        plt.errorbar(center, hist, yerr = err, fmt = 'o', c = 'b', label = 'S (test)')
        sb.distplot(train[train['target'] == target][score], bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": 'b'}, label='S (train)')

        ## false positive
        hist, bins = np.histogram(test[test['target'] != target][score], range = (0, 1), bins = 50, density = True)
        width  = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2 
        scale  = len(test) / sum(hist)
        err    = np.sqrt(hist*scale) / scale
        plt.errorbar(center, hist, yerr = err, fmt = 'o', c = 'r', label = 'B (test)')
        sb.distplot(train[train['target'] != target][score], bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": 'r'}, label='B (train)')

        ks_sig = ks_2samp(train[train['target'] == target][score], test[test['target'] == target][score])
        ks_bkg = ks_2samp(train[train['target'] != target][score], test[test['target'] != target][score])

        plt.title('KS p-value: sig = %.3f%s - bkg = %.2f%s' %(ks_sig.pvalue * 100., '%', ks_bkg.pvalue * 100., '%'))

        plt.legend(loc = 'right')
        plt.suptitle(title)
        plt.xlim([0.0, 1.0])
        plt.yscale('log')
        plt.savefig(filename)
        plt.clf()

def plot_ROC(y, score, color = 'm', title = '', filename = 'roc.pdf', label = '', xlab = '', ylab = '', save_file = False, lower_edge = 1.e-5, alpha = 1):
        fpr, tpr, wps = roc_curve(y, score) 
        
        plt.xscale('log')
        plt.plot(fpr, tpr, color=color, label=label, alpha = alpha)

        xy = [i*j for i,j in product([10.**i for i in range(-2, 0)], [1,2,4,8])]+[1]
        plt.plot(xy, xy, color='grey', linestyle='--')

        plt.suptitle(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xlim([lower_edge, 1.0])
        plt.ylim([lower_edge, 1.0])
        plt.grid(True)
        plt.legend(loc='lower right')
        
        if save_file: 
            plt.savefig(filename)
            plt.clf()

def plot_features(classifiers, labels, filename = 'f_score.pdf'):
    if isinstance(classifiers, list):
        fscore   = {kk: 0 for kk in classifiers[0].get_booster().get_fscore().keys()}
        totsplit = sum([sum(clf.get_booster().get_fscore().values()) for clf in classifiers])

        for clf in classifiers:
            partial   = clf.get_booster().get_fscore()
            parsplit  = sum(partial.values())

            for kk in fscore.keys(): fscore[kk] += partial[kk] * totsplit / parsplit if hasattr(partial, kk) else 0
    else:
        fscore = classifiers.get_booster().get_fscore()

    fscore = OrderedDict(sorted(fscore.iteritems(), key=lambda x : x[1], reverse=False))
    
    bars  = [labels[kk] for kk in fscore.keys()]
    y_pos = np.arange(len(bars))

    plt.barh(y_pos, fscore.values())
    plt.yticks(y_pos, bars)

    plt.xlabel('F-score')
    plt.ylabel('feature')

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_correlation_matrix(sample, features, labels, label = '', filename = 'correlation.pdf'):
    labels = [labels[ll] for ll in features]

    corr = sample[features].corr()
    f, ax = plt.subplots(figsize=(12, 10))
    cmap = sb.diverging_palette(220, 10, as_cmap=True)

    g = sb.heatmap(corr,    cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f', annot_kws={'size':11},
                            square=True, linewidths=.8, cbar_kws={"shrink": .8})

    g.set_xticklabels(labels, rotation='vertical')
    g.set_yticklabels(labels, rotation='horizontal')

    plt.title('linear correlation matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
