import pickle
import numpy    as np
import pandas   as pd
import seaborn  as sb ; sb.set(style="white")
import matplotlib
import matplotlib.pyplot as plt
import ROOT

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

            for kk in fscore.keys(): fscore[kk] += 1. * partial[kk] * totsplit / parsplit if kk in partial.keys() else 0
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

def plot_efficiency_vs_taumass(sample, cut, bins = np.linspace(1.6, 2.0, 41), save_file = True, filename = 'efficiency_vs_mass.pdf'):
    eff = ROOT.TEfficiency("eff", "", len(bins) - 1, bins) 

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("3mu mass [GeV]")
    ax1.set_ylabel("MC entries (a.u.)")
    ax1.yaxis.label.set_color('b')
    ax2 = ax1.twinx()
    ax2.set_ylabel("BDT efficiency")
    ax2.yaxis.label.set_color('r')

    for ii in range(len(bins) - 1): 
        total   = sample[(sample.cand_refit_tau_mass > bins[ii]) & (sample.cand_refit_tau_mass <= bins[ii+1])].shape[0]
        passing = sample[(sample.cand_refit_tau_mass > bins[ii]) & (sample.cand_refit_tau_mass <= bins[ii+1]) & (sample.bdt > cut)].shape[0]

        eff.SetTotalEvents(ii+1, total)
        eff.SetPassedEvents(ii+1, passing)

    sb.distplot(sample.cand_refit_tau_mass, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": 'b'}, label='MC sample', ax = ax1)
    y_val = [eff.GetEfficiency(ii)          for ii in range(len(bins) - 1)]
    x_val = (bins[1:] + bins[:-1]) / 2.
    y_eup = [eff.GetEfficiencyErrorUp(ii)   for ii in range(len(bins) - 1)]
    y_elo = [eff.GetEfficiencyErrorLow(ii)  for ii in range(len(bins) - 1)]

    ax2.set_ylim(0, 1.5 * max(y_val))

    ax2.errorbar(x_val, y_val, yerr = [y_elo, y_eup], fmt = '.', c = 'r', label = 'BDT > %s' %cut)
    plt.title("BDT efficiency VS. three-muon mass")
    fig.tight_layout()

    if save_file:
        plt.legend(loc='upper right')
        fig.savefig(filename)
        plt.clf()
