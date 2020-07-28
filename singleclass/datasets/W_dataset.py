# import pandas
import numpy    as np
import pandas   as pd
import seaborn  as sns
import root_numpy

from sklearn.model_selection    import train_test_split
sns.set(style="white")

from features import features, branches, labels

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

category = '( (sqrt(cand_refit_tau_massE) / cand_refit_tau_mass) > 0.007 & (sqrt(cand_refit_tau_massE) / cand_refit_tau_mass) < 0.012 )'
sig_selection = ' & '.join([
    "( abs(cand_refit_tau_mass - 1.8) < 0.2 )",
    "( abs(cand_refit_charge) == 1 )",
    category,
])
bkg_selection = ' & '.join([
    "( abs(cand_refit_tau_mass - 1.8) < 0.2 )",
    "( abs(cand_refit_charge) == 1 )",
    "( abs(cand_refit_tau_mass - 1.78) > 0.06 )",
    category,
])

signal_W_2017     = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia/WToTauTo3Mu/WTau3MuTreeProducer/tree.root',   ## Pythia 2017
    #'/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_MadGraph/WToTauTo3Mu/WTau3MuTreeProducer/tree.root',     ## MadGraph 2017
]
signal_W_2018 = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia/WToTauTo3Mu/WTau3MuTreeProducer/tree.root',     ## Pythia 2018
    #'/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_MadGraph/WToTauTo3Mu/WTau3MuTreeProducer/tree.root',   ## MadGraph 2018
]

shifted_W_2017 = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia_shiftedMasses/WToTauTo3Mu_1p65/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia_shiftedMasses/WToTauTo3Mu_1p70/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia_shiftedMasses/WToTauTo3Mu_1p85/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia_shiftedMasses/WToTauTo3Mu_1p90/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2017_Pythia_shiftedMasses/WToTauTo3Mu_1p95/WTau3MuTreeProducer/tree.root',
]
shifted_W_2018 = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia_shiftedMasses/WToTauTo3Mu_1p65/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia_shiftedMasses/WToTauTo3Mu_1p70/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia_shiftedMasses/WToTauTo3Mu_1p85/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia_shiftedMasses/WToTauTo3Mu_1p90/WTau3MuTreeProducer/tree.root',
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/MC2018_Pythia_shiftedMasses/WToTauTo3Mu_1p95/WTau3MuTreeProducer/tree.root',
]

backgrounds_2017 = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2017/DoubleMuonLowMass_Run2017C_31Mar2018/WTau3MuTreeProducer/tree.root', ## 2017C
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2017/DoubleMuonLowMass_Run2017D_31Mar2018/WTau3MuTreeProducer/tree.root', ## 2017D
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2017/DoubleMuonLowMass_Run2017E_31Mar2018/WTau3MuTreeProducer/tree.root', ## 2017E
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2017/DoubleMuonLowMass_Run2017F_31Mar2018/WTau3MuTreeProducer/tree.root', ## 2017F
]
backgrounds_2018 = [
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2018_PromptReco/DoubleMuonLowMass_Run2018A_PromptReco/WTau3MuTreeProducer/tree.root',    ## 2018A
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2018_PromptReco/DoubleMuonLowMass_Run2018B_PromptReco/WTau3MuTreeProducer/tree.root',    ## 2018B
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2018_PromptReco/DoubleMuonLowMass_Run2018C_PromptReco/WTau3MuTreeProducer/tree.root',    ## 2018C
    '/gwpool/users/lguzzi/Tau3Mu/2017_2018/ntuple/data2018_PromptReco/DoubleMuonLowMass_Run2018D_PromptReco/WTau3MuTreeProducer/tree.root',    ## 2018D
]

sigW_2017 = pd.DataFrame( root_numpy.root2array(signal_W_2017   , 'tree', branches  = branches + ['weight'], selection = sig_selection))
sigW_2018 = pd.DataFrame( root_numpy.root2array(signal_W_2018   , 'tree', branches  = branches + ['weight'], selection = sig_selection))
shiW_2017 = pd.DataFrame( root_numpy.root2array(shifted_W_2017  , 'tree', branches  = branches + ['weight'], selection = sig_selection))
shiW_2018 = pd.DataFrame( root_numpy.root2array(shifted_W_2018  , 'tree', branches  = branches + ['weight'], selection = sig_selection))
bkg_2017  = pd.DataFrame( root_numpy.root2array(backgrounds_2017, 'tree', branches  = branches + ['weight'], selection = bkg_selection))
bkg_2018  = pd.DataFrame( root_numpy.root2array(backgrounds_2018, 'tree', branches  = branches + ['weight'], selection = bkg_selection))

features.append('year')
sigW_2017['year'] = np.full(sigW_2017.shape[0], 2017)
sigW_2018['year'] = np.full(sigW_2018.shape[0], 2018)
shiW_2017['year'] = np.full(shiW_2017.shape[0], 2017)
shiW_2018['year'] = np.full(shiW_2018.shape[0], 2018)
bkg_2017 ['year'] = np.full(bkg_2017.shape [0], 2017)
bkg_2018 ['year'] = np.full(bkg_2018.shape [0], 2018)

branches.append('shifted')
sigW_2017['shifted'] = np.full(sigW_2017.shape[0], 0)
sigW_2018['shifted'] = np.full(sigW_2018.shape[0], 0)
shiW_2017['shifted'] = np.full(shiW_2017.shape[0], 1)
shiW_2018['shifted'] = np.full(shiW_2018.shape[0], 1)
bkg_2017 ['shifted'] = np.full(bkg_2017.shape [0], 0)
bkg_2018 ['shifted'] = np.full(bkg_2018.shape [0], 0)

sigW = pd.concat([sigW_2017, sigW_2018, shiW_2017, shiW_2018]   , ignore_index = True)
#sigW = pd.concat([sigW_2017, sigW_2018]   , ignore_index = True)
bkg  = pd.concat([bkg_2017 , bkg_2018 ]                         , ignore_index = True)

NEV = -1
if NEV > 0:    
    print '!'* 20, 'WARNING: using only', NEV, 'events per class', '!'*20, '\n'

    ## shuffle
    sigW = sigW.sample(frac=1).reset_index(drop=True)
    bkg  = bkg .sample(frac=1).reset_index(drop=True)
    
    sigW = sigW[:NEV]
    bkg  = bkg [:NEV]

## correctly set the target shape as (#events, #classes)
sigW['target'] =  np.ones (sigW.shape[0]).astype(np.int)
bkg ['target'] =  np.zeros(bkg. shape[0]).astype(np.int)

sigW_integral = np.sum(sigW['weight'])
bkg_integral  = np.sum(bkg ['weight'])

## NOTE: set the number of bin edges -1 and not the number of bins (i.e. 41 edges for 40 bins)
bins = np.linspace(1.6, 2.0, 41)

sigW_mass_sum_weight = []
bkg_mass_sum_weight  = []

for ibin in range(len(bins)-1):
    m_min = bins[ibin]
    m_max = bins[ibin+1]

    sigW_mass_sum_weight.append( np.sum(sigW[(sigW.cand_refit_tau_mass>=m_min) & (sigW.cand_refit_tau_mass<m_max)]['weight']))
    bkg_mass_sum_weight .append( np.sum(bkg [(bkg .cand_refit_tau_mass>=m_min) & (bkg .cand_refit_tau_mass<m_max)]['weight']))

sigW_mass_weights = np.array(sigW_mass_sum_weight) / sigW_integral
bkg_mass_weights  = np.array(bkg_mass_sum_weight)  / bkg_integral

@np.vectorize
def massWeighterSigW(mass):
    bin_low = np.max(np.where(mass>=bins))
    return sigW_mass_weights[bin_low]

@np.vectorize
def massWeighterBkg(mass):
    bin_low = np.max(np.where(mass>=bins))
    return bkg_mass_weights[bin_low]

## mcweight will be used in combine
sigW['mcweight'] = sigW['weight']
bkg ['mcweight'] = np.ones(bkg.shape[0]).astype(np.int)

## weight will be used by XGBoost
sigW['weight'] *= 1. / massWeighterSigW(sigW['cand_refit_tau_mass'])
bkg ['weight'] *= 1. / massWeighterBkg (bkg ['cand_refit_tau_mass'])

# further weight adjustment
sigW['weight'] *= 1.
bkg ['weight'] *= 1. * len(sigW) / len(bkg)

##########################################################################################
#####   COMPACTIFY AND ADD THE (POG) MUON ID TO THE SAMPLES
##########################################################################################
@np.vectorize
def muID(loose, medium, tight):
    if   tight  > 0.5: return 3
    elif medium > 0.5: return 2
    elif loose  > 0.5: return 1
    else             : return 0

@np.vectorize
def tauEta(eta):
    if   abs(eta) > 2.1 : return 7
    elif abs(eta) > 1.8 : return 6
    elif abs(eta) > 1.5 : return 5
    elif abs(eta) > 1.1 : return 4
    elif abs(eta) > 0.8 : return 3
    elif abs(eta) > 0.5 : return 2
    elif abs(eta) > 0.2 : return 1
    else                : return 0

for mu in [1,2,3]:
    name = 'mu%iID' % mu
    features.append(name)
    for dd in [bkg, sigW]:
        dd[name] = muID(
            dd['mu%d_refit_muonid_loose'  % mu], 
            dd['mu%d_refit_muonid_medium' % mu], 
            dd['mu%d_refit_muonid_tight'  % mu],
        )

for dd in [bkg, sigW]:
    dd['tauEta'                                   ] = tauEta(dd['cand_refit_tau_eta'])                           
    #dd['abs(cand_refit_dPhitauMET)'               ] = abs(dd['cand_refit_dPhitauMET'])
    dd['abs(mu1_z-mu2_z)'                         ] = abs(dd['mu1_z']-dd['mu2_z'])                                
    dd['abs(mu1_z-mu3_z)'                         ] = abs(dd['mu1_z']-dd['mu3_z'])                                
    dd['abs(mu2_z-mu3_z)'                         ] = abs(dd['mu2_z']-dd['mu3_z'])             
    dd['cand_refit_tau_pt*(cand_refit_met_pt**-1)'] = dd['cand_refit_tau_pt']/dd['cand_refit_met_pt']                  
    #dd['cand_refit_tau_pt/cand_refit_met_pt'      ] = dd['cand_refit_tau_pt']/dd['cand_refit_met_pt']
    
##########################################################################################
#####   ETA BINS
##########################################################################################
@np.vectorize
def tauEta(eta):
    if   abs(eta) > 2.1 : return 7
    elif abs(eta) > 1.8 : return 6
    elif abs(eta) > 1.5 : return 5
    elif abs(eta) > 1.1 : return 4
    elif abs(eta) > 0.8 : return 3
    elif abs(eta) > 0.5 : return 2
    elif abs(eta) > 0.2 : return 1
    else                : return 0
    
features.append('tauEta')
sigW['tauEta'] = tauEta(sigW['cand_refit_tau_eta'])
bkg ['tauEta'] = tauEta(bkg ['cand_refit_tau_eta'])

##########################################################################################
data = pd.concat([sigW, bkg], ignore_index = True, sort=True)
#data['id'] = np.arange(len(data))
train, test  = train_test_split(data ,  test_size = 0.4, random_state=1986)
train, valid = train_test_split(train,  test_size = 0.2, random_state=1986)

#train, valid = train_test_split(train,  test_size = 0.1, random_state=1986)
## assign an id to the test and train sets seprately to avoid mismatch when folding
#train.insert(len(train.columns), 'id', np.arange(len(train)))
#test .insert(len(test .columns), 'id', np.arange(len(test )))
#valid.insert(len(valid.columns), 'id', np.arange(len(valid)))
#import pdb; pdb.set_trace()

if __name__ == '__main__':
    print "[INFO] Interactive mode: saving dataset to disk"
    import root_pandas
    root_pandas.to_root(data, 'dataframe.root', key='tree')

