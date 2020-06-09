from collections import OrderedDict
labels = OrderedDict()

labels['cand_refit_tau_pt'                            ] = '$\\tau$ $p_{T}$'
labels['cand_refit_mttau'                             ] = '$m_{T}(\\tau, MET)$'
labels['cand_refit_tau_dBetaIsoCone0p8strength0p2_rel'] = '$\\tau$ iso'
labels['abs(cand_refit_dPhitauMET)'                   ] = '$\Delta\phi(\\tau MET)$'
labels['cand_refit_met_pt'                            ] = 'MET $p_{T}$'
labels['cand_refit_tau_pt/cand_refit_met_pt'          ] = '$\\tau$ $p_{T}$/MET $p_{T}$'     # only for >= v2
labels['cand_refit_tau_pt*(cand_refit_met_pt**-1)'    ] = '$\\tau$ $p_{T}$/MET $p_{T}$'     # only for >= v2
labels['cand_refit_dRtauMuonMax'                      ] = 'max($\Delta R(\\tau \mu_{i})$)'
labels['cand_refit_w_pt'                              ] = 'W $p_{T}$'
labels['cand_refit_mez_1'                             ] = '$max(ME_z^i)$'
labels['cand_refit_mez_2'                             ] = '$min(ME_z^i)$'
labels['abs(mu1_z-mu2_z)'                             ] = '$\Delta z (\mu_1, \mu_2)$'
labels['abs(mu1_z-mu3_z)'                             ] = '$\Delta z (\mu_1, \mu_3)$'
labels['abs(mu2_z-mu3_z)'                             ] = '$\Delta z (\mu_2, \mu_3)$'
labels['tau_sv_ls'                                    ] = 'SV L/$\sigma$'
labels['tau_sv_prob'                                  ] = 'SV prob'
labels['log(-log(tau_sv_prob))'                       ] = 'log(-log(SV prob))'
labels['tau_sv_cos'                                   ] = 'SV cos($\\theta_{IP}$)'
labels['mu1ID'                                        ] = '$\mu_1$ ID'
labels['mu2ID'                                        ] = '$\mu_2$ ID'
labels['mu3ID'                                        ] = '$\mu_3$ ID'
labels['tauEta'                                       ] = '$|\eta_{\\tau}|$'
labels['bdt'                                          ] = 'BDT'
labels['cand_refit_tau_mass'                          ] = '$\\tau$ mass'
labels['year'                                         ] = 'year'

features = [
    'cand_refit_tau_pt',
    'cand_refit_mttau',
    'cand_refit_tau_dBetaIsoCone0p8strength0p2_rel',
    'abs(cand_refit_dPhitauMET)',
    'cand_refit_met_pt',
    'cand_refit_tau_pt*(cand_refit_met_pt**-1)',    # only for >= v4
#    'cand_refit_tau_pt/cand_refit_met_pt',          # only for >= v2
#    'cand_refit_dRtauMuonMax',
    'cand_refit_w_pt',
    'cand_refit_mez_1',
    'cand_refit_mez_2',
    'abs(mu1_z-mu2_z)', 
    'abs(mu1_z-mu3_z)', 
    'abs(mu2_z-mu3_z)',
    'tau_sv_ls',
    'tau_sv_prob',
    #'log(-log(tau_sv_prob))',
    'tau_sv_cos',
]

core_features = [ff for ff in features] ## import purpose

branches = features + [
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_fired',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_matched',
    'HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_fired',
    'HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_matched',
    'HLT_DoubleMu3_Trk_Tau3mu_fired',
    'HLT_DoubleMu3_Trk_Tau3mu_matched',
    'cand_refit_charge',
    'cand_refit_tau_eta',
    'cand_refit_tau_mass',
    'cand_refit_mass12',
    'cand_refit_mass13',
    'cand_refit_mass23',
    'cand_charge',
    'cand_charge12',
    'cand_charge13',
    'cand_charge23',
    'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
    'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
    'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
    'cand_refit_dPhitauMET',
    'mu1_z', 'mu2_z','mu3_z',
    #'HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_matched',
    #'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_matched',
]