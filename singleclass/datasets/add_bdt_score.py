import pandas
import root_pandas
import numpy

import os
import sys
sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/libs'    )
sys.path.append('/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/datasets')

print 'Loading dataset'
from W_dataset  import data ; data = data[data['target'] == 1]
from trainer    import add_bdt_score_pck
from features   import features

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

pck = '/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/pck/classifier_26may2020.pck'
path = '/gwpool/users/lguzzi/Tau3Mu/2017_2018/BDT/singleclass/ntuples/signal_2016BDT_16apr2018_v16.root'

if os.path.exists(path):
    print 'file %s', path, 'already exists. Exit'
    sys.exit()
    
print 'Adding BDT score'
data['id'] = numpy.arange(len(data))
add_bdt_score_pck(pck, data, features)
root_pandas.to_root(data, path, key = "tree")

print 'ntuple saved to %s' %path