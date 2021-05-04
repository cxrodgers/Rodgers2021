## This whole directory runs behavioral decoders
# This script slices the aggregated_features by dataset and dumps.
# This is the first point where opto trials are dropped.
#
# Dumps for each dataset:
#   features : aggregated, obliviated features for that set of trials
#   labels : rewside, choice, and servo_pos for each trial


import json
import pandas
import numpy as np
import my
import my.decoders
import os


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Create directories
# Features and labels will be dumped here
datasets_dir = os.path.join(params['logreg_dir'], 'datasets')
my.misc.create_dir_if_does_not_exist(datasets_dir)


## Load data
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))

# Results from main0b
aggregated_features = pandas.read_pickle(
    os.path.join(params['logreg_dir'], 'obliviated_aggregated_features'))    


## Load lick data
# Load licks
big_licks = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_licks'))

# Binarize big_licks
# Because there's rarely more than 1 per bin, and when it does happen 
# it's probably artefactual
big_licks.values[big_licks.values > 1] = 1

# Massage big licks
# Get session * trial on rows, and lick * time on columns
big_licks.index.name = 'time'
big_licks = big_licks.T.unstack('lick')
big_licks = big_licks.reorder_levels(
    ['lick', 'time'], axis=1).sort_index(axis=1)

# Rename lick human readable
big_licks = big_licks.rename({1: 'left', 2: 'right'}, axis=1, level='lick')
big_licks = big_licks.sort_index(axis=1)

# Sum over left and right
big_licks = big_licks.sum(axis=1, level='time')

# Sum over two time windows
time_values = big_licks.columns
time_window1 = time_values[time_values < -1]
time_window2 = time_values[time_values < -0.5]
summed_time_window1 = big_licks.loc[:, time_window1].sum(1)
summed_time_window2 = big_licks.loc[:, time_window2].sum(1)

# Join on big_tm
big_tm = big_tm.join(
    summed_time_window1.rename('n_licks1')).join(
    summed_time_window2.rename('n_licks2'))
assert not big_tm.loc[:, ['n_licks1', 'n_licks2']].isnull().any().any()


## Get labels
# Must include here any decode targets and any other parameters to be used
# for trial balancing
labels = big_tm[['rewside', 'choice']].copy()


## Dump each dataset
assert not big_tm['munged'].any()
print("saving")
for dataset in ['no_opto', 'no_opto_no_licks1',]:
    ## Choose the trials to include
    if dataset == 'no_opto':
        # Random trials, no opto
        this_big_tm = big_tm[
            big_tm['opto'].isin([0, 2]) &
            big_tm['isrnd']
        ]
    
    elif dataset == 'no_opto_no_licks1':
        # Random trials, no opto, few licks
        this_big_tm = big_tm[
            big_tm['opto'].isin([0, 2]) &
            big_tm['isrnd'] &
            (big_tm['n_licks1'] == 0)
        ]
    
    else:
        1/0


    ## Slice features and labels accordingly
    dataset_features = aggregated_features.loc[this_big_tm.index]
    dataset_labels = labels.loc[this_big_tm.index]
    
    # Remove unused levels
    dataset_features.index = dataset_features.index.remove_unused_levels()
    dataset_features.columns = dataset_features.columns.remove_unused_levels()
    dataset_labels.index = dataset_labels.index.remove_unused_levels()
    

    ## Dump
    dataset_dir = os.path.join(datasets_dir, dataset)
    if not os.path.exists(dataset_dir):
        print("creating: %s" % dataset_dir)
        os.mkdir(dataset_dir)
    dataset_features.to_pickle(os.path.join(dataset_dir, 'features'))
    dataset_labels.to_pickle(os.path.join(dataset_dir, 'labels'))