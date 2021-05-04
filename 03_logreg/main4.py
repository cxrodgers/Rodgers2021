## This whole directory runs behavioral decoders
# main1-5 is for running decoders
# main4 is for running decoders on the full dataset
#
# DATASET=no_opto and STRATIFY_BY=('rewside', 'choice') is used in most cases
# DATASET=no_opto and STRATIFY_BY=None is used in 04_logreg_vis/main1a.py to make figure S3A
# Not clear to me now whether no_licks* is ever used? So I deleted it


import json
import os
import pandas
import numpy as np
import tqdm
import my.decoders

## Extra strict: raise Exception on ConvergenceWarning
# Shouldn't actually happen anymore
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('error', ConvergenceWarning) 


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Paths
datasets_dir = os.path.join(params['logreg_dir'], 'datasets')


## Parameters
balancing_method = 'sample weighting'

stratify_by_l = [
    ('rewside', 'choice'), 
    None,
    ]

## Which datasets to run
datasets = ['no_opto',]

# Regularizations
reg_l = np.linspace(-6, 6, 13)
to_optimize = 'weighted_correct'

# Splitting
# Increasing this mitigates the problem of small trial counts per class
n_splits = 7

# stratify_by
for stratify_by in stratify_by_l:
    print("stratify_by: {}".format(stratify_by))


    # Store individual model results here
    if stratify_by == ('rewside', 'choice'):
        results_dir = os.path.join(params['logreg_dir'], 'full_model_results_sbrc')
    elif stratify_by is None:
        results_dir = os.path.join(params['logreg_dir'], 'full_model_results_sbnull')
    else:
        1/0
    
    my.misc.create_dir_if_does_not_exist(results_dir)


    ## Iterate over datasets
    for dataset in datasets:
        # Announce
        print("dataset: {}".format(dataset))
        
        # random seed (used for breaking symmetry in stratification)
        np.random.seed(0)

       
        ## Load data from this dataset
        dataset_dir = os.path.join(datasets_dir, dataset)
        features = pandas.read_pickle(os.path.join(dataset_dir, 'features'))
        labels = pandas.read_pickle(os.path.join(dataset_dir, 'labels'))

        
        ## Drop sessions with insufficient trials
        n_trials = labels.groupby(
            ['session', 'rewside', 'choice']).size().unstack(
            'session').T.fillna(0).astype(np.int)        

        # threshold
        bad_sessions = n_trials.index[n_trials.min(1) < 5]
        
        # drop
        if len(bad_sessions) > 0:
            print("dropping {} sessions with too few trials".format(
                len(bad_sessions)))
            features = features.drop(bad_sessions).copy()
            features.index = features.index.remove_unused_levels()
            labels = labels.drop(bad_sessions).copy()
            labels.index = labels.index.remove_unused_levels()
        

        ## Where to save the results for this dataset
        dataset_model_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_model_dir):
            os.mkdir(dataset_model_dir)    


        ## Indent to facilitate meld
        if True:
            ## Full model, so use all the features
            feature_set = features

            # Store here
            dataset_component_model_dir = dataset_model_dir


            ## Run decoders on all targets * sessions
            res = my.decoders.iterate_behavioral_classifiers_over_targets_and_sessions(
                feature_set=feature_set, labels=labels, 
                reg_l=reg_l, to_optimize=to_optimize, n_splits=n_splits,
                stratify_by=stratify_by,
                balancing_method=balancing_method,
                )

            
            ## Store in dataset_component_model_dir
            for key, value in res.items():
                filename = os.path.join(dataset_component_model_dir, key)
                value.to_pickle(filename)
