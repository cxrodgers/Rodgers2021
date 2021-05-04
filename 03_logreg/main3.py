## This whole directory runs behavioral decoders
# main1-5 is for running decoders
# main3 is for running reduced models using subsampling

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
balancing_method = 'subsampling'

stratify_by_l = [
    ('rewside', 'choice'), 
    ]

# Which datasets to run
datasets = ['no_opto']

# Models to run
reduced_models = [
    # This is the "optimized behavioral decoder"
    'contact_binarized+anti_contact_count+angle',
]


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
        results_dir = os.path.join(params['logreg_dir'], 'reduced_model_results_sbrc_subsampling')
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


        ## Iterate over models
        for reduced_model in reduced_models:
            # Announce
            print(reduced_model)
            
            
            ## Store here
            dataset_component_model_dir = os.path.join(
                dataset_model_dir, reduced_model)
            if not os.path.exists(dataset_component_model_dir):
                os.mkdir(dataset_component_model_dir)

            
            ## Extract components, maintaining the levels for consistency
            if reduced_model == 'contact_binarized+anti_contact_count+angle':
                feature_set = my.misc.fetch_columns_with_error_check(
                    features, [
                    'contact_binarized', 
                    'angle', 
                    'anti_contact_count', 
                    ])   

            elif reduced_model == 'contact_binarized+anti_contact_count':
                feature_set = my.misc.fetch_columns_with_error_check(
                    features, [
                    'contact_binarized', 
                    'anti_contact_count', 
                    ]) 
            
            elif reduced_model == 'contact_binarized+anti_contact_count+angle+anti_angle_max':
                feature_set = my.misc.fetch_columns_with_error_check(
                    features, [
                    'contact_binarized', 
                    'anti_contact_count', 
                    'angle',
                    'anti_angle_max',
                    ])                 

            else:
                raise ValueError("unknown reduced model")
            
            # Remove levels because these will be iterated over
            feature_set.columns = feature_set.columns.remove_unused_levels()

            
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
