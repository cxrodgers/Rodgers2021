## Neural encoding model
## main1a : Mangle the feature names for each model


import os
import json
import pandas
import numpy as np
import my


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Do each model in turn
model_names = [
    # Full (too big to fit, but useful for extracting the features)
    'full',

    # Null model
    'null',

    # NULL_PLUS models -- This is how to identify potentially useful factors
    'whisking',
    'contact_binarized',
    'task',
    'fat_task',
    
    # The minimal model
    'minimal',
    
    # Minimal with whisk permutation
    'minimal+permute_whisks_with_contact',

    # Minimal with random_regressor
    'minimal+random_regressor',

    # MINIMAL_MINUS models
    # Whether the minimal model contains anything unnecessary
    'minimal-whisking',
    'minimal-contacts',
    'minimal-task',
    
    # CONTACTS_PLUS models
    # This identifies any additional features about contacts that matter at all
    'contact_binarized+contact_interaction',
    'contact_binarized+contact_angle',
    'contact_binarized+kappa_min',
    'contact_binarized+kappa_max',
    'contact_binarized+kappa_std',
    'contact_binarized+velocity2_tip',
    'contact_binarized+n_within_trial',
    'contact_binarized+contact_duration',
    'contact_binarized+contact_stimulus',
    'contact_binarized+xw_latency_on',
    'contact_binarized+phase',
    'contact_binarized+xw_angle',
    'contact_binarized+touching',
    
    # CONTACTS_MINUS
    # Currently this is just to test whether whisker identity matters
    'contact_count_by_time',
    
    # WHISKING
    # To compare the coding for position of each whisker
    'start_tip_angle+amplitude_by_whisker',
    'start_tip_angle+global_amplitude',
]


## Paths
# Where to put mangled features
session_mangled_features_dir = os.path.join(
    params['glm_dir'], 'session_mangled_features')
my.misc.create_dir_if_does_not_exist(session_mangled_features_dir)


## Iterate over models
for model_name in model_names:
    ## Where to dump the mangled features
    model_session_mangled_features_dir = os.path.join(
        session_mangled_features_dir, model_name)
    
    if not os.path.exists(model_session_mangled_features_dir):
        os.mkdir(model_session_mangled_features_dir)
    
    
    ## Load features from main0a1
    features = pandas.read_pickle(os.path.join( 
        params['glm_dir'], 'features', model_name, 'neural_unbinned_features'))


    ## For each session, mangle features and dump
    for session_name in features.index.levels[0]:
        # Get features from this session
        session_mangled_features = features.loc[session_name].copy()        
        
        # Mangle the columns
        if len(session_mangled_features.columns.names) > 1:
            session_mangled_features.columns = ['^'.join(map(str, idx)) 
                for idx in session_mangled_features.columns]

        # Dump data with mangled column names
        session_mangled_features.to_pickle(
            os.path.join(model_session_mangled_features_dir, session_name))
