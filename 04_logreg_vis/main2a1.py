## main2a* is for plotting weights
# main2a1 is for preparing this data
#
# uses data from:
#   reduced_model_results_sbrc/no_opto
#   reduced_model_results_sbrc/no_opto_no_licks1
#   reduced_model_results_sbrc_subsampling/no_opto
#   DATASET/features

import json
import os
import pandas
import numpy as np
import my.decoders


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Which models to load
# Include everything needed by main2a2 or main3b
reduced_models = [
    'contact_binarized+anti_contact_count+angle',
    'contact_binarized+anti_contact_count+angle+anti_angle_max',
]

# Partition features with these names as raw
raw_features_names = [
    'contact_binarized',
    'contact_count_total',
    'contact_count_by_time',
    'contact_count_by_whisker',
    'contact_count_total',
    'contact_interaction',
    'contact_interaction_count_by_label',
    'contact_surplus',
    'task',
    'anti_contact_count',
    ]


## Which datasets to include
# This goes dirname, dataset, model
iterations = [
    ('reduced_model_results_sbrc', 'no_opto', 'contact_binarized+anti_contact_count+angle',),
    ('reduced_model_results_sbrc', 'no_opto', 'contact_binarized+anti_contact_count+angle+anti_angle_max',),
    ('reduced_model_results_sbrc', 'no_opto_no_licks1', 'contact_binarized+anti_contact_count+angle',),
    ('reduced_model_results_sbrc_subsampling', 'no_opto', 'contact_binarized+anti_contact_count+angle',),
    ]


## Load data from each dataset * model in turn
weights_part_l = []
icpt_transformed_part_l = []
keys_l = []
for dirname, dataset, model in iterations:
    ## Identify subsampling or not
    if 'subsampling' in dirname:
        subsampling = True
    else:
        subsampling = False

    ## Load features
    # We need these to partition the results from the reduced models
    print("loading features")
    full_model_features = pandas.read_pickle(os.path.join(
        params['logreg_dir'], 'datasets', dataset, 'features'))    
    print("done")
    

    ## Path to model results
    # Path to model
    model_dir = os.path.join(
        params['logreg_dir'], dirname, dataset, model)
    
    
    ## Load
    model_res = my.decoders.load_model_results(model_dir)    

    
    ## Extract the features used in this model from the full set
    model_features = full_model_features.loc[:, 
        model_res['weights'].columns]

    # Remove levels
    model_features.index = model_features.index.remove_unused_levels()
    model_res['weights'].index = model_res[
        'weights'].index.remove_unused_levels()

    
    ## Partition
    # Identify which features are raw
    raw_mask = pandas.Series(
        model_res['weights'].columns.get_level_values('metric').isin(
        raw_features_names),
        index=model_res['weights'].columns)
    
    # TODO: remove dependence on model_features here, by partitioning
    # just the weights, not the features
    part_res = my.decoders.partition(model_features, model_res, raw_mask)

    
    ## Add mouse as level
    part_res['weights_part'].index = pandas.MultiIndex.from_tuples([
        (session.split('_')[1], session, decode_label)
        for session, decode_label in part_res['weights_part'].index],
        names=['mouse', 'session', 'decode_label'])
    part_res['icpt_transformed_part'].index = pandas.MultiIndex.from_tuples([
        (session.split('_')[1], session, decode_label)
        for session, decode_label in part_res['icpt_transformed_part'].index],
        names=['mouse', 'session', 'decode_label'])


    ## Store
    weights_part_l.append(part_res['weights_part'])
    icpt_transformed_part_l.append(part_res['icpt_transformed_part'])
    keys_l.append((subsampling, dataset, model))


## Concat
big_weights_part = pandas.concat(
    weights_part_l, axis=1, keys=keys_l, names=['subsampling', 'dataset', 'model'])
big_icpt_part = pandas.concat(
    icpt_transformed_part_l, axis=1, keys=keys_l, names=['subsampling', 'dataset', 'model'])


## Dump
big_weights_part.to_pickle('big_weights_part')
big_icpt_part.to_pickle('big_icpt_part')