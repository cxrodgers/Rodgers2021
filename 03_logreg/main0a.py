## This whole directory runs behavioral decoders
# This script prepares the features:
#   Extracts relevant cycle, contact, anti-contact, and grasp features
#
# Dumps:
#   unobliviated_unaggregated_features

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
my.misc.create_dir_if_does_not_exist(params['logreg_dir'])


## Load data
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))
big_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_ccs_df'))
big_grasp_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_grasp_df'))
big_binarized_touching_by_cycle = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_binarized_touching_by_cycle'))
big_anti_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_anti_ccs_df2'))


## Extract pan-whisker cycle features from C2_whisk_cycles
# protract_time and retract_time will later be combined into protract_ratio
panwhisker_cycle_features = C2_whisk_cycles.loc[:, [
    'duration', 
    'protract_time', 
    'retract_time',
    ]].copy()

# Rename this one for clarity
panwhisker_cycle_features = panwhisker_cycle_features.rename(
    columns={'duration': 'cycle_duration'})

# protract_time + retract_time = duration
# So make it less redundant
panwhisker_cycle_features['protract_ratio'] = (
    panwhisker_cycle_features['protract_time'].divide(
    panwhisker_cycle_features['cycle_duration']))
panwhisker_cycle_features = panwhisker_cycle_features.drop([
    'protract_time', 'retract_time'], axis=1)


## Extract contact features and index by cycle
# 'whisker' is on the index because we'll unstack and consider each separately
# 'cluster' is on the index because there can be multiple contacts per cycle
contact_features_by_cluster = big_ccs_df.loc[:, [
    'angle', 'cycle', 'duration', 'kappa_max', 'kappa_min', 'kappa_std',
    'phase', # bit weird because it's circular, but mostly contained around 0
    'trial', 
    'velocity2_tip', # the mean of velocity1 and velocity3
    'whisker', 
    'frame_start_wrt_peak', 
    ]].reset_index().set_index(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()

# Rename this one for clarity
contact_features_by_cluster = contact_features_by_cluster.rename(
    columns={'duration': 'contact_duration'})

# Mean over cluster, leaving session * trial * cycle * whisker
contact_features = contact_features_by_cluster.mean(
    level=['session', 'trial', 'cycle', 'whisker'])

# For contact_duration it makes more sense to sum
contact_features['contact_duration'] = contact_features_by_cluster[
    'contact_duration'].sum(
    level=['session', 'trial', 'cycle', 'whisker'])

# Add the 'count' by whisker
# Count contacts per session * trial * cycle * whisker
contact_features['contact_count'] = contact_features_by_cluster.groupby(
    ['session', 'trial', 'cycle', 'whisker']).size()

# Unstack whisker
contact_features.columns.name = 'metric'
contact_features = contact_features.unstack('whisker')


## Append touching to contact_features
to_concat = big_binarized_touching_by_cycle.loc[contact_features.index]
assert not to_concat.isnull().any().any()
to_concat.columns = pandas.MultiIndex.from_tuples(
    [('touching', whisker) for whisker in to_concat.columns],
    names=['metric', 'whisker'])
concatted = pandas.concat(
    [contact_features, to_concat], axis=1).sort_index(axis=1)
assert len(concatted) == len(contact_features)
contact_features = concatted.copy()


## Extract anti_contact features and index by cycle
# 'whisker' is on the index because we'll unstack and consider each separately
# 'cluster' is on the index because there can be multiple contacts per cycle
anti_contact_features_by_cluster = big_anti_ccs_df.loc[:, [
    'angle', 'angle_max', 'cycle', 'duration',
    'trial', 
    'whisker', 
    'frame_start_wrt_peak', 
    ]].reset_index().set_index(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()

# Rename this one for clarity
anti_contact_features_by_cluster = anti_contact_features_by_cluster.rename(
    columns={'duration': 'anti_contact_duration'})

# Mean over cluster, leaving session * trial * cycle * whisker
anti_contact_features = anti_contact_features_by_cluster.mean(
    level=['session', 'trial', 'cycle', 'whisker'])

# For contact_duration it makes more sense to sum
anti_contact_features['anti_contact_duration'] = anti_contact_features_by_cluster[
    'anti_contact_duration'].sum(
    level=['session', 'trial', 'cycle', 'whisker'])

# Add the 'count' by whisker
# Count contacts per session * trial * cycle * whisker
anti_contact_features['anti_contact_count'] = anti_contact_features_by_cluster.groupby(
    ['session', 'trial', 'cycle', 'whisker']).size()

# Make sure every metric starts with "anti_" to avoid collision with contact features
anti_contact_features.columns = [
    column if column.startswith('anti_') else 'anti_{}'.format(column) 
    for column in anti_contact_features.columns]

# Unstack whisker
anti_contact_features = anti_contact_features.unstack('whisker')


## Extract grasp features and index by cycle
# there is only one grasp per cycle so no need for additional indexing or
# grouping at this time
grasp_features = big_grasp_df.loc[:, [
    'label_noC0', # This is replaced below with something else
    'C1vC2_latency_on', 'C3vC2_latency_on', 
    'C1vC2_latency_off', 'C3vC2_latency_off', 
    'C1vC2_angle', 'C3vC2_angle', 
    'C1vC2_duration', 'C3vC2_duration'
    ]].reset_index().set_index(
    ['session', 'trial', 'cycle']).sort_index()

# Convert labels into indicators
grasp_counts = my.decoders.to_indicator_df(
    grasp_features.pop('label_noC0'))


## Concatenate all features so far
# Normalize columns to two levels
panwhisker_cycle_features.columns = pandas.MultiIndex.from_tuples([
    (metric, 'all') for metric in panwhisker_cycle_features.columns],
    names=['metric', 'label'])

grasp_features.columns = pandas.MultiIndex.from_tuples([(
    'xw_' + ('_'.join(colname.split('_')[1:])),
    colname.split('_')[0], 
    ) for colname in grasp_features.columns],
    names=['metric', 'label'])

grasp_counts.columns = pandas.MultiIndex.from_tuples([
    ('grasp_count', label) for label in grasp_counts.columns],
    names=['metric', 'label'])

anti_contact_features.columns.names = ['metric', 'label']
contact_features.columns.names = ['metric', 'label']

# Concat
# grasp_features, contact_features, and grasp_counts lack indexes with no contacts
# these will be filled with nan here
features = pandas.concat([
    contact_features, 
    anti_contact_features,
    grasp_features, 
    grasp_counts,
    panwhisker_cycle_features, 
    ], axis=1, verify_integrity=True, sort=True).sort_index(axis=1)


## Fill null counts with zeroes
features['contact_count'] = features[
    'contact_count'].fillna(0).astype(np.int)
features['anti_contact_count'] = features[
    'anti_contact_count'].fillna(0).astype(np.int)    
features['grasp_count'] = features[
    'grasp_count'].fillna(0).astype(np.int)
features['touching'] = features[
    'touching'].fillna(0).astype(np.int)


## Parameterize grasps differently: 
# Parameterize as:
#   binarized_contact_count : whether each whisker made contact
#   contact_interaction : AND combinations of binarized_contact_count
#   surplus_contact_count : additional whisker contacts above one
#
# The interaction terms are only pairwise, and they ignore C0

# Binarized contact indicator
binarized_contacts = (
    features['contact_count'] > 0).fillna(0).astype(np.int)

# Surplus contacts
surplus_contacts = features['contact_count'] - binarized_contacts

# Interactions to consider
interactions_l = [
    ('C1', 'C2'), ('C2', 'C3'), ('C1', 'C3'),
    ('C1', 'C2', 'C3'),
    ]

# Generate interaction terms
interaction_value_l = []
interaction_keys_l = []
for interaction in interactions_l:
    # AND all of the whiskers
    interaction_value = (
        binarized_contacts.loc[:, interaction] > 0).all(1).astype(np.int)
    
    # Store
    interaction_value_l.append(interaction_value)
    interaction_keys_l.append('-'.join(sorted(interaction)))

# Concat interaction_df
interaction_df = pandas.concat(
    interaction_value_l, keys=interaction_keys_l, axis=1, names=['label'])

# Add levels
binarized_contacts.columns = pandas.MultiIndex.from_tuples([
    ('contact_binarized', whisker) 
    for whisker in binarized_contacts.columns],
    names=['metric', 'label'])
surplus_contacts.columns = pandas.MultiIndex.from_tuples([
    ('contact_surplus', whisker) 
    for whisker in surplus_contacts.columns],
    names=['metric', 'label'])
interaction_df.columns = pandas.MultiIndex.from_tuples([
    ('contact_interaction', whisker) 
    for whisker in interaction_df.columns],
    names=['metric', 'label'])


## Replace contact_count and grasp_count with this new parameterization
# Drop the old ways of measuring contacts
features = features.drop(
    ['contact_count', 'grasp_count'], axis=1)

# Replace with the new ways
concatted = pandas.concat([
    features, 
    binarized_contacts, 
    surplus_contacts, 
    interaction_df,
    ], axis=1, verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(features)
features = concatted.copy()


## Remove the contact onset cycle from 'touching'
# Get both
stacked_cb = features['contact_binarized'].stack()
stacked_t = features['touching'].stack()
assert (stacked_t.loc[stacked_cb > 0] > 0).all()

# Subtract
stacked_t2 = stacked_t - stacked_cb
assert stacked_t2.min() >= 0
assert (stacked_t2.loc[stacked_cb > 0] == 0).all()

# Store
to_concat = stacked_t2.unstack('label')
to_concat.columns = pandas.MultiIndex.from_tuples([
    ('touching', whisker) 
    for whisker in to_concat.columns],
    names=['metric', 'label'])
features = features.drop('touching', axis=1)
concatted = pandas.concat([
    features, 
    to_concat,
    ], axis=1, verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(features)
features = concatted.copy()


## Error check
assert not features.isnull().all(1).any()


## Dump
# The unobliviated, unaggregated features
features.to_pickle(
    os.path.join(params['logreg_dir'], 'unobliviated_unaggregated_features'))
