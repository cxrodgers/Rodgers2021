## Descriptive plots
# Dumps psth data by cycle type and cell type to be plotted in main1b

import json
import pandas
import numpy as np
import os
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)

units = 'fold' #'Hz' # 'percent'
    
    
## Set up plotting
my.plot.manuscript_defaults()
my.plot.font_embed()
this_WHISKER2COLOR = {'C0': 'gray', 'C1': 'b', 'C2': 'g', 'C3': 'r'}


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load behavioral features
behavioral_features = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features')


## Load unit metadata
# Drop 1 and 6b here so they can be excluded later
big_waveform_info_df = my.dataload.load_bwid(params, drop_1_and_6b=True)
    

## Load results of main0b
# These are all for neural sessions only
big_sliced_touching = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'big_sliced_touching'))
big_sliced_tip_pos = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'big_sliced_tip_pos'))
big_psth = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'big_psth'))
big_alfws = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'big_alfws'))    
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'spikes_by_cycle'))


## Harmonize the cycles in big_alfws and glm_features
# Cycles from original_features (including sessions without neural data)
behavioral_features_cycles = pandas.MultiIndex.from_frame(
    behavioral_features.index.to_frame().reset_index(drop=True)[
    ['session', 'trial', 'cycle']].drop_duplicates())

# Cycles for which we have spike counts by frame (from main0b)
big_alfws_cycles = pandas.MultiIndex.from_frame(
    big_alfws.index.to_frame().reset_index(drop=True)[
    ['session', 'trial', 'cycle']].drop_duplicates())

# Spike cycles should be a subset of behavioral cycles
assert big_alfws_cycles.isin(behavioral_features_cycles).all()


## Keep only the features for which we have neural data in big_alfws
orig_behavioral_features = behavioral_features.copy()
behavioral_features = my.misc.slice_df_by_some_levels(
    behavioral_features, big_alfws_cycles)


## Calculate mean FR per bin for normalization
# Consider also using the session mean FR here
mean_FR_per_bin = big_psth.mean(level=['session', 'neuron'])


## Join metadata to group on
big_psth_with_metadata = big_psth.to_frame()

# Join layer and NS
# This inner join drops neurons that aren't in bwid (1 and 6b)
# All of these joins take a long time
big_psth_with_metadata = big_psth_with_metadata.join(
    big_waveform_info_df[['stratum', 'NS']], 
    on=['session', 'neuron'], how='inner')

# Calculate cycle features to group on
has_contact = (behavioral_features['contact_binarized'] > 0).any(1)
has_anti_contact = (behavioral_features['anti_contact_count'] > 0).any(1)
cycle_features = pandas.concat(
    [has_contact, has_anti_contact], 
    keys=['contact', 'anti_contact'], axis=1)

# If it has both contact and anti_contact, score as contact
cycle_features['typ'] = 'none'
cycle_features.loc[
    cycle_features['anti_contact'].values, 'typ'] = 'anti_contact'
cycle_features.loc[
    cycle_features['contact'].values, 'typ'] = 'contact'

# Join cycle type
big_psth_with_metadata = big_psth_with_metadata.join(
    cycle_features['typ'], on=['session', 'trial', 'cycle'])

# Error check
assert not big_psth_with_metadata.isnull().any().any() 


## Add cycle type and layer to the MultiIndex
big_psth_with_metadata = big_psth_with_metadata.set_index(
    ['stratum', 'NS', 'typ'], append=True)


## First mean over trial * cycle, but maintaining neuron and shift
# Group over everything except trial * cycle
gobj = big_psth_with_metadata.groupby([
    lev for lev in big_psth_with_metadata.index.names 
    if lev not in ['trial', 'cycle']
    ])['spikes']

# Aggregate
# This takes a long time
mean_by_neuron_and_shift = gobj.mean().dropna()
n_by_neuron_and_shift = gobj.size()


## Dump for plotting in main1b
mean_FR_per_bin.to_pickle('mean_FR_per_bin')
mean_by_neuron_and_shift.to_pickle('mean_by_neuron_and_shift')
n_by_neuron_and_shift.to_pickle('n_by_neuron_and_shift')


