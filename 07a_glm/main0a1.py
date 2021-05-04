## Neural encoding model
## main0a1 : Construct neural features
# These are mostly the same as unbinned_features in the logreg dir
# Adding a few:
#   * ordinality
#   * whisking (intentionally left out of decoding)
#   * contact interactions
#   * task features
#   * log_cycle_duration


import json
import os
import pandas
import numpy as np
import my


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Paths
glm_features_dir = os.path.join(params['glm_dir'], 'features')
my.misc.create_dir_if_does_not_exist(params['glm_dir'])
my.misc.create_dir_if_does_not_exist(glm_features_dir)


## Load trial matrix that went into the spikes
big_tm = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'neural_big_tm'))


## Load patterns and whisk cycles
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')

big_cycle_features = my.dataload.load_data_from_patterns(
    params, 'big_cycle_features')


## Load features
unbinned_features = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features')


## Load spiking data
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'spikes_by_cycle'))


## Slice out cycles after t=-500 frames and cycles before t=200 frames
# This accounts for weird possible edge effects
include_cycles = (
    (C2_whisk_cycles['peak_frame_wrt_rwin'] >= params['glm_start_frame']) &
    (C2_whisk_cycles['peak_frame_wrt_rwin'] < params['glm_stop_frame'])
    )
include_cycles = include_cycles.index[include_cycles.values]

# Slice
spikes_by_cycle = my.misc.slice_df_by_some_levels(
    spikes_by_cycle, include_cycles)
unbinned_features = my.misc.slice_df_by_some_levels(
    unbinned_features, include_cycles)


## Ensure consistent session * trial (* cycle)
# Check that same session * trial are in big_tm and spikes_by_cycle
my.misc.assert_index_equal_on_levels(
    big_tm, spikes_by_cycle, levels=['session', 'trial'])

# Choose these same session * trial in unbinned_features
session_trial_midx = pandas.MultiIndex.from_frame(
    spikes_by_cycle.index.to_frame().reset_index(drop=True)[
    ['session', 'trial']].drop_duplicates())

# Slice these trials
neural_unbinned_features = my.misc.slice_df_by_some_levels(
    unbinned_features, session_trial_midx)
neural_unbinned_features.index = (
    neural_unbinned_features.index.remove_unused_levels())

# Check that big_tm and neural_unbinned_features match on session * trial
my.misc.assert_index_equal_on_levels(
    big_tm, neural_unbinned_features, levels=['session', 'trial'])

# And that spikes_by_cycle and neural_unbinned_features match on 
# session * trial * cycle
my.misc.assert_index_equal_on_levels(
    spikes_by_cycle, neural_unbinned_features, 
    levels=['session', 'trial', 'cycle'])


## Count ordinality of each whisker contact within the trial
# Extract the session, trial, cycle, whisker of every contact
# Extract contact count
cctemp = neural_unbinned_features['contact_binarized'].copy()
cctemp.columns.name = 'whisker'

# Stack and include only session * trial * cycle * whisker on which contact occurred
cctemp = cctemp.stack('whisker')
cctemp = cctemp[cctemp > 0].copy()

# DataFrame the index
cctempidx = cctemp.index.to_frame().reset_index(drop=True)

# Sort by whisker and cycle
cctempidx = cctempidx.sort_values(
    ['session', 'trial', 'whisker', 'cycle'])

assert not cctempidx.duplicated(
    ['session', 'trial', 'whisker', 'cycle']).any()

# Identify first within cycle
cctempidx['first_within_trial'] = ~cctempidx.duplicated(
    ['session', 'trial', 'whisker'])

# Count within trial, restarting at first_within_cycle
cctempidx['n_within_trial'] = range(len(cctempidx))
cctempidx['n_within_trial'] -= cctempidx.loc[
    cctempidx['first_within_trial'].values, 'n_within_trial'].reindex(
    cctempidx.index).ffill().astype(np.int)

# Drop the redundant first_within_trial
assert (cctempidx.loc[cctempidx['first_within_trial'], 'n_within_trial'] == 0).all()
cctempidx = cctempidx.drop('first_within_trial', 1)

# Reform this to be session * trial * cycle on index and metric * label on 
# columns, like neural_unbinned_features
to_join = cctempidx.set_index(
    ['session', 'trial', 'cycle', 'whisker'])[
    'n_within_trial'].unstack('whisker').sort_index()
to_join.columns = pandas.MultiIndex.from_tuples([
    ('n_within_trial', whisker) for whisker in to_join.columns],
    names=['metric', 'label'])

# Concat onto neural_unbinned_features
concatted = pandas.concat(
    [neural_unbinned_features, to_join], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Drop C1-C2-C3 interaction to keep things simple
neural_unbinned_features = neural_unbinned_features.drop(
    ('contact_interaction', 'C1-C2-C3'), axis=1)


## Extract task features
# Code the current and previous outcome as binary variables
current_outcome = big_tm['outcome'].replace({'hit': 1, 'error': -1})
assert current_outcome.isin([1, -1]).all()

# The previous outcome could be 0 if it was a spoil. This rarely happens
previous_outcome = (
    (~big_tm['prev_hit'].isnull()).astype(np.int) -
    (~big_tm['prev_err'].isnull()).astype(np.int)
)
assert previous_outcome.isin([1, 0, -1]).all()

# Current rewside (NB: 'left' means nothing for dection)
current_rewside = big_tm['rewside'].replace({'left': -1, 'right': 1})

# Current stimulus: convex=1, concave=-1, nothing=nan
# This will be redundant with current_rewside during discrimination
current_stimulus = big_tm['stepper_pos'].replace(
    {50: 1, 150: -1, 199: np.nan, 100: np.nan})

# Current choice
current_choice = big_tm['choice'].replace({'left': -1, 'right': 1})
prev_choice = big_tm['prev_choice'].replace({'left': -1, 'right': 1})

# Put into DataFrame
task_features = pandas.DataFrame.from_dict({
    'current_outcome': current_outcome, 
    'previous_outcome': previous_outcome, 
    #~ 'current_rewside': current_rewside,  # because redundant
    'current_choice': current_choice, 
    'prev_choice': prev_choice,
    'current_stimulus': current_stimulus,
    })
task_features.columns.name = 'label'


## Bin each cycle temporally so we can extract the appropriate indicated task feature
# Define the TIV levels
# These will be the edges of the temporal indicators
TIV_TEMPORAL_WIDTH = 100 # bins
tiv_index = pandas.Series(np.arange(
    params['glm_start_frame'], params['glm_stop_frame'] + 1, TIV_TEMPORAL_WIDTH, 
    dtype=np.int)).rename('tiv')
    
# Get the time of every cycle in neural_unbinned_features
each_cycle_frame = C2_whisk_cycles['peak_frame_wrt_rwin'].loc[
    neural_unbinned_features.index]

# Cut the cycle frame by the bins
each_cycle_bin = pandas.cut(
    each_cycle_frame, bins=tiv_index, labels=False, right=False)

# Map it to the start frame of that bin
each_cycle_bin_start_frame = each_cycle_bin.map(tiv_index).rename(
    'bin_start_frame')
assert not each_cycle_bin_start_frame.isnull().any()


## Indicate these task features over time
# Create a new copy of task_features for every TIV
# Exclude the last edge, so that this is now a start frame
wtiv = pandas.concat([task_features] * len(tiv_index[:-1]), 
    keys=tiv_index[:-1], axis=1).stack().stack()

# New column that will be the value of the tiv metric
tiv_val = wtiv.index.get_level_values('tiv').to_series().rename('bin_start_frame')
tiv_val.index = wtiv.index
tiv_concatted = pandas.concat([wtiv.rename('xxx'), tiv_val], axis=1)

# Put locked_frame on columns and reindex with ffill to forward propagate
# the tiv value
tiv_task_features = tiv_concatted.reset_index().set_index(
    list(wtiv.index.names) + ['bin_start_frame'])['xxx'].unstack(
    'bin_start_frame').fillna(0).astype(np.int).reindex(
    tiv_index[:-1].values, 
    axis=1, method='ffill')

# Make index session * trial * bin_start_frame, and columns label * tiv
tiv_task_features = tiv_task_features.unstack(
    ['label', 'tiv']).stack('bin_start_frame').sort_index().sort_index(axis=1)

# Slice the tiv_task_features by the cycle bins
slicing_midx = pandas.MultiIndex.from_frame(
    each_cycle_bin_start_frame.reset_index()[
    ['session', 'trial', 'bin_start_frame']])
sliced_task_features = tiv_task_features.loc[slicing_midx].copy()
#~ assert not sliced_task_features.isnull().any().any() # but now stimulus can be null

# Replace with the original index by cycle instead of start bin frame
sliced_task_features.index = each_cycle_bin_start_frame.index.copy()

# Concat
sliced_task_features.columns = pandas.MultiIndex.from_tuples([
    ('task', '{}_{:03d}'.format(param, value)) 
    for param, value in sliced_task_features.columns],
    names=['metric', 'label'])
concatted = pandas.concat(
    [neural_unbinned_features, sliced_task_features], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Add a fat_task, which is just one indicator per trial
slicing_idx = neural_unbinned_features.index.droplevel('cycle')
fat_task_features = task_features.reindex(slicing_idx)
fat_task_features.index = neural_unbinned_features.index

# add metric level
fat_task_features.columns = pandas.MultiIndex.from_tuples([
    ('fat_task', col) for col in fat_task_features.columns],
    names=['metric', 'label'])

# Concat
concatted = pandas.concat(
    [neural_unbinned_features, fat_task_features], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Add an interaction on contacts and stimulus
# Get the stimulus for every cycle
stim_series = task_features['current_stimulus'].reindex(
    neural_unbinned_features.index.droplevel('cycle'))
stim_series.index = neural_unbinned_features.index

# Repeat for each whisker and then mask below
contact_stim_interaction = pandas.concat(
    [stim_series] * 4, keys=['C0', 'C1', 'C2', 'C3'], axis=1)

# Make it null where no contact
contact_stim_interaction.values[
    neural_unbinned_features['angle'].isnull()] = np.nan

# add metric level
contact_stim_interaction.columns = pandas.MultiIndex.from_tuples([
    ('contact_stimulus', col) for col in contact_stim_interaction.columns],
    names=['metric', 'label'])

# Concat
concatted = pandas.concat(
    [neural_unbinned_features, contact_stim_interaction], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Add a log_cycle_duration parameter
# Log the duration
logged_duration = np.log10(
    neural_unbinned_features.loc[:, ('cycle_duration', 'all')])
logged_duration.name = ('log_cycle_duration', 'all')

# Concat
concatted = pandas.concat(
    [neural_unbinned_features, logged_duration], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()
neural_unbinned_features = neural_unbinned_features.sort_index(axis=1)

# Drop the now-redundant cycle_duration
neural_unbinned_features = neural_unbinned_features.drop(
    ['cycle_duration'], axis=1)
neural_unbinned_features.columns = (
    neural_unbinned_features.columns.remove_unused_levels())


## Add a contact_count_over_whisker parameter
# This was only in the aggregated version during logreg
# Sum contacts by time over whisker
# Consider whether this should be binarized
contact_count_by_time = (
    neural_unbinned_features.loc[:, 'contact_binarized'].sum(axis=1)
    ).to_frame()
anti_contact_count_by_time = (
    neural_unbinned_features.loc[:, 'anti_contact_count'].sum(axis=1)
    ).to_frame()

# Relevel to match the other features: metric, label
contact_count_by_time.columns = pandas.MultiIndex.from_tuples(
    [('contact_count_by_time', 'all')],
    names=['metric', 'label'])
anti_contact_count_by_time.columns = pandas.MultiIndex.from_tuples(
    [('anti_contact_count_by_time', 'all')],
    names=['metric', 'label'])

# Concat
concatted = pandas.concat(
    [neural_unbinned_features, 
    contact_count_by_time, anti_contact_count_by_time], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Add whisking terms, which were explicitly not used for behavioral decoding
# Extract from C2_whisk cycles
global_whisking_terms = C2_whisk_cycles.loc[
    neural_unbinned_features.index, 
    ['amplitude', 'set_point', 'spread', 'C3vC2', 'C1vC2']].copy()
assert not global_whisking_terms.isnull().any().any()

# Also extract for individual whiskers
# C2 here is redundant with amplitude and set_point above
individual_whisking_set_point = big_cycle_features.loc[
    neural_unbinned_features.index,
    'start_tip_angle']
individual_whisking_amplitude = (
    big_cycle_features['peak_tip_angle'] -
    big_cycle_features['start_tip_angle']).loc[neural_unbinned_features.index]

# Concat whisking terms
whisking_terms = pandas.concat([
    global_whisking_terms,
    individual_whisking_set_point,
    individual_whisking_amplitude],
    axis=1, 
    keys=['whisking_global', 'whisking_indiv_set_point', 
    'whisking_indiv_amplitude'])
assert not whisking_terms.isnull().any().any()

# Also smooth whisking terms
smoothed_whisking = whisking_terms.groupby(['session', 'trial']).apply(
    lambda x: x.rolling(
    win_type='triang', min_periods=0, window=3, center=True).mean())
smoothed_whisking.columns = pandas.MultiIndex.from_tuples([
    ('{}_smoothed'.format(lev0), lev1) 
    for lev0, lev1 in smoothed_whisking.columns])

# Concat
concatted = pandas.concat(
    [neural_unbinned_features, whisking_terms, smoothed_whisking], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(neural_unbinned_features)
neural_unbinned_features = concatted.copy()


## Drop some metrics
# Drop the current_stimulus metric, because it's too correlated with
# other task parameters, and is handled more correctly as an interaction
# on contacts anyway
to_drop = [('task', task_col) for task_col in 
    np.unique(neural_unbinned_features.columns.get_level_values(1))
    if task_col.startswith('current_stimulus_')]
to_drop.append(('fat_task', 'current_stimulus'))
neural_unbinned_features = neural_unbinned_features.drop(to_drop, axis=1).copy()

# Remove unused
neural_unbinned_features.columns = (
    neural_unbinned_features.columns.remove_unused_levels())

# Sort
neural_unbinned_features = (
    neural_unbinned_features.sort_index().sort_index(axis=1))


## Apply various reductions
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


# Helper function
def fetch_by_columns(columns):
    # Extract
    to_dump = neural_unbinned_features.loc[:, columns].copy()
    to_dump.columns = to_dump.columns.remove_unused_levels()

    # Error check
    # A missing label will be silently ignored above
    assert list(to_dump.columns.levels[0]) == sorted(columns)    
    
    return to_dump


## Iterate over models
for model_name in model_names:
    ## Full
    if model_name == 'full':
        to_dump = neural_unbinned_features.copy()
    
    
    ## Null
    elif model_name == 'null':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            ])
    
    
    ## NULL_PLUS
    elif model_name == 'whisking':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])

    elif model_name == 'contact_binarized':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            ])

    elif model_name == 'task':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'task', 
            ])

    elif model_name == 'fat_task':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'fat_task', 
            ])

    
    ## Minimal
    elif model_name == 'minimal':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'task', 
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])


    ## Minimal with random_regressor
    elif model_name == 'minimal+random_regressor':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'task', 
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])
        
        # Generate random regressor
        np.random.seed(0)
        random_regressor = pandas.Series(
            np.random.standard_normal(len(to_dump)),
            index=to_dump.index,
            name=('random_regressor', 'random_regressor'),
            )
        
        # Concat and sort
        to_dump = pandas.concat([to_dump, random_regressor], axis=1)
        to_dump = to_dump.sort_index(axis=1)


    ## Minimal with permutation
    elif model_name == 'minimal+permute_whisks_with_contact':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'task', 
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])


    ## MINIMAL_MINUS
    elif model_name == 'minimal-whisking':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'task', 
            ])

    elif model_name == 'minimal-contacts':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'task', 
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])

    elif model_name == 'minimal-task':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'whisking_indiv_set_point_smoothed',
            'whisking_global_smoothed',
            ])


    ## CONTACTS_PLUS models
    elif model_name == 'contact_binarized+contact_interaction':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'contact_interaction',
            ])

    elif model_name == 'contact_binarized+contact_angle':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'angle',
            ])

    elif model_name == 'contact_binarized+kappa_min':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'kappa_min',
            ])

    elif model_name == 'contact_binarized+kappa_max':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'kappa_max',
            ])

    elif model_name == 'contact_binarized+kappa_std':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'kappa_std',
            ])

    elif model_name == 'contact_binarized+velocity2_tip':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'velocity2_tip',
            ])

    elif model_name == 'contact_binarized+n_within_trial':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'n_within_trial',
            ])

    elif model_name == 'contact_binarized+contact_duration':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'contact_duration',
            ])

    elif model_name == 'contact_binarized+contact_stimulus':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'contact_stimulus',
            ])

    elif model_name == 'contact_binarized+xw_latency_on':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'xw_latency_on',
            ])

    elif model_name == 'contact_binarized+phase':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'phase',
            ])

    elif model_name == 'contact_binarized+xw_angle':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'xw_angle',
            ])

    elif model_name == 'contact_binarized+touching':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_binarized',
            'touching',
            ])
    
    
    ## CONTACTS_MINUS
    elif model_name == 'contact_count_by_time':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'contact_count_by_time',
            ])
    
    
    ## WHISKING
    elif model_name == 'start_tip_angle+amplitude_by_whisker':
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'whisking_indiv_set_point',
            'whisking_indiv_amplitude',
            ])
    
    elif model_name == 'start_tip_angle+global_amplitude':
        # The whisking model is this one with smoothing
        to_dump = my.misc.fetch_columns_with_error_check(
            neural_unbinned_features, [
            'log_cycle_duration',
            'whisking_indiv_set_point',
            'whisking_global',
            ])

    # This one is now just the whisking model
    #~ elif model_name == 'start_tip_angle_smoothed+global_amplitude_smoothed':
        #~ to_dump = my.misc.fetch_columns_with_error_check(
            #~ neural_unbinned_features, [
            #~ 'log_cycle_duration',
            #~ 'whisking_indiv_set_point_smoothed',
            #~ 'whisking_global_smoothed',
            #~ ])

    ## ELSE
    else:
        raise ValueError("unknown model: {}".format(model_name))

    
    ## Error check
    assert not to_dump.isnull().all().any()
    assert 'log_cycle_duration' in to_dump.columns.get_level_values(0)
    
    # Remove all variables other than amplitude from whisking_global_smoothed
    # (except for the full model which is never actually trained anyway)
    if model_name != 'full':
        if 'whisking_global_smoothed' in to_dump.columns.get_level_values(0):
            to_dump = to_dump.drop([
                ('whisking_global_smoothed', 'set_point'), 
                ('whisking_global_smoothed', 'spread'), 
                ('whisking_global_smoothed', 'C1vC2'), 
                ('whisking_global_smoothed', 'C3vC2'),
                ], axis=1).copy()            
    
    
    ## Dump
    # The glm_features dir for this model
    model_features_dir = os.path.join(glm_features_dir, model_name)
    
    if not os.path.exists(model_features_dir):
        os.mkdir(model_features_dir)
    
    # Dump the features
    to_dump.to_pickle(os.path.join(model_features_dir,
        'neural_unbinned_features'))
    

## Also dump spikes_by_cycle, now aligned to neural_unbinned_features
spikes_by_cycle.to_pickle(
    os.path.join(params['glm_dir'], 'aligned_spikes_by_cycle'))


## Also dump the whisks that made contact, for scoring
whisks_with_contact = (
    neural_unbinned_features['contact_binarized'].sum(1) > 0)

whisks_with_contact.to_pickle(
    os.path.join(params['glm_dir'], 'whisks_with_contact'))

