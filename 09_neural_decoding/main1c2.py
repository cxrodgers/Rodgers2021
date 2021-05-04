## Decode stimulus and choice from neural activity, 
## excluding trials with early licks


import json
import os
import pandas
import numpy as np
import my, my.decoders
import tqdm
import extras


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Where to store results
neural_decoding_dir = os.path.join(params['neural_dir'], 'decoding')
if not os.path.exists(neural_decoding_dir):
    os.mkdir(neural_decoding_dir)

 
## Load session metadata
session_df, task2mouse, mouse2task = (
    my.dataload.load_session_metadata(params))


## Load data
# Neural trial matrix
big_tm = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'neural_big_tm'))

# Patterns
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')

# Logreg
uuf = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features')


## Load spiking data
# Load big_waveform_info_df
big_waveform_info_df = my.dataload.load_bwid(params)

# Load spikes_by_cycle
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'spikes_by_cycle'))

# Discard neurons not in big_waveform_info_df (L1 and L6b)
spikes_by_cycle = my.misc.slice_df_by_some_levels(
    spikes_by_cycle, big_waveform_info_df.index)


## Include only no-lick trials
# Load the no licks dataset
datasets_dir = os.path.join(params['logreg_dir'], 'datasets')
no_licks_tm = pandas.read_pickle(
    os.path.join(datasets_dir, 'no_opto_no_licks1', 'features'))

# Slice these trials from big_tm
joint_index = no_licks_tm.index.intersection(big_tm.index)
big_tm = big_tm.loc[joint_index].sort_index().copy()


## Slice out discrimination only
# Add mouse and task level to big_tm
big_tm = my.misc.insert_mouse_and_task_levels(big_tm, mouse2task=mouse2task)

# Slice out discrimination, and drop mouse level
big_tm = big_tm.loc['discrimination'].droplevel('mouse')
big_tm.index = big_tm.index.remove_unused_levels()

# Slice spikes in the same way
#~ spikes_by_cycle = spikes_by_cycle.loc[big_tm.index.levels[0]] # only does session
spikes_by_cycle = my.misc.slice_df_by_some_levels(spikes_by_cycle, big_tm.index) # session and trial
spikes_by_cycle.index = spikes_by_cycle.index.remove_unused_levels()


## Drop sessions with insufficient error trials
# Count trials of each class
n_trials_per_class_by_session = big_tm.groupby(
    ['session', 'rewside', 'choice']).size().unstack(['rewside', 'choice'])

# Thereshold
MIN_TRIALS_PER_CLASS = 10
include_session_mask = (
    n_trials_per_class_by_session.min(1) >= MIN_TRIALS_PER_CLASS)

# Apply mask
big_tm = big_tm.loc[
    include_session_mask.index[include_session_mask.values]]
big_tm.index = big_tm.index.remove_unused_levels()

spikes_by_cycle = spikes_by_cycle.loc[
    include_session_mask.index[include_session_mask.values]]
spikes_by_cycle.index = spikes_by_cycle.index.remove_unused_levels()


## Ensure consistent session * trial
# Check that same session * trial are in big_tm and spikes_by_cycle
st1 = spikes_by_cycle.index.to_frame()[
    ['session', 'trial']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial']).reset_index(drop=True)
st2 = big_tm.index.to_frame().reset_index(drop=True).sort_values(
    ['session', 'trial']).reset_index(drop=True)
assert st1.equals(st2)


## Bin the cycles
# Locking times
locking_frames = C2_whisk_cycles['peak_frame_wrt_rwin'].copy()

# Bins
n_bins = 20
bins = np.rint(np.linspace(-400, 200, n_bins + 1)).astype(np.int)

# Bin each cycle
locking_frame_bins = pandas.cut(
    locking_frames, bins=bins, right=False, labels=False)

# Drop bins outside the range
locking_frame_bins = locking_frame_bins.dropna().astype(np.int).rename('bin')


## Join bin on spikes
# Inner-join, so only spikes within the temporally binned range are included
spikes_by_cycle_with_bin = spikes_by_cycle.to_frame().join(
    locking_frame_bins, on=['session', 'trial', 'cycle'], how='inner')


## Define no-frontier-crossings mask
# Identify contact cycles
contact_cycles_mask = (uuf['contact_binarized'].sum(1) > 0)
anti_contact_cycles_mask = (uuf['anti_contact_count'].sum(1) > 0)
cycles_crossing_frontier_mask = contact_cycles_mask | anti_contact_cycles_mask

# Shifts
shifted_1 = cycles_crossing_frontier_mask.shift(-1).fillna(False)
shifted1 = cycles_crossing_frontier_mask.shift(1).fillna(False)
shifted2 = cycles_crossing_frontier_mask.shift(2).fillna(False)

# Mask
nfc_mask = ~(cycles_crossing_frontier_mask | shifted_1 | shifted1 | shifted2)


## Define experiments
experiments_l = [
    {'include': 'all', 'group_by_string': 'rewside',},
    {'include': 'all', 'group_by_string': 'choice',},
    {'include': 'all', 'group_by_string': 'both',},
    {'include': 'no_frontier_crossings', 'group_by_string': 'both',},
    ]

# How to fold the data
n_splits = 5
shuffle = False

# How many times to repeat the resampling
N_BOOTS = 100

# How many trials go into the test set on each boot
resampling_factor = 30

# Size of the training set makes the size of each split equal
setname2n_resamples_per_split_and_strat = {
    'test': resampling_factor, 
    'train': resampling_factor * (n_splits - 1),
    }


## Iterate over experiments
for dexp in tqdm.tqdm(experiments_l):

    ## Random seed
    np.random.seed(0)

    
    ## Extract data for this experiment
    include = dexp['include']
    group_by_string = dexp['group_by_string']

    # How to stratify the data
    # It has to be stratified by decode_labels, to ensure that spikes are taken
    # from trials with the same decode_label
    if group_by_string == 'both':
        group_by = ['rewside', 'choice']
    else:
        group_by = [group_by_string]
    
    # Apply `include`
    if include == 'all':
        # Include everything
        include_spikes_by_cycle_with_bin = spikes_by_cycle_with_bin

    elif include == 'no_frontier_crossings':
        # Include only spikes that are on those cycles
        include_spikes_by_cycle_with_bin = my.misc.slice_df_by_some_levels(
            spikes_by_cycle_with_bin, 
            nfc_mask.index[nfc_mask.values],
            )

    else:
        1/0


    ## Sum over cycles within bin
    include_spikes_by_bin = include_spikes_by_cycle_with_bin.set_index(
        'bin', append=True)['spikes'].reorder_levels(
        ['session', 'trial', 'cycle', 'bin', 'neuron']).sort_index()
    include_spikes_by_bin = include_spikes_by_bin.sum(
        level=['session', 'trial', 'bin', 'neuron'])
    
    
    ## Reindex
    # There could be missing bins, if there were no cycles in the bin
    # (190515_221CR trial 270 bin 0)
    # or if all cycles in a bin got dropped for frontier crossings.
    
    # Build a full MultiIndex and insert zeros where no data
    # This gets all session * trial * neuron
    stn_df = spikes_by_cycle_with_bin.reset_index()[
        ['session', 'trial', 'neuron']].drop_duplicates() 
    
    # Cross with bins
    stn_df['key'] = 0
    to_merge = pandas.DataFrame.from_dict({
        'bin': np.arange(n_bins, dtype=np.int), 
        'key': np.zeros(n_bins, dtype=np.int)
        })
    merged = pandas.merge(stn_df, to_merge, on='key').drop('key', 1)
    
    # Convert to MultiIndex, matching order of include_spikes_by_bin
    full_midx = pandas.MultiIndex.from_frame(
        merged[['session', 'trial', 'bin', 'neuron']])

    # Reindex and impute zero
    include_spikes_by_bin = (
        include_spikes_by_bin.reindex(full_midx).sort_index())
    include_spikes_by_bin = include_spikes_by_bin.fillna(0).astype(np.int)


    ## Split the data into folds
    # First stratify the data so that we can include equal amounts from each class
    # in each fold
    strats = my.decoders.intify_classes(big_tm[group_by], by=tuple(group_by))

    # Define the folds
    folds = extras.define_folds(
        strats, n_splits=n_splits, shuffle=shuffle, n_tune_splits=0)
    

    ## Iterate over bootstraps
    mean_accuracy_l = []
    meaned_weights_l = []
    meaned_intercepts_l = []
    for n_boot in tqdm.tqdm(range(N_BOOTS)):
        ## Resample the data
        # I think this is the bottleneck
        resampled_trials, resampled_spikes = extras.resample_from_data_with_splits(
            big_tm=big_tm,
            spikes_by_bin=include_spikes_by_bin,
            folds=folds,
            setname2n_resamples_per_split=setname2n_resamples_per_split_and_strat,
            group_by=group_by,
            )

        # Transpose both for below
        # sessions and neurons should be on the columns
        # split, set, resample should be on the index
        resampled_trials = resampled_trials.T
        resampled_spikes = resampled_spikes.T

        # This can be used to inspect how frequently each trial is included
        n_uses_by_trial = resampled_trials.stack().rename(
            'trial').reset_index().groupby(['session', 'trial']).size().sort_values()


        ## Prepare input features
        # Normalize each session * bin *neuron
        features = np.sqrt(resampled_spikes)
        norm_features, normalizing_mu, normalizing_sigma = (
            my.decoders.normalize_features(features))


        ## Decoding params
        #~ reg_l = np.linspace(-6, 6, 13)
        reg_l = np.array([0.])
        to_optimize = 'pred_correct'

        # Decode from each bin separately
        bin_l = sorted(np.unique(norm_features.columns.get_level_values('bin')))


        ## Iterate over splits
        tuning_results_l = []
        tuning_keys_l = []
        for n_split in range(n_splits):
            ## Iterate over temporal bins
            for bin in bin_l:
                ## Iterate over decode_label
                decode_label_l = group_by
                for decode_label in decode_label_l:
                    ## Slice
                    # Slice the data to decode
                    data_to_decode = norm_features.loc[n_split].xs(
                        bin, level='bin', axis=1)
                    
                    # Slice the trials similarly
                    trials_to_decode = resampled_trials.loc[n_split]
                    
                    
                    ## Determine the target labels and the train/test set
                    # Can take from either data_to_decode or trials_to_decode
                    # since they have the same index
                    idxdf = trials_to_decode.index.to_frame().reset_index(drop=True)
                    
                    # This is the target
                    target_label = idxdf[decode_label]
                    intified_target_label = (target_label == 'right').astype(np.int)
                    
                    # This is the train and test indices
                    train_indices = np.where(idxdf['set'] == 'train')[0]
                    test_indices = np.where(idxdf['set'] == 'test')[0]
                    

                    ## Iterate over reg
                    for reg in reg_l:
                        # Run the decoder
                        logreg_res = my.decoders.logregress2(
                            features=data_to_decode.values,
                            labels=intified_target_label,
                            train_indices=train_indices,
                            test_indices=test_indices,
                            sample_weights=np.ones(len(intified_target_label)),
                            regularization=10 ** reg,
                            )

                        # Error check
                        assert (
                            logreg_res['per_row_df']['set'] == idxdf['set']).all()

                        # Add the index back
                        logreg_res['per_row_df'].index = data_to_decode.index
                        logreg_res['weights'] = pandas.Series(logreg_res['weights'],
                            index=data_to_decode.columns, name='weights')
                        
                        # Store
                        tuning_results_l.append(logreg_res)
                        tuning_keys_l.append((decode_label, n_split, bin, reg))        


        ## Concat the results
        tuning_scores_l = [tuning_result['scores_df'].loc['test', :]
            for tuning_result in tuning_results_l]
        tuning_scores = pandas.concat(tuning_scores_l, keys=tuning_keys_l, 
            axis=1, names=['decode_label', 'split', 'bin', 'reg']).T    

            
        ## Choose the best regularization
        tuning_scores = tuning_scores.reset_index()

        # For each split, choose the reg that jointly optimizes over all params
        # except reg and split
        scores_by_reg_and_split = tuning_scores.groupby(['reg', 'split']
            )[to_optimize].mean().mean(level=['reg', 'split']).unstack('split')

        # Choose best reg
        best_reg_by_split = scores_by_reg_and_split.idxmax()

        # Force a best reg
        best_reg_by_split = best_reg_by_split * 0 + reg_l[len(reg_l) // 2]


        ## Now extract just the best reg from each split
        best_per_row_results_by_split_l = []
        best_weights_by_split_l = []
        best_intercepts_by_split_l = []
        best_keys_l = []

        # Iterate over all results and keep just the ones corresponding to 
        # the best_reg for that split
        for n_key, key in enumerate(tuning_keys_l):
            # Split the key
            decode_label, split, bin, reg = key
            
            # Check whether this result is with the best_reg on this split
            if reg == best_reg_by_split.loc[split]:
                # Extract those results
                best_split_results = tuning_results_l[n_key]
                
                # Store
                best_per_row_results_by_split_l.append(best_split_results['per_row_df'])
                best_weights_by_split_l.append(best_split_results['weights'])
                best_intercepts_by_split_l.append(best_split_results['intercept'])
                best_keys_l.append((decode_label, split, bin))

        # Concat over session, decode_label, split
        best_per_row_results_by_split = pandas.concat(
            best_per_row_results_by_split_l,
            keys=best_keys_l, names=['decode_label', 'split', 'bin']).sort_index()

        best_weights_by_split = pandas.concat(best_weights_by_split_l, 
            keys=best_keys_l, axis=1, names=['decode_label', 'split', 'bin']
            ).sort_index(axis=1)

        best_intercepts_by_split = pandas.Series(best_intercepts_by_split_l,
            index=pandas.MultiIndex.from_tuples(best_keys_l, 
            names=['decode_label', 'split', 'bin'])).sort_index()

        # Drop set from the MultiIndex because it's always put in the columns anyway
        best_per_row_results_by_split = best_per_row_results_by_split.droplevel('set')


        ## Finalize predictions by taking only the ones on the test set
        # This is a bit different from the usual case because "resample" has no 
        # meaning across sets, splits, or groupbys
        # But we still want to only keep the ones in the test sets
        finalized_predictions = best_per_row_results_by_split.loc[
            best_per_row_results_by_split['set'] == 'test'].sort_index()

        # Mean the weights and intercepts over splits
        meaned_weights = best_weights_by_split.mean(
            level=[lev for lev in best_weights_by_split.columns.names 
            if lev != 'split'], axis=1)
        meaned_intercepts = best_intercepts_by_split.mean(
            level=[lev for lev in best_intercepts_by_split.index.names
            if lev != 'split'])
        
        
        ## Mean performance
        mean_accuracy = finalized_predictions['pred_correct'].mean(
            level=['decode_label', 'bin'])
        
        
        ## Store
        mean_accuracy_l.append(mean_accuracy)
        meaned_weights_l.append(meaned_weights)
        meaned_intercepts_l.append(meaned_intercepts)

    
    ## Concat over boots
    # Index of each is boot * decode_label * bin
    boot_index = pandas.Index(range(N_BOOTS), name='boot')
    
    mean_accuracy_df = pandas.concat(
        mean_accuracy_l, axis=0, keys=boot_index)

    mean_weights_df = pandas.concat(
        meaned_weights_l, axis=1, keys=boot_index).T

    mean_intercepts_df = pandas.concat(
        meaned_intercepts_l, axis=0, keys=boot_index)


    ## Dump
    # Form directory
    this_neural_decoding_dir = os.path.join(neural_decoding_dir, 
        '{}-{}-{}'.format('no_licks1', include, group_by_string))
    if not os.path.exists(this_neural_decoding_dir):
        os.mkdir(this_neural_decoding_dir)
    
    # Dump
    my.misc.pickle_dump(bins, os.path.join(this_neural_decoding_dir, 'bins'))

    mean_accuracy_df.to_pickle(
        os.path.join(this_neural_decoding_dir, 'meaned_accuracy'))

    mean_weights_df.to_pickle(
        os.path.join(this_neural_decoding_dir, 'meaned_weights'))

    mean_intercepts_df.to_pickle(
        os.path.join(this_neural_decoding_dir, 'meaned_intercepts'))
