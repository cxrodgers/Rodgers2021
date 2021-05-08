## Parse and reconstitute
# main4b1 unmangles, reconsitutes, and dumps
# Loads the fit files for each model * neuron from params['glm_fits_dir']
# Loads the model from 
#   os.path.join(params['glm'], 'models', model_name, session_name)
# Unmangles and reconstitutes
# Stores results concatenated over session to 
#   os.path.join(params['glm'], 'results', model_name)

import json
import os
import pandas
import numpy as np
import scipy.stats
import my
import whiskvid


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
    #~ 'minimal+random_regressor',

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
# Where the fits from the cluster are
glm_fits_dir = params['glm_fits_dir']

# Inputs to those fits
glm_models_dir = os.path.join(params['glm_dir'], 'models')

# Further analysis results of those fits
glm_results_dir = os.path.join(params['glm_dir'], 'results')
if not os.path.exists(glm_results_dir):
    os.mkdir(glm_results_dir)

    
## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params, drop_1_and_6b=False)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


def unmangle(family_ser, family_level_names):
    """Unmangle indices for a given family.
    
    Takes all the values of the index at the level 'mangled' and splits
    by the ^ character. Tries to intify each. Makes these into a multi-index
    with specified level names.
    
    family_ser : Series with level 'mangled'
    family_level_names : level names to apply after unmangling
    """
    # Iterate over indices
    tuples_l = []
    for mangled_string in family_ser.index.get_level_values('mangled'):
        #~ # Fix this
        #~ if family_level_names == ('drift',):
            #~ mangled_string = mangled_string.split('_')[1]
        
        # Split on ^
        try:
            split = mangled_string.split('^')
        except AttributeError:
            # This happens if it's an int
            split = [mangled_string]
        
        # Try to intify each
        res_l = []
        for val in split:
            # Intify if possible
            try:
                res = int(val)
            except ValueError:
                res = val
            
            # Store
            res_l.append(res)

        # Error check
        assert len(res_l) == len(family_level_names)

        # Store
        tuples_l.append(tuple(res_l))        
    
    # Mask out the appropriate unmangled tuples and convert to MultiIndex
    family_midx = pandas.MultiIndex.from_tuples(tuples_l, 
        names=family_level_names)
    
    # Apply the MultiIndex
    family_ser.index = family_midx.copy()
    
    return family_ser


## Load trial matrix from neural_patterns_dir
big_tm = pandas.read_pickle(os.path.join(params['neural_dir'], 'neural_big_tm'))


## Slice out sessions of interest
session_names = list(big_tm.index.levels[0])


for model_name in model_names:
    print(model_name)
    
    
    ## Iterate over sessions
    keys_l = []
    fitting_results_l = []
    coef_wscale_l = []
    for session_name in session_names:
        ## Where the model is
        session_model_dir = os.path.join(
            glm_models_dir, model_name, session_name)


        ## Which clusters were run
        included_clusters = big_waveform_info_df.loc[session_name].index.values
        
        
        ## Load each result
        for cluster in included_clusters:
            fit_filename = os.path.join(
                glm_fits_dir, model_name, '%s-%d' % (session_name, cluster))

            if os.path.exists(fit_filename):
                ## Load fit
                results = pandas.read_pickle(fit_filename)
                coef_df = results['coef_df'].sort_index()
                fitting_results = results['fitting_results'].sort_index()
                
                
                ## Calculate mean coefficients over folds
                # Mean the best reg over folds
                coef_df_best_reg_mean_over_folds = (    
                    coef_df.loc['actual_best'].mean())
                
                # Mean the single reg over folds
                coef_df_single_reg_mean_over_folds = (    
                    coef_df.loc['actual_single'].mean())

                
                ## Store the various coefficients in coef_wscale
                # Concat coef, scale, mean
                coef_wscale = pandas.concat([
                    coef_df_best_reg_mean_over_folds, 
                    coef_df_single_reg_mean_over_folds,
                    results['input_scale'], 
                    results['input_mean'],
                    ], axis=1,
                    verify_integrity=True, sort=True,
                    keys=['coef_best', 'coef_single', 'scale', 'mean'],
                    names='coef_metric')
                
                # Scale the coef_best and coef_single by the input_scale
                coef_wscale['scaled_coef_best'] = (
                    coef_wscale['coef_best'].divide(coef_wscale['scale']))
                coef_wscale['scaled_coef_single'] = (
                    coef_wscale['coef_single'].divide(coef_wscale['scale']))
                

                ## Identify which permutations failed to converge on any fold
                permutations_converged = fitting_results.loc[
                    'permuted_single', 'converged'].droplevel(
                    'n_reg_lambda').unstack('n_fold')
                keep_permutations = permutations_converged.all(1)
                
                # If this happens a lot then raise n_iter
                if not keep_permutations.all():
                    print("warning: failed to converge in {} permutations".format(
                        (~keep_permutations).sum()))

                
                ## Calculate the spread of the permutations
                ## Use only permuted_single here
                ## Mean over folds
                # Mean over folds, maintaining each permute
                coef_df_permuted_mean_over_folds = coef_df.loc[
                    'permuted_single'].droplevel(
                    'n_reg_lambda').mean(level='n_permute')

                # Store in coef_wscale
                # Default ddof=1 seems appropriate here
                coef_wscale['perm_single_std'] = (
                    coef_df_permuted_mean_over_folds.std())
                coef_wscale['perm_single_mean'] = (
                    coef_df_permuted_mean_over_folds.mean())

                # Z-score the actual coefficients by the permutation scale
                coef_wscale['coef_single_z'] = ((
                    coef_wscale['coef_single'] - coef_wscale['perm_single_mean']) / 
                    coef_wscale['perm_single_std']
                )                
                
                # Convert to p-value
                coef_wscale['coef_single_p'] = my.stats.z2p(
                    coef_wscale['coef_single_z'])
                
                
                ## Unmangle (in place)
                coef_wscale.index.name = 'mangled'
                unmangle(coef_wscale, ['metric', 'label'])

                # Store
                keys_l.append((session_name, cluster))
                fitting_results_l.append(fitting_results)
                coef_wscale_l.append(coef_wscale)

            else:
                print("cannot find: %s" % results_filename)
                1/0


    ## Concat
    # Fitting results (e.g., scores, convergence, etc)
    fitting_results_df = pandas.concat(
        fitting_results_l, keys=keys_l, names=['session', 'neuron'])

    # The coef, mean, and scale are different for every neuron due to the 
    # population and drift terms, so concat along axis 0
    coef_wscale_df = pandas.concat(coef_wscale_l, keys=keys_l, axis=0, 
        verify_integrity=True, names=['session', 'neuron'])


    ## Add task level
    coef_wscale_df = my.misc.insert_level(
        my.misc.insert_mouse_level(coef_wscale_df), 
        name='task', func=lambda idx: idx['mouse'].map(mouse2task)).droplevel(
        'mouse')

    fitting_results_df = my.misc.insert_level(
        my.misc.insert_mouse_level(fitting_results_df), 
        name='task', func=lambda idx: idx['mouse'].map(mouse2task)).droplevel(
        'mouse')

    
    ## Dump
    # Make directory
    results_model_dir = os.path.join(glm_results_dir, model_name)
    if not os.path.exists(results_model_dir):
        os.mkdir(results_model_dir)

    # Dump
    coef_wscale_df.to_pickle(os.path.join(
        results_model_dir, 'coef_wscale_df'))
    fitting_results_df.to_pickle(os.path.join(
        results_model_dir, 'fitting_results_df'))
