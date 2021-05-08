## Exemplar plot of GLM
"""
7A, bottom left	
    GLM_EXAMPLE_TRIAL_180223_KF132-80_145.svg	
    N/A
    Example of GLM working (whisking-responsive)

7A, bottom right
    GLM_EXAMPLE_TRIAL_191013_229CR-74_122.svg
    N/A
    Example of GLM working (contact-responsive)
"""

import json
import os
import pandas
import numpy as np
import my.plot 
import matplotlib.pyplot as plt
import matplotlib


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

my.plot.manuscript_defaults()
my.plot.font_embed()


## Model to use
model_name = 'minimal'


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)
    
    
## Paths
glm_results_dir = os.path.join(params['glm_dir'], 'results')
model_results_dir = os.path.join(glm_results_dir, model_name)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load results from main4b
coef_wscale_df = pandas.read_pickle(os.path.join(
    model_results_dir, 'coef_wscale_df'))
fitting_results_df = pandas.read_pickle(os.path.join(
    model_results_dir, 'fitting_results_df'))

# Normalize likelihood
fitting_results_df['ll_per_whisk'] = (
    (fitting_results_df['likelihood'] - fitting_results_df['null_likelihood']) / 
    fitting_results_df['len_ytest'])
    

## Load trial matrix from neural_patterns_dir
big_tm = pandas.read_pickle(os.path.join(params['neural_dir'], 'neural_big_tm'))


## Load patterns and whisk cycles
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')
big_tip_pos = my.dataload.load_data_from_patterns(
    params, 'big_tip_pos')
big_touching_df = my.dataload.load_data_from_patterns(
    params, 'big_touching_df')


## Load spiking data
big_psth = pandas.read_pickle(os.path.join(params['neural_dir'], 'big_psth'))
big_alfws = pandas.read_pickle(os.path.join(params['neural_dir'], 'big_alfws'))
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'aligned_spikes_by_cycle'))

# Include only neurons in bwid (because I dropped 1 and 6b)
spikes_by_cycle = my.misc.slice_df_by_some_levels(
    spikes_by_cycle, big_waveform_info_df.index)


## Load neural features
neural_features = pandas.read_pickle(os.path.join(
    params['glm_dir'], 'features', model_name, 'neural_unbinned_features'))


## Choose exemplar
PLOT_EXAMPLE_V2 = True

exemplars = [
    ('180223_KF132', 80, 145), # driven by setpoint
    ('191013_229CR', 74, 122), # driven by contacts
    ]


## Iterate over exemplars
for session, neuron, trial in exemplars:

    ## Extract coefficients to reconstitute response
    # Load the results of the fit
    data_matrix = pandas.read_pickle(
        os.path.join(params['glm_dir'], 'models', model_name, session, 'data_matrix'))
    drift_matrix = pandas.read_pickle(
        os.path.join(params['glm_dir'], 'models', model_name, session, 'drift_matrix'))
    fitting_results = pandas.read_pickle(
        os.path.join(params['glm_fits_dir'], model_name, 
        '{}-{}'.format(session, neuron)))

    # Concat the drift onto the data matrix
    data_matrix = pandas.concat(    
        [data_matrix, np.sqrt(drift_matrix.loc[:, neuron]).rename('drift^drift')], 
        axis=1, verify_integrity=True)


    ## Slice features
    session_features = neural_features.loc[session]


    ## This mechanism is for finding example trials
    if trial is None:
        potential_trials = np.unique(
            session_features.index.get_level_values('trial').values)
        trial = potential_trials[trial_offset + len(potential_trials) // 2]


    ## Extract data for this trial
    trial_features = session_features.loc[trial]
    neuron_spikes = spikes_by_cycle.loc[session].xs(neuron, level='neuron')
    trial_spikes_by_cycle = neuron_spikes.loc[trial]
    trial_whisker_angle = big_tip_pos.loc[session].loc[trial].loc['tip'].xs(
        'angle', level='metric').unstack('whisker').drop('C0', axis=1)
    trial_cycle_frames = C2_whisk_cycles.loc[session].loc[trial].loc[
        :, 'peak_frame_wrt_rwin']
    trial_spikes = big_psth.loc[session].loc[trial].xs(neuron, level='neuron')
    trial_spikes_frames_wrt_rwin = big_alfws.loc[session].loc[trial]
    trial_spikes = pandas.Series(
        trial_spikes.values, index=trial_spikes_frames_wrt_rwin.values)

    # Figure out what fold we're on
    N_FOLDS = 3
    rowfold = np.mod(np.floor(
        np.arange(0, len(data_matrix) * N_FOLDS * 4, N_FOLDS * 4) 
        / len(data_matrix)).astype(np.int), N_FOLDS)
    try:
        this_fold = pandas.Series(rowfold, index=data_matrix.index).loc[
            trial].unique().item()
    except ValueError:
        raise #continue

    # Figure out what reg_lambda we used for best (or single)
    n_reg_lambda = fitting_results_df.xs(session, level='session').xs(
        neuron, level='neuron').xs('actual_best', level='analysis').xs(
        this_fold, level='n_fold').index.get_level_values(
        'n_reg_lambda').values.item()

    # Get the coefs for THIS fold (because rowfold defines test_mask)
    neuron_coefs = fitting_results['coef_df'].loc['actual'].loc[this_fold].loc[
        n_reg_lambda].loc[0]

    # Get the intercept for THIS fold
    this_icpt = fitting_results['fitting_results'].loc['actual'].loc[this_fold].loc[
        n_reg_lambda].loc[0].loc['icpt']

    # Predict
    scaled_dm = data_matrix.sub(
        fitting_results['input_mean']).divide(
        fitting_results['input_scale']).mul(
        neuron_coefs)
    decfun = data_matrix.sub(
        fitting_results['input_mean']).divide(
        fitting_results['input_scale']).mul(
        neuron_coefs).sum(1) + this_icpt
    pred_response = np.exp(decfun)

    # Slice for this example trial
    trial_pred_response = pred_response.loc[trial]


    ## Convert spike rates to Hz and then evenly sample
    cycle_durations = (10 ** trial_features[('log_cycle_duration', 'all')]) / 200.
    trial_pred_response_Hz = trial_pred_response / cycle_durations 
    trial_pred_response_Hz_vs_t = pandas.Series(
        trial_pred_response_Hz.values, 
        index=trial_cycle_frames.loc[trial_pred_response_Hz.index].values)
    trial_pred_response_Hz_vs_t_smoothed = trial_pred_response_Hz_vs_t.reindex(
        trial_whisker_angle.index).interpolate(
        limit_direction='both', method='polynomial', order=1)

    # Same with actual spike counts
    trial_spikes_by_cycle_Hz = trial_spikes_by_cycle / cycle_durations
    trial_spikes_by_cycle_Hz_vs_t = pandas.Series(
        trial_spikes_by_cycle_Hz.values, 
        index=trial_cycle_frames.loc[trial_spikes_by_cycle_Hz.index].values,
        )
    trial_spikes_by_cycle_Hz_vs_t_smoothed = trial_spikes_by_cycle_Hz_vs_t.reindex(
        trial_whisker_angle.index).interpolate(
        limit_direction='both', method='polynomial', order=1)


    ## Plot
    whisker2color = {'C1': 'b', 'C2': 'g', 'C3': 'r'}
    whisker_yspacing = 70
    whisker_l = ['C1', 'C2', 'C3']

    # Make handles
    # axes: 0) whisker traces; 1) contacts/touching; 2) rasters; 3) predictions
    f, axa = plt.subplots(
        4, 1, sharex=True, figsize=(2.7, 3.2),
        gridspec_kw={'height_ratios': [1.2, 0.4, 0.2, 1]})
    f.subplots_adjust(hspace=.1, bottom=.175, top=1, left=.22, right=.9)

    
    ## Plot whisker angle
    ax = axa[0]
    for n_whisker, whisker in enumerate(whisker_l):
        # Demean
        topl = trial_whisker_angle[whisker].copy()
        topl = topl - topl.mean()
        topl_xval = topl.index.values / 200.
        
        # Plot the individual whisker
        color = whisker2color[whisker]
        DS_RATIO = 2
        ax.plot(
            topl_xval[::DS_RATIO],
            topl[::DS_RATIO].values + n_whisker * whisker_yspacing,
            color=color, lw=1)

        # Label
        ax.text(
            1.03, n_whisker * whisker_yspacing, whisker, 
            color=color, ha='left', va='center', size=12, clip_on=False)

    # Pretty
    axa[0].axis('off')
    axa[0].set_xlim((-2, 1))
    
    # Scale bars
    t_start = -2.2
    y_start = 45
    y_len = 60
    axa[0].plot(
        [t_start, t_start], [y_start, y_start + y_len], 
        'k-', lw=.8, clip_on=False)
    axa[0].text(
        t_start - .01, np.mean([y_start, y_start + y_len]) + y_len / 10, 
        '{}{}'.format(y_len, chr(176)), 
        ha='right', va='center', rotation=90, size=12)
    
    
    ## Plot the touching
    ax = axa[1]
    rect_half_height = .4
    for n_whisker, whisker in enumerate(whisker_l):
        # Get touching
        whisker_touching = big_touching_df.loc[
            session].loc[trial].loc[whisker]
        
        # Get each contiguous segment of ones
        starts = np.where(np.diff(whisker_touching) == 1)[0]
        stops = np.where(np.diff(whisker_touching) == -1)[0]
        assert len(starts) == len(stops)
        assert (starts < stops).all()
        
        # Plot
        color = whisker2color[whisker]
        yval = n_whisker
        for start, stop in zip(starts, stops):
            # If this is half-open, check that the plotting works
            if stop - start == 0:
                1/0
            
            # Plot touching bar
            # Use rectangle to avoid disappearing
            t_start = trial_whisker_angle.index.values[start] / 200.
            t_stop = trial_whisker_angle.index.values[stop] / 200.
            rect = matplotlib.patches.Rectangle(
                (t_start, yval - rect_half_height), 
                t_stop - t_start, 2 * rect_half_height, 
                color=color)
            ax.add_patch(rect)
    
    # Pretty
    axa[1].set_ylim((-rect_half_height - 1, 2 + rect_half_height))
    axa[1].axis('off')


    ## Plot rasters
    # Clip rasters by display times
    topl_trial_spikes = trial_spikes.loc[
        (trial_spikes.index > -400) &
        (trial_spikes.index < 100)
        ]

    # Plot rasters
    trial_spike_times = topl_trial_spikes[topl_trial_spikes > 0].index.values
    axa[2].plot(
        trial_spike_times / 200., 
        np.zeros_like(trial_spike_times), 
        marker='|', color='k', ls='none', clip_on=False, ms=16)

    # Pretty
    axa[2].axis('off')
    axa[2].set_ylim((-1, 0.5))
    
    
    ## Plot predictions
    axa[3].plot(
        trial_pred_response_Hz_vs_t_smoothed.index / 200.,
        trial_pred_response_Hz_vs_t_smoothed.values,
        color='magenta', clip_on=True)
    axa[3].set_xlabel('time in trial (s)')
    axa[3].set_ylabel('firing rate (Hz)')
    axa[3].set_ylim((0, 20))
    my.plot.despine(axa[3])
    

    ## Save
    f.savefig(
        'GLM_EXAMPLE_TRIAL_{}-{}_{}.svg'.format(session, neuron, trial))
    f.savefig(
        'GLM_EXAMPLE_TRIAL_{}-{}_{}.png'.format(session, neuron, trial), 
        dpi=300)

    
plt.show()    
    
    