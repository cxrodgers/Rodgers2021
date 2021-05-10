## Neural encoding model
## main2* : Deal with spikes
# This one forms and dumps data_matrix, response_matrix, and drift_matrix

import json
import os
import MCwatch.behavior
import whiskvid
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
session_mangled_features_dir = os.path.join(
    params['glm_dir'], 'session_mangled_features')

# Original spikes location for computing drift
raw_spikes_dir = os.path.expanduser('~/mnt/nas2_home/whisker/processed')

# Create glm_models_dir if it doesn't exist
glm_models_dir = os.path.join(params['glm_dir'], 'models')
my.misc.create_dir_if_does_not_exist(glm_models_dir)


## Load trial matrix from neural_patterns_dir
big_tm = pandas.read_pickle(os.path.join(params['neural_dir'], 'neural_big_tm'))


## Load patterns and whisk cycles
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))


## Load spiking data
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'aligned_spikes_by_cycle'))
    
    
## Iterate over sessions
session_names = spikes_by_cycle.index.levels[0]
for session_name in session_names:
    print(session_name)


    ## Get session objects
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    

    ## Load spikes (for calculating drift)
    # Error check these are the same, then remove this raw_spikes_dir
    spikes = pandas.read_pickle(
        os.path.join(raw_spikes_dir, session_name, 'spikes'))

    spikes2 = pandas.read_pickle(
        os.path.join(vs.session_path, 'spikes'))
    assert (spikes == spikes2).all().all()
    

    ## Trial matrix (for syncing drift times to frames)
    trial_matrix = big_tm.loc[session_name]
    
    
    ## Slice cycles for this session (for syncing drift times to frames)
    session_wc = C2_whisk_cycles.loc[session_name]
    
    
    ## Slice response
    this_response = spikes_by_cycle.loc[session_name].copy()
    this_response.index = this_response.index.remove_unused_levels()
    included_clusters = np.unique(
        this_response.index.get_level_values('neuron'))

    # Put neurons on columns
    response_matrix = this_response.unstack('neuron')
    

    ## Iterate over models
    for model_name in model_names:
        ## Where to put results
        model_dir = os.path.join(glm_models_dir, model_name)
        session_model_dir = os.path.join(model_dir, session_name)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        if not os.path.exists(session_model_dir):
            os.mkdir(session_model_dir)
            

        ## Load the data matrix so we can index it the same way
        data_matrix = pandas.read_pickle(os.path.join(
            session_mangled_features_dir, model_name, session_name))


        ## Calculate drift in 10 equal bins over session
        # Select the bins
        drift_bins = np.linspace(
            spikes['time'].min() - .001, 
            spikes['time'].max() + .001, 
            11
        )
        drift_bin_centers = (drift_bins[:-1] + drift_bins[1:]) / 2
        drift_bin_width = (np.diff(drift_bins)).mean()
        
        # Histogram the total spikes for each cluster within each drift bin
        drift_counts_l = []
        for cluster in included_clusters:
            # Get spike times for this cluster
            cluster_times = spikes.loc[spikes['cluster'] == cluster, 'time']
            
            drift_counts, drift_edges = np.histogram(cluster_times, bins=drift_bins)
            drift_counts_l.append(drift_counts)
        
        # DataFrame it
        drift_df = pandas.DataFrame(np.transpose(drift_counts_l), 
            index=pandas.Series(drift_bin_centers, name='time'),
            columns=pandas.Series(included_clusters, name='neuron'),
        )
        
        # Convert to Hz
        drift_df = drift_df / drift_bin_width

        # Extract drift values at the cycle times
        # Get the time of each frame to extract in the neural timebase
        extraction_times = (
            session_wc.index.get_level_values('trial').map(
            trial_matrix['rwin_time_nbase']) + 
            session_wc['peak_frame_wrt_rwin'] / 200.0
        ).rename('time').dropna()
        
        # Only do this for the trials * cycles in the data_matrix
        extraction_times = extraction_times.loc[data_matrix.index].copy()
        assert not extraction_times.isnull().any().any()

        # Linearly interpolate the drift values at the extraction times
        drift_matrix = drift_df.reindex(
            drift_df.index.union(extraction_times.values)
            ).sort_index().interpolate(
            limit_direction='both', method='linear').loc[
            extraction_times.values]
        drift_matrix.index = extraction_times.index
        assert not drift_matrix.isnull().any().any()
        
        
        ## response_matrix, data_matrix, and drift_matrix should be aligned
        my.misc.assert_index_equal_on_levels(
            data_matrix, response_matrix, ['trial', 'cycle'])
        my.misc.assert_index_equal_on_levels(
            data_matrix, drift_matrix, ['trial', 'cycle'])

        
        ## Dump
        data_matrix.to_pickle(
            os.path.join(session_model_dir, 'data_matrix'))
        response_matrix.to_pickle(
            os.path.join(session_model_dir, 'response_matrix'))
        drift_matrix.to_pickle(
            os.path.join(session_model_dir, 'drift_matrix'))
