## Plot PSTHs to licks and contacts, plus example rasters
"""
5E, bottom
    PLOT_PSTHS_TO_LICK_AND_CONTACT_DISC_ONLY
    N/A
    PSTHs to licks and contacts by whisker

5E, top
    PLOT_EXAMPLE_RASTERS_TO_LICK_AND_CONTACT
    N/A
    Example neuron rasters to licks and contact by whiskers

"""

import json
import os
import tqdm
import numpy as np
import pandas
import matplotlib.pyplot as plt
import kkpandas
import whiskvid
import ArduFSM
import my
import my.plot
import my.dataload


## Plots
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load data
big_ccs_df = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_ccs_df'))

FR_overall = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'FR_overall'))

session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load trial matrix that went into the spikes
big_tm = pandas.read_pickle(os.path.join(params['neural_dir'], 'neural_big_tm'))

# Sessions
session_name_l = sorted(big_tm.index.levels[0])


## Iterate over sessions
dfolded = {}
binned_l = []
binned_keys_l = []
orig_times_l = []
orig_times_raw_l = []

for session_name in tqdm.tqdm(session_name_l):
    
    ## Get the video session
    vs = whiskvid.django_db.VideoSession.from_name(session_name)

    
    ## Load spikes
    spikes = pandas.read_pickle(
        os.path.join(vs.session_path, 'spikes'))
    included_clusters = np.sort(spikes['cluster'].unique())
    
    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = big_tm.loc[session_name]
    
    # Also get the pldf and use that to get lick times
    ldf = ArduFSM.TrialSpeak.read_logfile_into_df(
        os.path.join(vs.session_path, 'behavioral_logfile'))
    
    # Get the lick times
    lick_times = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(ldf, 'TCH')
    
    # Keep only lick types 1 and 2
    lick_times = lick_times[lick_times.arg0.isin([1, 2])].copy()
    
    # Convert times from s to ms (in behavioral timebase)
    lick_times['time'] = lick_times['time'] / 1000.0

    
    ## Convert lick times to neural timebase
    # Identify trial label of each lick
    # -1 means before the first trial
    # These values will be really high for dropped trials, but we'll drop
    # those licks below anyway
    lick_times['trial_idx'] = np.searchsorted(
        trial_matrix['start_time'].values, 
        lick_times['time']) - 1

    # Index the ones before the first trial by the first trial
    lick_times.loc[lick_times['trial_idx'] == -1, 'trial_idx'] = 0

    # Convert to trial number
    lick_times['trial'] = trial_matrix.index[lick_times['trial_idx'].values].values
    lick_times = lick_times.drop('trial_idx', 1)    

    # Re-index each lick time to the start of the trial
    lick_times['t_wrt_start'] = (lick_times['time'] - 
        lick_times['trial'].map(trial_matrix['start_time']))
    
    # Add the neural start time of each trial to t_wrt_start
    lick_times['ntime'] = lick_times['t_wrt_start'] + lick_times['trial'].map(
        trial_matrix['start_time_nbase'])

    # Resort by 'ntime', because this procedure can cause slight realignment
    lick_times = lick_times.sort_values('ntime')   
    
    
    #~ ## Debounce licking
    #~ if (np.diff(lick_times['ntime']) < .05).any():
        #~ 1/0


    ## Clean up licking
    # First store the original values for debugging
    orig_times_raw = lick_times['ntime'].copy()
    orig_times = lick_times['t_wrt_start'].copy()
    orig_times_raw_l.append(orig_times_raw)
    orig_times_l.append(orig_times)

    # Drop lick times that are refractory
    # There's a huge peak near 0, followed by a dip around .08, and then
    # a "real" peak around 150ms
    lick_times['diff'] = lick_times['ntime'].diff()
    lick_times = lick_times.loc[
        lick_times['diff'].isnull() | 
        (lick_times['diff'] > .08)
        ].copy()

    # Include only licks within (0, 10) s after start_time
    # This includes the peak
    # Later peaks could correspond to licking during timeout, but they
    # could also be dropped trials
    # And as it gets farther from start time, errors accumulate
    lick_times = lick_times.loc[
        (lick_times['t_wrt_start'] >= 0) &
        (lick_times['t_wrt_start'] < 10)
        ].copy()

    # Error check that spike range contains lick range
    assert lick_times['ntime'].min() > spikes['time'].min()
    assert lick_times['ntime'].max() < spikes['time'].max()
    
    
    ## Process contacts
    # Data from this session
    session_ccs = big_ccs_df.loc[session_name].copy()
    
    # Join stepper_pos
    session_ccs['stepper_pos'] = session_ccs['trial'].map(trial_matrix['stepper_pos'])
    
    # Already locked to rwin
    # Convert to time in neural base
    session_ccs['ntime'] = session_ccs['locked_t'] + session_ccs['trial'].map(
        trial_matrix['rwin_time_nbase'])
    
    # Error check we're only analyzing contacts near triggers
    # In 180125_KF119 he can occasionally touch with C0 during rotation
    assert (
        (session_ccs['locked_t'] >= -2.6) &
        (session_ccs['locked_t'] < 2)
        ).all()

    
    ## Lock spikes on lick times
    lick_bins = np.linspace(-.1, .1, 41)
    for neuron in included_clusters:
        # Slice spikes
        neuron_spikes = spikes[spikes['cluster'] == neuron]
        
        for lick_type, licks in lick_times.groupby('arg0'):
            # Fold
            # We've already checked that spike range includes lick range
            folded = kkpandas.Folded.from_flat(
                neuron_spikes['time'].values,
                centers=licks['ntime'].values,
                dstart=lick_bins[0], 
                dstop=lick_bins[-1],
            )

            # Bin by trial
            binned = kkpandas.Binned.from_folded(folded, bins=lick_bins)
            
            # Store
            dfolded[
                (session_name, 'lick', 'tongue', lick_type, neuron)
                ] = folded            
            binned_l.append(binned.rate_in('Hz')[0])
            binned_keys_l.append(
                (session_name, 'lick', 'tongue', lick_type, neuron))


    ## Lock spikes on contact times
    contact_bins = np.linspace(-.1, .1, 41)
    for neuron in included_clusters:
        # Slice spikes
        neuron_spikes = spikes[spikes['cluster'] == neuron]
        
        for contact_type, sub_contacts in session_ccs.groupby(
                ['whisker', 'stepper_pos']): 
            whisker, stepper_pos = contact_type
            label = '{}-{}'.format(whisker, stepper_pos)
            
            # Fold
            # We've already checked that spike range includes lick range
            folded = kkpandas.Folded.from_flat(
                neuron_spikes['time'].values,
                centers=sub_contacts['ntime'].values,
                dstart=contact_bins[0], 
                dstop=contact_bins[-1],
            )

            # Bin by trial
            binned = kkpandas.Binned.from_folded(folded, bins=contact_bins)
            
            # Store
            dfolded[
                (session_name, 'contact', whisker, stepper_pos, neuron)
                ] = folded
            binned_l.append(binned.rate_in('Hz')[0])
            binned_keys_l.append(
                (session_name, 'contact', whisker, stepper_pos, neuron))

# Concat
big_binned = pandas.concat(binned_l, keys=binned_keys_l, 
    names=['session', 'event', 'organ', 'event_type', 'neuron'], axis=1).T


## Process
# Add task
big_binned = my.misc.insert_mouse_and_task_levels(big_binned, mouse2task)

# Normalize FR
orig_big_binned = big_binned.copy()
big_binned = big_binned.divide(FR_overall, axis=0)

# Drop
big_binned = big_binned.droplevel('mouse')

# Drop C0
big_binned = big_binned.drop('C0', level='organ')


## Save dfolded for main6
my.misc.pickle_dump(dfolded, 'dfolded')


## Plot
PLOT_PSTHS_TO_LICK_AND_CONTACT_DISC_ONLY = True
PLOT_EXAMPLE_RASTERS_TO_LICK_AND_CONTACT = True

event2color = {50: 'orange', 150: 'purple', 1: 'purple', 2: 'orange'}
    
if PLOT_PSTHS_TO_LICK_AND_CONTACT_DISC_ONLY:
    event_l = ['contact', 'lick']
    organ_l = ['tongue', 'blank', 'C1', 'C2', 'C3']    
    task = 'discrimination'

    # events on rows, organs on columns
    f, axa = plt.subplots(
        1, len(organ_l), 
        sharex=True, sharey=True, figsize=(7.5, 2.2),
        gridspec_kw={'width_ratios': [1, .001, 1, 1, 1]})
    #~ f.suptitle('{} {}'.format(session, neuron))
    f.subplots_adjust(left=.075, right=.95, bottom=.3, top=.85, hspace=.2, wspace=.6)

    for event in event_l:
        # Slice data
        event_binned = big_binned.xs(task, level='task').xs(event, level='event')
        event_binned.index = event_binned.index.remove_unused_levels()
        
        # How many event types to plot
        event_type_l = sorted(event_binned.index.get_level_values('event_type').unique())
        
        if event == 'contact':
            event_type_l = event_type_l[::-1]
        
        # Plot each
        for event_type in event_type_l:
            color = event2color[event_type]
            
            for organ in organ_l:
                # Get ax
                ax = axa[
                    organ_l.index(organ),
                    ]
                
                if organ == 'tongue':
                    pretty_organ = 'lick'
                else:
                    pretty_organ = organ + ' contact'
                ax.set_title(pretty_organ)
                
                if organ == 'blank':
                    ax.set_visible(False)

                try:
                    topl = event_binned.xs(
                        organ, level='organ').xs(
                        event_type, level='event_type')
                except KeyError:
                    continue
                
                ax.plot(binned.t, topl.mean(), color=color, alpha=.5)
                ax.fill_between(
                    x=binned.t, 
                    y1=topl.mean() - topl.sem(),
                    y2=topl.mean() + topl.sem(),
                    color=color, lw=0, alpha=.5)
        

    
    for ax in axa:
        my.plot.despine(ax)
        ax.set_ylim((0, 4.5))
        ax.plot([-.1, .1], [1, 1], 'k--', lw=.75)
        ax.set_xlim((-.1, .1))
        ax.set_yticks([0, 1, 2, 3, 4])
    
    axa[0].set_ylabel('normalized firing rate')
    axa[0].set_xlabel('time from lick (s)')
    axa[3].set_xlabel('time from contact (s)')

    f.savefig('PLOT_PSTHS_TO_LICK_AND_CONTACT_DISC_ONLY.svg')
    f.savefig('PLOT_PSTHS_TO_LICK_AND_CONTACT_DISC_ONLY.png')
    
    
if PLOT_EXAMPLE_RASTERS_TO_LICK_AND_CONTACT:
    ## Plot rasters
    event_l = ['contact', 'lick']
    organ_l = ['tongue', 'blank', 'C1', 'C2', 'C3']
    session = '180222_KF132'
    neuron = 472

    # events on rows, organs on columns
    f, axa = plt.subplots(
        2, len(organ_l), 
        sharex=True, sharey=True, squeeze=False, figsize=(7.5, 3),
        gridspec_kw={'width_ratios': [1, .001, 1, 1, 1]})
    f.subplots_adjust(left=.075, right=.95, bottom=.175, hspace=.2, wspace=.6)

    for event in event_l:
        # Slice data
        event_binned = big_binned.xs(task, level='task').xs(event, level='event')
        event_binned.index = event_binned.index.remove_unused_levels()
        
        # How many event types to plot
        event_type_l = sorted(event_binned.index.get_level_values('event_type').unique())
        
        if event == 'contact':
            event_type_l = event_type_l[::-1]
        
        # Plot each
        for event_type in event_type_l:
            color = event2color[event_type]
            
            for organ in organ_l:
                # Get ax
                ax = axa[
                    event_type_l.index(event_type),
                    organ_l.index(organ),
                    ]
                
                if ax in axa[0]:
                    if organ == 'tongue':
                        pretty_organ = 'lick'
                    else:
                        pretty_organ = organ + ' contact'
                    ax.set_title(pretty_organ)
                
                if organ == 'blank':
                    ax.set_visible(False)
                
                #~ # Slice binned
                #~ topl_binned = event_binned.xs(
                    #~ event_type, level='event_type').xs(organ, level='organ')
                #~ topl_binned.index = topl_binned.index.remove_unused_levels()
                
                # Slice folded
                try:
                    folded = dfolded[(session, event, organ, event_type, neuron)]
                except KeyError:
                    continue
                
                # Subsample
                idxs2 = np.sort(np.unique(my.misc.take_equally_spaced(
                    np.arange(len(folded), dtype=np.int),
                    30)))
                folded2 = [folded.values[idx] for idx in idxs2]
                
                # Raster plot
                ax.eventplot(folded2, color=color, linewidths=.5)
                
                

    for ax in axa.flatten():
        if ax in axa[0]:
            my.plot.despine(ax, which=('left', 'top', 'right', 'bottom'))
        else:
            my.plot.despine(ax, which=('left', 'top', 'right'))
            ax.set_xticks([-.1, 0, .1])
            
        ax.set_yticks([])
        if ax in axa[-1, :]:
            ax.set_xlabel('time (s)')
        
        ax.set_xlim((-.1, .1))
        ax.set_ylim((-1, 31))


    axa[0, 0].set_ylabel('lick left', color=event2color[1])
    axa[1, 0].set_ylabel('lick right', color=event2color[2])
    axa[0, 2].set_ylabel('concave', color=event2color[150])
    axa[1, 2].set_ylabel('convex', color=event2color[50])

    f.savefig('PLOT_EXAMPLE_RASTERS_TO_LICK_AND_CONTACT.svg')
    f.savefig('PLOT_EXAMPLE_RASTERS_TO_LICK_AND_CONTACT.png')
