## Plot PSTHs split by animal's choice (or shape?)
"""
6A
    PLOT_EXAMPLE_CHOICE_SELECTIVE_NEURON
    N/A
    Rasters and PSTHs from example neuron that encodes choice over trial
"""

import json
import os
import tqdm
import numpy as np
import pandas
import kkpandas
import matplotlib.pyplot as plt
import whiskvid
import my
import my.neural
import my.plot
import matplotlib


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
    
    
## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)


## Load data
# Load trial matrix that went into the spikes
big_tm = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'neural_big_tm'))

# cycle data
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')

# Load spiking data
# The version in the neural_dir contains a broader temporal range of cycles
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'aligned_spikes_by_cycle'))
FR_overall = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'FR_overall'))

# Load the contact results
# These have to be manually sliced to remove opto trials
big_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_ccs_df'))

# index by session * trial * cycle * whisker * cluster
big_ccs_df = big_ccs_df.set_index(
    ['trial', 'whisker', 'cycle'], append=True).reorder_levels(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()
    
# Load licks
big_licks = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_licks'))

# Extract bin_width
bin_width = np.mean(np.diff(big_licks.index.values))


## Slice everything by trials in spikes_by_cycle
# Sanity check that all session * trial in spikes are included in big_tm
slicing_midx2 = pandas.MultiIndex.from_frame(
    spikes_by_cycle.index.to_frame().reset_index(drop=True)[
    ['session', 'trial']].drop_duplicates())
assert (big_tm.index == slicing_midx2).all()

# These are the session * trial * cycle to actually include
slicing_midx = pandas.MultiIndex.from_frame(
    spikes_by_cycle.index.to_frame().reset_index(drop=True)[
    ['session', 'trial', 'cycle']].drop_duplicates())

# big_licks
big_licks = my.misc.slice_df_by_some_levels(big_licks.T, big_tm.index).T

# big_ccs_df
big_ccs_df = my.misc.slice_df_by_some_levels(big_ccs_df, slicing_midx)

# C2_whisk_cycles
C2_whisk_cycles = my.misc.slice_df_by_some_levels(C2_whisk_cycles, slicing_midx)


## Lock spikes to rwin
dfolded = {}

for session_name in tqdm.tqdm(sorted(big_tm.index.levels[0])):
    
    ## Get the video session
    vs = whiskvid.django_db.VideoSession.from_name(session_name)

    
    ## Load spikes
    spikes = pandas.read_pickle(
        os.path.join(vs.session_path, 'spikes'))
    included_clusters = np.sort(spikes['cluster'].unique())
    
    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = big_tm.loc[session_name]

    
    ## Lock spikes on contact times
    bins = np.linspace(-2, 1, 41)
    for neuron in included_clusters:
        # Slice spikes
        neuron_spikes = spikes[spikes['cluster'] == neuron]
        
        # Fold
        folded = my.neural.lock_spikes_to_events(
            neuron_spikes['time'],
            trial_matrix['rwin_time_nbase'],
            dstart=-2,
            dstop=1,
            spike_range_t=(spikes['time'].min(), spikes['time'].max()),
            event_range_t=(
                trial_matrix['rwin_time_nbase'].min(), 
                trial_matrix['rwin_time_nbase'].max(), 
            ),
            event_labels=trial_matrix.index,
            )
        

        #~ # Bin by trial
        #~ binned = kkpandas.Binned.from_folded(folded, bins=bins)
        
        # Store
        dfolded[
            (session_name, neuron)
            ] = folded
        #~ binned_l.append(binned.rate_in('Hz')[0])
        #~ binned_keys_l.append(
            #~ (session_name, 'contact', whisker, stepper_pos, neuron))



## Bin contacts in the same way as the licks
# Set up bins
contact_bins = np.linspace(-2, 1, 21)
contact_bin_centers = (contact_bins[1:] + contact_bins[:-1]) / 2.0
#~ assert np.allclose(contact_bin_centers, big_licks.index.values)

# Bin
big_ccs_df['bin'] = pandas.cut(
    big_ccs_df['locked_t'], bins=contact_bins, right=False, labels=False)

# Drop contacts outside range
big_ccs_df = big_ccs_df.dropna(subset=['bin']).copy()
big_ccs_df['bin'] = big_ccs_df['bin'].astype(np.int)

# Count contacts
counted_contacts = big_ccs_df.groupby(['session', 'trial', 'whisker', 'bin']).size()

# Insert missing trials
counted_contacts = counted_contacts.unstack(['whisker', 'bin']).reindex(big_tm.index)

# Reindex by all possible whiskers and bins
midx = pandas.MultiIndex.from_product([
    pandas.Series(['C0', 'C1', 'C2', 'C3'], name='whisker'),
    pandas.Series(range(len(contact_bin_centers)), name='bin')
    ])
counted_contacts = counted_contacts.reindex(midx, axis=1)

# fillna and intify
counted_contacts = counted_contacts.fillna(0).astype(np.int)


## Bin the spikes in the same way as the licks
# Bin the cycles
C2_whisk_cycles['locked_t'] = C2_whisk_cycles['peak_frame_wrt_rwin'] / 200.
C2_whisk_cycles['bin'] = pandas.cut(
    C2_whisk_cycles['locked_t'], bins=contact_bins, right=False, labels=False)

# Drop cycles outside range
C2_whisk_cycles = C2_whisk_cycles.dropna(subset=['bin']).copy()
C2_whisk_cycles['bin'] = C2_whisk_cycles['bin'].astype(np.int)

# Join bin on spikes_by_cycle
spikes_by_cycle = spikes_by_cycle.to_frame().join(C2_whisk_cycles['bin'])

# Drop spikes outside range
spikes_by_cycle = spikes_by_cycle.dropna(subset=['bin']).copy()
spikes_by_cycle['bin'] = spikes_by_cycle['bin'].astype(np.int)

# Sum spikes over cycles within bin, separately by trial
spike_counts = spikes_by_cycle.groupby(
    ['session', 'trial', 'bin', 'neuron'])['spikes'].sum().unstack(
    'bin').fillna(0)


## Aggregate spikes
# Join anatomical info
midx = spike_counts.index.to_frame().reset_index(drop=True)
midx = midx.join(
    big_waveform_info_df[['layer', 'stratum', 'NS']], on=['session', 'neuron'])

# Join trial info
midx = midx.join(
    big_tm[['rewside', 'choice']], on=['session', 'trial'])

# Put back on index
spike_counts.index = pandas.MultiIndex.from_frame(
    midx[['session', 'rewside', 'choice', 'trial', 'stratum', 'NS', 'neuron']])

# Add mouse and task
spike_counts = my.misc.insert_mouse_and_task_levels(spike_counts, mouse2task)

# Mean over everything but trial
msc = spike_counts.mean(level=[lev for lev in spike_counts.index.names if lev != 'trial'])
msc_err = spike_counts.sem(level=[lev for lev in spike_counts.index.names if lev != 'trial'])

# Convert to Hz
spike_bin_width = np.mean(np.diff(contact_bins))
msc = msc / spike_bin_width
msc_err = msc_err / spike_bin_width


## Aggregate contacts
# Add mouse and task
counted_contacts = my.misc.insert_mouse_and_task_levels(
    counted_contacts, mouse2task)

# Mean with task * mouse (or should it go through session?)
mcc = counted_contacts.mean(level=['task', 'mouse'])

# Convert to Hz
mcc = mcc / bin_width


## Binarize big_licks
# Because there's rarely more than 1 per bin, and when it does happen 
# it's probably artefactual
big_licks.values[big_licks.values > 1] = 1


## Massage big licks
# Get session * trial on rows, and lick * time on columns
big_licks.index.name = 'time'
big_licks = big_licks.T.unstack('lick')
big_licks = big_licks.reorder_levels(
    ['lick', 'time'], axis=1).sort_index(axis=1)

# Rename lick human readable
big_licks = big_licks.rename({1: 'left', 2: 'right'}, axis=1, level='lick')
big_licks = big_licks.sort_index(axis=1)


## Relabel licks as correct and incorrect
this_licks_l = []
this_licks_keys_l = []
for rewside in ['left', 'right']:
    # Label the other side
    other_side = {'left': 'right', 'right': 'left'}[rewside]
    
    # Find corresponding trials
    idx = big_tm[big_tm['rewside'] == rewside].index
    
    # Slice big_licks accordingly
    correct_licks = big_licks.loc[idx, rewside]
    incorrect_licks = big_licks.loc[idx, other_side]
    
    # Concat
    this_licks = pandas.concat(
        [correct_licks, incorrect_licks], 
        keys=['correct', 'incorrect'], names=['lick_typ'], axis=1)
    
    # Store
    this_licks_l.append(this_licks)
    this_licks_keys_l.append(rewside)

# Concat
big_licks_by_outcome = pandas.concat(
    this_licks_l, keys=this_licks_keys_l, names=['rewside'])

# Drop the now-useless level rewside
big_licks_by_outcome = big_licks_by_outcome.droplevel('rewside').sort_index()


## Separately relabel licks as concordant and discordant (with choice)
this_licks_l = []
this_licks_keys_l = []
for choice in ['left', 'right']:
    # Label the other side
    other_side = {'left': 'right', 'right': 'left'}[choice]
    
    # Find corresponding trials
    idx = big_tm[big_tm['choice'] == choice].index
    
    # Slice big_licks accordingly
    concordant_licks = big_licks.loc[idx, choice]
    discordant_licks = big_licks.loc[idx, other_side]
    
    # Concat
    this_licks = pandas.concat(
        [concordant_licks, discordant_licks], 
        keys=['concordant', 'discordant'], names=['lick_typ'], axis=1)
    
    # Store
    this_licks_l.append(this_licks)
    this_licks_keys_l.append(choice)
    
# Concat
big_licks_by_cordance = pandas.concat(
    this_licks_l, keys=this_licks_keys_l, names=['choice'])

# Drop the now-useless level choice
big_licks_by_cordance = big_licks_by_cordance.droplevel('choice').sort_index()


## Concat by_outcome and by_cordance
# But note that the same exact licks are contained in each
# The units are lick count (integer) on each trial * timepoint
categorized_licks = pandas.concat(
    [big_licks_by_outcome, big_licks_by_cordance], axis=1,
    keys=['outcome', 'cordance'], names=['by'])

# Error check
assert big_licks_by_outcome.sum().sum() == big_licks_by_cordance.sum().sum()
assert not categorized_licks.isnull().any().any()


## Add mouse and task
categorized_licks = my.misc.insert_mouse_and_task_levels(
    categorized_licks, mouse2task)


## Mean over rewside * session * trial, leaving mouse * task
# The units are still licks (not Hz), but now meaned over trials
categorized_lick_rate = categorized_licks.mean(level=['task', 'mouse'])


## Sum over lick types
# Drop by=cordance, and sum correct and incorrect
all_lick_rate = categorized_lick_rate['outcome'].sum(level='time', axis=1)

# Convert to Hz
all_lick_rate_hz = all_lick_rate / bin_width


## Fractionate over cordance and outcome
# Divide the correct lick rate by total lick rate
normalized_rate_of_correct_licks = (
    categorized_lick_rate['outcome']['correct'].divide(
    categorized_lick_rate['outcome']['correct'] + 
    categorized_lick_rate['outcome']['incorrect'])
    )

# Divide the concordant lick rate by total lick rate
normalized_rate_of_concordant_licks = (
    categorized_lick_rate['cordance']['concordant'].divide(
    categorized_lick_rate['cordance']['concordant'] + 
    categorized_lick_rate['cordance']['discordant'])
    )


## Plot
# All together, there doesn't seem to be any difference between tasks here
PLOT_EXAMPLE_CHOICE_SELECTIVE_NEURON = True

## This also plots licks
if False:
    ## Make figure
    f, ax = my.plot.figure_1x1_small()
    
    
    ## Plot spikes
    # Replace bin number with time
    sliced.index = pandas.Series(contact_bin_centers, name='bincenter')
    
    ax.plot(sliced.loc[:, ('left', 'left')], color='b', ls='-')
    ax.plot(sliced.loc[:, ('left', 'right')], color='b', ls='--')
    ax.plot(sliced.loc[:, ('right', 'right')], color='r', ls='--')
    ax.plot(sliced.loc[:, ('right', 'left')], color='r', ls='-')


    ## Plot licks
    ax.plot(sliced_licks.loc[:, 'left'] * 100, color='k', ls='-')
    ax.plot(sliced_licks.loc[:, 'right'] * 100, color='k', ls='--')
    
    
    ## Pretty
    ax.set_xlabel('time in trial (s)')
    ax.set_ylabel('rate (Hz)')    
    ax.set_ylim(ymin=0)
    ax.set_xlim((-2, 1))
    ax.set_xticks((-2, -1, 0, 1))
    #~ ax.plot([0, 0], [0, 8], '-', lw=.75, color='gray', zorder=0)
    #~ ax.set_ylim((0, 8))
    #~ ax.set_yticks((0, 4, 8))
    my.plot.despine(ax)
    
    #~ my.plot.despine(ax2)
    #~ ax2.set_ylim((0, 32))
    

if PLOT_EXAMPLE_CHOICE_SELECTIVE_NEURON:
    ## Choose example
    session, neuron = '180223_KF132', 349


    ## Slice
    sliced = msc.xs(session, level='session').xs(neuron, level='neuron')
    sliced = sliced.droplevel(['task', 'mouse', 'stratum', 'NS']).T

    sliced_err = msc_err.xs(session, level='session').xs(neuron, level='neuron')
    sliced_err = sliced_err.droplevel(['task', 'mouse', 'stratum', 'NS']).T

    sliced_licks = big_licks.loc[session].mean().unstack('lick')

    folded = dfolded[(session, neuron)]
    session_tm = big_tm.loc[session]


    ## Plot rasters
    choice_l = ['right', 'left']
    n_trials_to_plot = 25
    f, axa = plt.subplots(1, 3, figsize=(6, 2), sharex=True)
    f.subplots_adjust(left=.1, right=.95, top=.9, bottom=.3, wspace=.4)

    for choice in choice_l:
        ax = axa.flatten()[1 + choice_l.index(choice)]

        # Find trials with this choice
        included_trials = session_tm[
            session_tm['choice'] == choice].index.values
        
        # Subsample
        subsampled_trials = np.sort(np.unique(my.misc.take_equally_spaced(
            included_trials, n_trials_to_plot)))
        
        # Slice from folded
        folded2 = [
            folded.values[list(folded.labels).index(trial)] 
            for trial in subsampled_trials]

        # Raster plot
        color = 'b' if choice == 'left' else 'r'
        ax.eventplot(
            folded2, color=color, linewidths=.5, linelengths=.8, alpha=.5)
        
        # Pretty
        my.plot.despine(ax, which=('left', 'top', 'right'))
        ax.set_ylim((-.5, n_trials_to_plot + 0.5))
        #~ ax.set_xticks([])
        ax.set_yticks([])
        
        chosen_stim = 'concave' if choice == 'left' else 'convex'
        ax.set_title('report {}'.format(chosen_stim), color=color, pad=0)
    

    ## Plot PSTHs
    ax = axa[0]
    # Replace bin number with time
    sliced.index = pandas.Series(contact_bin_centers, name='bincenter')
    sliced_err.index = pandas.Series(contact_bin_centers, name='bincenter')

    for choice in ['left', 'right']:
        for rewside in ['left', 'right']:
            # Color by choice
            color = 'b' if choice == 'left' else 'r'
            
            # Linestyle by outcome
            linestyle = '-' if rewside == choice else '--'
            
            # Slice
            topl_m = sliced.loc[:, (rewside, choice)]
            topl_e = sliced_err.loc[:, (rewside, choice)]
            
            # Plot mean
            ax.plot(topl_m, color=color, linestyle=linestyle)
            
            # Plot err
            ax.fill_between(
                x=topl_e.index.values,
                y1=(topl_m - topl_e).values,
                y2=(topl_m + topl_e).values,
                color=color, alpha=.25)


    ## Plot licks
    #~ ax.plot(sliced_licks.loc[:, 'left'] * 100, color='k', ls='-')
    #~ ax.plot(sliced_licks.loc[:, 'right'] * 100, color='k', ls='--')


    for ax in axa:
        ax.set_xlabel('time in trial (s)')


    ## Pretty
    ax = axa[0]
    ax.set_ylabel('firing rate (Hz)')    
    ax.set_ylim(ymin=0)
    ax.set_xlim((-2, 1))
    ax.set_xticks((-2, -1, 0, 1))
    #~ ax.plot([0, 0], [0, 8], '-', lw=.75, color='gray', zorder=0)
    #~ ax.set_ylim((0, 8))
    #~ ax.set_yticks((0, 4, 8))
    my.plot.despine(ax)
    
    f.savefig('PLOT_EXAMPLE_CHOICE_SELECTIVE_NEURON.svg')
    f.savefig('PLOT_EXAMPLE_CHOICE_SELECTIVE_NEURON.png', dpi=300)
