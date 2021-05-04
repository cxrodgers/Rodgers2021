## Plot average PSTH over trial vs licks and contact rates
"""
S5A
    PLOT_LICK_AND_CONTACT_RATE_OVER_TIME
    N/A
    Rate of licks, contacts, and spikes over the trial
"""
import json
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import whiskvid
import my
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


## Bin contacts in the same way as the licks
# Set up bins
contact_bins = np.linspace(-2, 1, 31)
contact_bin_centers = (contact_bins[1:] + contact_bins[:-1]) / 2.0
assert np.allclose(contact_bin_centers, big_licks.index.values)

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
spike_counts.index = pandas.MultiIndex.from_frame(
    midx[['session', 'trial', 'stratum', 'NS', 'neuron']])

# Mean over trials within neuron
msc = spike_counts.mean(level=['stratum', 'NS', 'session', 'neuron'])

# Convert to Hz
msc = msc / bin_width


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
PLOT_LICK_SPIKES_AND_CONTACT_RATE_OVER_TIME = True


if PLOT_LICK_SPIKES_AND_CONTACT_RATE_OVER_TIME:
    ## Make figure
    f, ax = plt.subplots(figsize=(3.5, 2))
    
    # left = .3 is for the case of yticklabels with two signif digits
    f.subplots_adjust(bottom=.28, left=.25, right=.75, top=.95)
    
    
    ## Aggregate
    topl_mean = all_lick_rate_hz.mean()
    topl_err = all_lick_rate_hz.sem()
    
    
    ## Plot
    ax.plot(
        topl_mean.index,
        topl_mean.values,
        color='k', clip_on=False)
    
    #~ # SEM errorbars
    #~ ax.fill_between(
        #~ x=topl_mean.index,
        #~ y1=topl_mean.values - topl_err.values,
        #~ y2=topl_mean.values + topl_err.values,
        #~ lw=0, alpha=.25, color='k', clip_on=False)
    
    
    ## Plot contacts
    # Sum over whiskers
    mcc_topl = mcc.sum(level='bin', axis=1)
    
    # Replace bin number with time
    mcc_topl.columns = pandas.Series(contact_bin_centers, name='bincenter')
    
    # Aggregate over mice (should I separate by task?)
    mcc_topl_mean = mcc_topl.mean()
    mcc_topl_err = mcc_topl.sem()

    # Plot mean
    ax.plot(
        mcc_topl_mean.index,
        mcc_topl_mean.values,
        color='magenta', clip_on=False)
    
    #~ # SEM errorbars
    #~ ax.fill_between(
        #~ x=mcc_topl_mean.index,
        #~ y1=mcc_topl_mean.values - mcc_topl_err.values,
        #~ y2=mcc_topl_mean.values + mcc_topl_err.values,
        #~ lw=0, alpha=.25, color='magenta', clip_on=False)    
    
    
    ## Plot spikes
    # Replace bin number with time
    msc_topl = msc
    msc_topl.columns = pandas.Series(contact_bin_centers, name='bincenter')
    
    # Aggregate over mice (should I separate by task?)
    msc_topl_mean = msc_topl.mean(level=['stratum', 'NS'])
    msc_topl_err = msc_topl.sem(level=['stratum', 'NS'])

    # Plot mean
    ax2 = ax.twinx()
    
    for stratum, NS in msc_topl_mean.index:
        # This cell class
        topl_m = msc_topl_mean.loc[(stratum, NS)]
        topl_err = msc_topl_err.loc[(stratum, NS)]
        
        color = 'b' if NS else 'r'
        ls = '-' if stratum == 'deep' else '--'
        
        # Plot mean
        ax2.plot(
            topl_m.index,
            topl_m.values,
            color=color, ls=ls, clip_on=False)
        
        #~ # SEM errorbars
        #~ ax2.fill_between(
            #~ x=topl_m.index,
            #~ y1=topl_m.values - topl_err.values,
            #~ y2=topl_m.values + topl_err.values,
            #~ lw=0, alpha=.25, color=color, clip_on=False)    

   
    
    ## Pretty
    ax.set_xlabel('time in trial (s)')
    #~ ax.set_ylabel('rate (Hz)')    
    ax.set_ylim(ymin=0)
    ax.set_xlim((-2, 1))
    ax.set_xticks((-2, -1, 0, 1))
    #~ ax.plot([0, 0], [0, 8], '-', lw=.75, color='gray', zorder=0)
    ax.set_ylim((0, 8))
    #~ ax.set_yticks((0, 4, 8))
    my.plot.despine(ax, which=('left', 'top', 'right'))
    
    my.plot.despine(ax2, which=('left', 'top', 'right'))
    ax.set_yticks([])
    ax2.set_yticks([])
    ax2.set_ylim((0, 32))
    
    
    # Plot scale bars
    ax.plot([-2.5, -2.5], [0, 4], 'k-', clip_on=False)
    ax.text(-2.7, 2, '4 Hz', ha='right', va='center')
    ax2.plot([1.5, 1.5], [0, 16], 'k-', clip_on=False)
    ax2.text(1.7, 8, '16 Hz', ha='left', va='center')
    
    f.savefig('PLOT_LICK_SPIKES_AND_CONTACT_RATE_OVER_TIME.svg')
    f.savefig('PLOT_LICK_SPIKES_AND_CONTACT_RATE_OVER_TIME.png', dpi=300)

plt.show()

