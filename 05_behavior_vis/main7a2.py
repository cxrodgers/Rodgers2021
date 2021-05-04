## Plot lick histograms over course of trial
# now also with contact histograms

"""
1I, left	    
    PLOT_LICK_AND_CONTACT_RATE_OVER_TIME	
    Lick rate versus time in trial
1I, right	
    PLOT_NORMALIZED_CONTACT_RATE_AND_LICK_CORRECT_AND_CONCORDANT_RATE_OVER_TIME	
    Correct/concordant lick probabilities versus time in trial
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


## Fonts
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(
    params, 'big_tm')
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')
   
# Load the contact results
# These have to be manually sliced to remove opto trials
big_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_ccs_df'))

# Load licks
big_licks = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_licks'))

# Use big_tm to slice big_licks
big_licks = my.misc.slice_df_by_some_levels(big_licks.T, big_tm.index).T
big_licks.columns = big_licks.columns.remove_unused_levels()

# Extract bin_width
bin_width = np.mean(np.diff(big_licks.index.values))


## Include only contacts from cycles in C2_whisk_cycles
# Use those cycles to slice big_ccs_df
big_ccs_df = big_ccs_df.set_index(
    ['trial', 'whisker', 'cycle'], append=True).reorder_levels(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()
big_ccs_df = my.misc.slice_df_by_some_levels(
    big_ccs_df, C2_whisk_cycles.index)

# Return big_ccs_df to its original index
big_ccs_df = big_ccs_df.reset_index().set_index(['session', 'cluster'])


## Ensure consistent trials included in both big_licks and big_ccs_df and big_tm
# These should be equal already
my.misc.assert_index_equal_on_levels(
    big_tm, big_licks.T, levels=['session', 'trial'])

# There shouldn't be any trials in big_ccs_df that aren't in big_tm
# big_ccs_df will be missing trials without contacts though
assert len(big_ccs_df) == len(
    my.misc.slice_df_by_some_levels(
    big_ccs_df.set_index('trial', append=True), big_tm.index))


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
PLOT_LICK_AND_CONTACT_RATE_OVER_TIME = True
PLOT_NORMALIZED_CONTACT_RATE_AND_LICK_CORRECT_AND_CONCORDANT_RATE_OVER_TIME = True


if PLOT_LICK_AND_CONTACT_RATE_OVER_TIME:
    ## Make figure
    #~ f, ax = my.plot.figure_1x1_small()
    f, ax = plt.subplots(figsize=(4, 2.25))
    f.subplots_adjust(bottom=.25)
    
    
    ## Aggregate
    topl_mean = all_lick_rate_hz.mean()
    topl_err = all_lick_rate_hz.sem()
    
    
    ## Plot
    ax.plot(
        topl_mean.index,
        topl_mean.values,
        color='k', clip_on=False)
    
    # SEM errorbars
    ax.fill_between(
        x=topl_mean.index,
        y1=topl_mean.values - topl_err.values,
        y2=topl_mean.values + topl_err.values,
        lw=0, alpha=.25, color='k', clip_on=False)
    
    
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
    
    # SEM errorbars
    ax.fill_between(
        x=mcc_topl_mean.index,
        y1=mcc_topl_mean.values - mcc_topl_err.values,
        y2=mcc_topl_mean.values + mcc_topl_err.values,
        lw=0, alpha=.25, color='magenta', clip_on=False)    
    
    
    ## legend
    #~ ax.text(
        #~ 1, 6, 'n = {} mice'.format(len(all_lick_rate_hz)), 
        #~ ha='center', va='bottom', size=12)
    ax.text(-.7, 8.5, 'all contacts', ha='center', va='center', color='magenta')
    ax.text(.6, 5, 'all licks', ha='left', va='center')
    
    ## Pretty
    ax.set_xlabel('time in trial (s)')
    ax.set_ylabel('rate (Hz)')    
    ax.set_ylim(ymin=0)
    ax.set_xlim((-2, 1))
    ax.set_xticks((-2, -1, 0, 1))
    ax.plot([0, 0], [0, 8], '-', lw=.75, color='gray', zorder=0)
    ax.set_ylim((0, 8))
    ax.set_yticks((0, 4, 8))
    my.plot.despine(ax)
    
    
    f.savefig('PLOT_LICK_AND_CONTACT_RATE_OVER_TIME.svg')
    f.savefig('PLOT_LICK_AND_CONTACT_RATE_OVER_TIME.png', dpi=300)



if PLOT_NORMALIZED_CONTACT_RATE_AND_LICK_CORRECT_AND_CONCORDANT_RATE_OVER_TIME:
    ## Make figure
    #~ f, ax = my.plot.figure_1x1_small()
    f, ax = plt.subplots(figsize=(4, 2.25))
    f.subplots_adjust(bottom=.25)
    
    
    ## Aggregate
    topl_mean_corr = normalized_rate_of_correct_licks.mean()
    topl_err_corr = normalized_rate_of_correct_licks.sem()
    
    topl_mean_conc = normalized_rate_of_concordant_licks.mean()
    topl_err_conc = normalized_rate_of_concordant_licks.sem()

    
    ## Plot correct
    # Mean
    ax.plot(
        topl_mean_corr.index,
        topl_mean_corr.values,
        color='k', clip_on=False)
    
    # SEM errorbars
    ax.fill_between(
        x=topl_mean_corr.index,
        y1=topl_mean_corr.values - topl_err_corr.values,
        y2=topl_mean_corr.values + topl_err_corr.values,
        lw=0, alpha=.25, color='k', clip_on=False)
    

    ## Plot concordant
    # Mean
    ax.plot(
        topl_mean_conc.index,
        topl_mean_conc.values,
        color='k', clip_on=False, ls='--')
    
    # SEM errorbars
    ax.fill_between(
        x=topl_mean_conc.index,
        y1=topl_mean_conc.values - topl_err_conc.values,
        y2=topl_mean_conc.values + topl_err_conc.values,
        lw=0, alpha=.25, color='k', clip_on=False)


    ## Plot contacts
    # Sum over whiskers
    mcc_topl = mcc.sum(level='bin', axis=1)
    
    # Replace bin number with time
    mcc_topl.columns = pandas.Series(contact_bin_centers, name='bincenter')
    
    # Aggregate over mice (should I separate by task?)
    mcc_topl_mean = mcc_topl.mean()
    mcc_topl_err = mcc_topl.sem()

    # Plot mean
    ax2 = ax.twinx()
    ax2.plot(
        mcc_topl_mean.index,
        mcc_topl_mean.values,
        color='magenta', clip_on=False)
    
    # SEM errorbars
    ax2.fill_between(
        x=mcc_topl_mean.index,
        y1=mcc_topl_mean.values - mcc_topl_err.values,
        y2=mcc_topl_mean.values + mcc_topl_err.values,
        lw=0, alpha=.25, color='magenta', clip_on=False)    
    
    
    ## Legend
    ax.text(0, 1.1, 'congruent licks', size=12, ha='center', va='center')
    ax.text(.95, .8, 'correct licks', size=12, ha='center', va='center')
    ax2.set_yticks([])
    
    ## Pretty
    ax.set_xlabel('time in trial (s)')
    ax.set_ylabel('lick probability')
    ax.plot([-2, 1], [.5, .5], '-', lw=.75, color='gray', zorder=0)
    ax.plot([0, 0], [0, 1], '-', lw=.75, color='gray', zorder=0)
    ax.set_ylim((0, 1))
    ax.set_xlim((-2, 1))
    ax.set_xticks((-2, -1, 0, 1))
    ax.set_yticks((0, 1))
    my.plot.despine(ax)
    my.plot.despine(ax2)

    f.savefig('PLOT_NORMALIZED_CONTACT_RATE_AND_LICK_CORRECT_AND_CONCORDANT_RATE_OVER_TIME.svg')
    f.savefig('PLOT_NORMALIZED_CONTACT_RATE_AND_LICK_CORRECT_AND_CONCORDANT_RATE_OVER_TIME.png', dpi=300)


plt.show()