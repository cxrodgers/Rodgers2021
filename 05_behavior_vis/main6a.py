## Plots performance vs contacts, and distributions of contact counts by stimulus
"""
4D
    PLOT_PERFORMANCE_VS_N_WHISKERS_THAT_TOUCHED
    TODO: Dump STATS
    Performance versus number of whiskers that touched the stimulus

4E
    PLOT_PERFORMANCE_VS_N_C1_CONTACTS
    TODO: Dump STATS
    Performance versus the number of C1 and C3 contacts on that trial

2E
    PLOT_PERFORMANCE_VS_N_CONTACTS
    STATS__PLOT_PERFORMANCE_VS_N_CONTACTS
    Performance versus number of contacts by task and stimulus

2H
    BAR_PLOT_CONTACT_COUNTS
    STATS__BAR_PLOT_CONTACT_COUNTS
    Contact count by task, by whisker or by single- vs multi-

3H
    PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY
    STATS__PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY
    Bar plot total number of contacts by whisker, stimulus, and position during discrimination
"""

import json
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import my
import my.plot
import scipy.stats


## Fonts
my.plot.poster_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(
    params, 'big_tm', mouse2task=mouse2task)
big_grasp_df = my.dataload.load_data_from_patterns(
    params, 'big_grasp_df')

# logreg data
features = my.dataload.load_data_from_logreg(
    params, 'obliviated_aggregated_features', mouse2task=mouse2task)
oufwb = my.dataload.load_data_from_logreg(
    params, 'obliviated_unaggregated_features_with_bin')

# Load BINS
BINS = pandas.read_pickle(os.path.join(params['logreg_dir'], 'BINS'))


## Process big_tm
# Add stimulus name
stepper_pos2stimulus = {
    50: 'convex', 150: 'concave', 100: 'nothing', 199: 'nothing'}
big_tm['stimulus'] = big_tm['stepper_pos'].map(stepper_pos2stimulus)

# Define task_stimulus which is 'nothing' vs 'something' for detection
big_tm['task_stimulus'] = big_tm['stimulus'].copy()
big_tm.loc[
    big_tm['task_stimulus'].isin(['concave', 'convex']) &
    (big_tm.index.get_level_values('task') == 'detection'),
    'task_stimulus'] = 'something'


## Extract performance by mouse
outcome_counts_by_session = big_tm.groupby(['task', 'mouse', 'session'])[
    'outcome'].value_counts().unstack()
mouse_perf = (
    outcome_counts_by_session['hit'] / outcome_counts_by_session.sum(1)).mean(
    level=['task', 'mouse']).rename('perf')


## Process features
# Extract grasp label by cycle, indexed like oufwb
grasp_label = oufwb.index.to_frame().reset_index(drop=True).join(
    big_grasp_df['label_noC0'], on=['session', 'trial', 'cycle'])

# Count labels by analysis_bin
grasp_count_by_bin = grasp_label.groupby(
    ['session', 'trial', 'analysis_bin'])[
    'label_noC0'].value_counts().unstack().fillna(0).astype(np.int)

# Add mouse and task levels
grasp_count_by_bin = my.misc.insert_mouse_and_task_levels(
    grasp_count_by_bin, mouse2task)
    
# Extract contact and grasp counts from features
contact_count_by_bin = features['contact_binarized'].stack('analysis_bin').copy()

# Reindex grasp_count_by_bin in the same way, to get the bins without grasps
grasp_count_by_bin = grasp_count_by_bin.reindex(
    contact_count_by_bin.index)

# Error check that the null bins are the ones without contacts
nullmask = grasp_count_by_bin.isnull().any(1)
assert (contact_count_by_bin.loc[nullmask.index[nullmask.values]] == 0).all().all()

# Fillna
grasp_count_by_bin = grasp_count_by_bin.fillna(0).astype(np.int)


## Sum over analysis_bin within trial
# Don't apply temporal windowing here because it's already been obliviated
contact_count_by_trial = contact_count_by_bin.sum(
    level=list(contact_count_by_bin.index.to_frame().drop(
    'analysis_bin', 1).columns))
grasp_count_by_trial = grasp_count_by_bin.sum(
    level=list(grasp_count_by_bin.index.to_frame().drop(
    'analysis_bin', 1).columns))

# Add a 'total' column that is the sum of all whiskers
whisker_l = ['C0', 'C1', 'C2', 'C3']
contact_count_by_trial['total'] = contact_count_by_trial[whisker_l].sum(1)
grasp_count_by_trial['total'] = grasp_count_by_trial.sum(1)


## Join on trial parameters
columns_to_join = [
    'rewside', 'outcome', 'choice',  
    'stimulus', 'task_stimulus', 'servo_pos',
    ]

contact_count_by_trial = pandas.merge(
    contact_count_by_trial, big_tm[columns_to_join], 
    validate='1:1', on=['task', 'mouse', 'session', 'trial'])

grasp_count_by_trial = pandas.merge(
    grasp_count_by_trial, big_tm[columns_to_join], 
    validate='1:1', on=['task', 'mouse', 'session', 'trial'])


## Exclude nothing trials
contact_count_by_bin_excluding_nothing = my.misc.slice_df_by_some_levels(
    contact_count_by_bin, 
    contact_count_by_trial[
    contact_count_by_trial['stimulus'] != 'nothing'].index)

grasp_count_by_bin_excluding_nothing = my.misc.slice_df_by_some_levels(
    grasp_count_by_bin, 
    grasp_count_by_trial[
    grasp_count_by_trial['stimulus'] != 'nothing'].index)

contact_count_by_trial_excluding_nothing = my.misc.slice_df_by_some_levels(
    contact_count_by_trial, 
    contact_count_by_trial[
    contact_count_by_trial['stimulus'] != 'nothing'].index)

grasp_count_by_trial_excluding_nothing = my.misc.slice_df_by_some_levels(
    grasp_count_by_trial, 
    grasp_count_by_trial[
    grasp_count_by_trial['stimulus'] != 'nothing'].index)


## Plot flags
# Performance by # of whiskers that touched
PLOT_PERFORMANCE_VS_N_WHISKERS_THAT_TOUCHED = True

# Performance by # of contacts in the trial
PLOT_PERFORMANCE_VS_N_C1_CONTACTS = True
PLOT_PERFORMANCE_VS_N_CONTACTS = True

# Summed counts over time
BAR_PLOT_CONTACT_COUNTS = True
 
# These show mean contact count by stimulus
PLOT_TOTAL_CONTACTS_BY_STIMULUS = False # Not in paper, but nice figure
PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY = True


## Plots
if PLOT_PERFORMANCE_VS_N_WHISKERS_THAT_TOUCHED:
    ## Count the total number of whiskers that touched on each trial
    this_count_by_trial = contact_count_by_trial.copy()
    this_count_by_trial['n_whiskers'] = (
        (this_count_by_trial['C1'] > 0).astype(np.int) + 
        (this_count_by_trial['C2'] > 0).astype(np.int) + 
        (this_count_by_trial['C3'] > 0).astype(np.int)
        )


    ## Calculate performance by task * mouse * rewside * n_whiskers
    outcomes_by_n_contacts_bin = this_count_by_trial.groupby(
        ['task', 'mouse', 'task_stimulus', 'n_whiskers'])['outcome'
        ].value_counts().unstack('outcome').fillna(0).astype(np.int)
    perf_by_n_contacts_bin = outcomes_by_n_contacts_bin['hit'].divide(
        outcomes_by_n_contacts_bin.sum(1)).rename('perf')

    # Put NaN in for low-trial counts
    trial_counts = outcomes_by_n_contacts_bin.sum(1)
    perf_by_n_contacts_bin.loc[(trial_counts < 10)] = np.nan
    
    # Unstack n_contacts_bin
    perf_by_n_contacts_bin = perf_by_n_contacts_bin.unstack('n_whiskers')
    
    
    ## Error check that "nothing" is always zero contacts
    nothing_perf = perf_by_n_contacts_bin.xs('nothing', level='task_stimulus')
    assert not nothing_perf.loc[:, 0].isnull().any()
    assert nothing_perf.iloc[:, 1:].isnull().all().all()
    
    
    ## Make figure
    f, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    f.subplots_adjust(bottom=.23, top=.775)
   
    # Do discrimination only
    task = 'discrimination'
    

    ## Slice by task
    # Get data
    task_pbncb = perf_by_n_contacts_bin.loc[task].copy()
    task_pbncb.index = task_pbncb.index.remove_unused_levels()
    
    # Nullify values where either task_stimulus is NaN, because it doesn't
    # make sense to average in this case
    to_nullify = task_pbncb.isnull().any(level='mouse')
    
    # Mean out task_stimulus
    task_pbncb = task_pbncb.mean(level='mouse')
    task_pbncb.values[to_nullify.values] = np.nan
    
    # Get topl
    topl = task_pbncb.T

    
    ## Stats: anova on n_whiskers
    aov_res = my.stats.anova(
        df=topl.stack().rename('perf').reset_index(), 
        fmla='perf ~ mouse + n_whiskers')
    
    # Extract pvalue on n_whiskers
    pvalue = aov_res['pvals']['p_n_whiskers']

    
    ## Plot
    # Mean mouse
    mtopl = topl.mean(1)
    errtopl = topl.std(1)    
    
    # Plot
    ax.bar(
        x=mtopl.index.values,
        height=mtopl.values,
        yerr=errtopl.values,
        fc='none',
        ec='k',
        error_kw={'lw': 1},
        )
    
    # Plot sig
    ax.plot([0, 3], [.93, .93], 'k-', lw=.8)
    ax.text(1.5, .95, my.stats.pvalue_to_significance_string(pvalue), 
        ha='center', va='center')
    
    # Pretty
    my.plot.despine(ax)
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_yticklabels(('0.0', '', '0.5', '', '1.0'))
    ax.plot([-.5, 3.5], [.5, .5], 'k--', lw=.8)
    ax.set_ylim((.4, 1))

    ax.set_ylabel('performance')
    ax.set_xlabel('number of whiskers that touched')
    ax.set_xticks([0, 1, 2, 3])
    
    
    ## Save
    f.savefig('PLOT_PERFORMANCE_VS_N_WHISKERS_THAT_TOUCHED.svg')
    f.savefig('PLOT_PERFORMANCE_VS_N_WHISKERS_THAT_TOUCHED.png')


if PLOT_PERFORMANCE_VS_N_C1_CONTACTS:
    PLOT_INDIVIDUAL_MICE = False

    ## Bin the number of contacts per trial
    edges = np.array([0, 1, 4, 9, 16, 1000], dtype=np.int)
    edges = np.array([0, 1, 2, 3, 1000], dtype=np.int)
    this_count_by_trial = contact_count_by_trial.copy()
    
    
    ## Cut each whisker separately
    this_count_by_trial = my.misc.cut_dataframe(
        this_count_by_trial, 'C1', edges, 
        new_column='n_C1_contacts_bin', dropna=False, right=False)
    this_count_by_trial = my.misc.cut_dataframe(
        this_count_by_trial, 'C2', edges, 
        new_column='n_C2_contacts_bin', dropna=False, right=False)        
    this_count_by_trial = my.misc.cut_dataframe(
        this_count_by_trial, 'C3', edges, 
        new_column='n_C3_contacts_bin', dropna=False, right=False)


    ## Aggregate perf separately by whisker (consider also jointly here)
    perf_by_n_contacts_bin_l = []
    perf_by_n_contacts_bin_keys_l = []
    for whisker in ['C1', 'C2', 'C3']:
        # Get key
        key = 'n_{}_contacts_bin'.format(whisker)
        
        # Calculate performance by task * mouse * rewside * n_contacts_bin
        outcomes_by_n_contacts_bin = this_count_by_trial.groupby(
            ['task', 'mouse', 'task_stimulus', key])['outcome'
            ].value_counts().unstack('outcome').fillna(0).astype(np.int)
        perf_by_n_contacts_bin = outcomes_by_n_contacts_bin['hit'].divide(
            outcomes_by_n_contacts_bin.sum(1)).rename('perf')

        # Put NaN in for low-trial counts
        # This affects the low-contact bins (esp 0) on convex on 4 mice,
        # and the highest contact bin on concave on KF132
        # It doesn't end up affecting the results very much
        # But since KM91 never touched the convex shape 0 times, there will
        # always be at least one bin that is missing at least one mouse
        trial_counts = outcomes_by_n_contacts_bin.sum(1)
        perf_by_n_contacts_bin.loc[(trial_counts < 10)] = np.nan
        
        # Unstack n_contacts_bin
        perf_by_n_contacts_bin = perf_by_n_contacts_bin.unstack(key)
        
        # Rename for consistency
        perf_by_n_contacts_bin.columns.name = 'n_contacts_bin'
        
        # Store
        perf_by_n_contacts_bin_l.append(perf_by_n_contacts_bin)
        perf_by_n_contacts_bin_keys_l.append(whisker)
    
    # Concat over whiskers
    perf_by_n_contacts_bin = pandas.concat(
        perf_by_n_contacts_bin_l, keys=perf_by_n_contacts_bin_keys_l, 
        names=['whisker'], axis=1)
    
    
    ## Slice by task
    task = 'discrimination' 
    
    # Get data
    task_pbncb = perf_by_n_contacts_bin.loc[task].copy()
    task_pbncb.index = task_pbncb.index.remove_unused_levels()
    
    # Get stimuli for this task
    stimulus_l = list(task_pbncb.index.get_level_values('task_stimulus').unique())

    
    ## Include only C1 and C3 for simplicity
    whisker_l = ['C1', 'C3']
    task_pbncb = task_pbncb.loc[:, whisker_l]


    ## Stats: 1way anova on n_contacts_bin, separately by shape and whisker
    pvalue_l = []
    pvalue_keys_l = []
    for stimulus in stimulus_l:
        for whisker in whisker_l:
            # Slice by stimulus and whisker
            aov_data = task_pbncb.xs(stimulus, level='task_stimulus').xs(whisker, level='whisker', axis=1)
            
            # Stack and dataframe
            aov_data = aov_data.stack().rename('perf').reset_index()
            
            # Run AOV
            aov_res = my.stats.anova(
                df=aov_data,
                fmla='perf ~ n_contacts_bin',
                typ=1)
            
            # Extract pvalue on n_whiskers
            pvalue = aov_res['pvals']['p_n_contacts_bin']
            pvalue_l.append(pvalue)
            pvalue_keys_l.append((stimulus, whisker))

    # Concat
    pvalues = pandas.Series(pvalue_l, 
        index=pandas.MultiIndex.from_tuples(
        pvalue_keys_l, names=['stimulus', 'whisker']))

    
    ## Plot
    whisker_l = ['C1', 'C3']
    f, axa = plt.subplots(1, len(stimulus_l), figsize=(8, 2.5), sharex=True, sharey=True)
    f.subplots_adjust(bottom=.23, top=.775, wspace=.2, right=.97, left=.1)
   
    
    ## Iterate over whisker
    for stimulus in stimulus_l:
        ## Slice
        # Get ax
        ax = axa.flatten()[stimulus_l.index(stimulus)]
        ax.set_title(stimulus, pad=25)
        
        # Extract data for this whisker
        whisker_pbncb = task_pbncb.xs(stimulus, axis=0, level='task_stimulus')
        
        # Get mouse (replicates) on columns
        topl = whisker_pbncb.T


        ## Plot
        def index2plot_kwargs(idx):
            return {'fc': 'w', 'ec': 'k'}

        def index2label(idx):
            if False: #idx['n_contacts_bin'] == edges[-2]:
                return '{}+'.format(idx['n_contacts_bin'])
            else:
                return '{}'.format(idx['n_contacts_bin'])

        my.plot.grouped_bar_plot(
            topl, 
            ax=ax,
            index2plot_kwargs=index2plot_kwargs, 
            index2label=index2label, 
            plot_error_bars_instead_of_points=True,
            group_name_fig_ypos=.85,
            )
        
        
        ## Plot sig
        sig_ypos = .935
        for whisker in whisker_l:
            pvalue = pvalues.loc[(stimulus, whisker)]
            sig_str = my.stats.pvalue_to_significance_string(pvalue)
            x_offset = 5 if whisker == 'C3' else 0
            ax.plot([x_offset, x_offset + len(edges) - 2], [sig_ypos] * 2, 'k-', lw=.75)
            ax.text(
                x_offset + 0.5 * (len(edges) - 2), 
                sig_ypos + .025, sig_str, ha='center')


    ## Pretty
    for ax in axa.flatten():
        my.plot.despine(ax)
        
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_ylim((.5, 1))
        #~ ax.set_yticklabels(('0.0', '', '0.5', '', '1.0'))
        #~ ax.plot([-.5, len(edges) - 1.5], [.5, .5], 'k--', lw=.8)
        #~ ax.set_xlim((-.25, len(edges) - 1.75))
        #~ ax.set_xticks(range(len(edges) - 1))
        #~ xtls = [
            #~ '{}{}'.format(bin, '+' if n_bin == len(edges) - 2 else '')
            #~ for n_bin, bin in enumerate(edges[:-1])]
        #~ ax.set_xticklabels(xtls)
    
    
    ## Pretty
    axa[0].set_ylabel('performance')
    #~ axa[1].set_yticklabels([])
    f.text(.55, .07, 'number of contacts', ha='center', va='center')    
    
    f.savefig('PLOT_PERFORMANCE_VS_N_C1_CONTACTS.svg')
    f.savefig('PLOT_PERFORMANCE_VS_N_C1_CONTACTS.png', dpi=300)

if PLOT_PERFORMANCE_VS_N_CONTACTS:
    ## Choose how to quantify
    # Counting grasps seems slighty cleaner, but basically the same
    TYP = 'grasp'
    PLOT_INDIVIDUAL_MICE = False
    
    if TYP == 'contact':
        ## Bin the number of contacts per trial
        edges = np.array([0, 1, 4, 9, 16, 1000], dtype=np.int)
        this_count_by_trial = contact_count_by_trial.copy()
        
        # Cut
        this_count_by_trial = my.misc.cut_dataframe(
            this_count_by_trial, 'total', edges, 
            new_column='n_contacts_bin', dropna=False, right=False)
        assert not this_count_by_trial['n_contacts_bin'].isnull().any()

    elif TYP == 'grasp':
        ## Bin the number of grasps per trial
        edges = np.array([0, 1, 2, 4, 8, 1000], dtype=np.int)
        this_count_by_trial = grasp_count_by_trial.copy()
        
        # Cut
        this_count_by_trial = my.misc.cut_dataframe(
            this_count_by_trial, 'total', edges, 
            new_column='n_contacts_bin', dropna=False, right=False)
        assert not this_count_by_trial['n_contacts_bin'].isnull().any()
    
    else:
        1/0


    ## Calculate performance by task * mouse * rewside * n_contacts_bin
    outcomes_by_n_contacts_bin = this_count_by_trial.groupby(
        ['task', 'mouse', 'task_stimulus', 'n_contacts_bin'])['outcome'
        ].value_counts().unstack('outcome').fillna(0).astype(np.int)
    perf_by_n_contacts_bin = outcomes_by_n_contacts_bin['hit'].divide(
        outcomes_by_n_contacts_bin.sum(1)).rename('perf')

    # Put NaN in for low-trial counts
    # This affects the low-contact bins (esp 0) on convex on 4 mice,
    # and the highest contact bin on concave on KF132
    # It doesn't end up affecting the results very much
    # But since KM91 never touched the convex shape 0 times, there will
    # always be at least one bin that is missing at least one mouse
    trial_counts = outcomes_by_n_contacts_bin.sum(1)
    perf_by_n_contacts_bin.loc[(trial_counts < 10)] = np.nan
    
    # Unstack n_contacts_bin
    perf_by_n_contacts_bin = perf_by_n_contacts_bin.unstack('n_contacts_bin')
    
    
    ## Slice by mouse
    # The "abnormal" ones that detect concave
    #~ perf_by_n_contacts_bin = perf_by_n_contacts_bin.loc[
        #~ pandas.IndexSlice[:, ['200CR', '221CR', 'KF134', '229CR']], :]
    
    #~ # The "normal" ones that detect convex
    #~ perf_by_n_contacts_bin = perf_by_n_contacts_bin.loc[
        #~ pandas.IndexSlice[:, ['200CR', 'KF119', 'KF132', 'KM101', 'KM131', 'KM91', '219CR']], :]


    
    ## Error check that "nothing" is always zero contacts
    nothing_perf = perf_by_n_contacts_bin.xs('nothing', level='task_stimulus')
    assert not nothing_perf.loc[:, 0].isnull().any()
    assert nothing_perf.iloc[:, 1:].isnull().all().all()


    ## Make figure
    task_l = ['detection', 'discrimination']
    f, axa = plt.subplots(1, 2, figsize=(4.2, 2.1))
    f.subplots_adjust(left=.175, right=.975, bottom=.275, top=.85, wspace=.4)
   
    
    ## Iterate over task and stimulus
    for task in task_l:
        ## Slice by task
        # Get data
        task_pbncb = perf_by_n_contacts_bin.loc[task].copy()
        task_pbncb.index = task_pbncb.index.remove_unused_levels()
        
        # Get ax
        ax = axa.flatten()[task_l.index(task)]
        ax.set_title(task)
        
        
        ## Iterate over stimulus
        stimulus_l = list(task_pbncb.index.get_level_values('task_stimulus').unique())
        
        for stimulus in stimulus_l:
            ## Extract data for this task and stimulus
            task_stim_pbncb = task_pbncb.xs(stimulus, level='task_stimulus')
            topl = task_stim_pbncb.T

            # Get color
            color = {
                'nothing': 'orange', 'something': 'purple', 
                'concave': 'b', 'convex': 'r',
                }[stimulus]
            
            
            ## Mean mouse
            mtopl = topl.mean(1)
            errtopl = topl.sem(1)
            
            
            ## Plot
            if PLOT_INDIVIDUAL_MICE:
                # Individual mice as lines (TODO: interpolate through null)
                ax.plot(
                    topl.index.values, topl.values,
                    color=color, alpha=.5, lw=.5, clip_on=False,
                )

                if stimulus == 'nothing':
                    # Individual mice as points
                    ax.plot(
                        topl.index.values, topl.values,
                        color=color, alpha=.5, lw=.5, clip_on=False,
                        ls='none', marker='.', mfc='w', ms=4,
                    )
            
            
            ## Plot the mean over mice
            ax.plot(
                mtopl.index.values, mtopl.values,
                color=color, clip_on=False, lw=1,
            )       
            
            # Shade the error bars
            ax.fill_between(
                x=mtopl.index.values,
                y1=(mtopl - errtopl).values,
                y2=(mtopl + errtopl).values,
                color=color, alpha=.25, lw=0)


            ## Special case plot the mean over mice for nothing stimulus
            if stimulus == 'nothing':
                if PLOT_INDIVIDUAL_MICE:
                    ax.plot(
                        mtopl.index.values, mtopl.values,
                        color=color, alpha=.5, lw=3, clip_on=False,
                        ls='none', marker='o', mfc='w', ms=4,
                    )            
                else:
                    # Errorbar
                    ax.errorbar(
                        x=mtopl.index.values, 
                        y=mtopl.values,
                        yerr=errtopl.values,
                        color=color, clip_on=False,
                        ls='none', marker='o', mfc='none', ms=4, mew=1, elinewidth=1,
                    )                                

    
        ## Pretty
        my.plot.despine(ax)
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_yticklabels(('0.0', '', '0.5', '', '1.0'))
        ax.plot([-.5, len(edges) - 1.5], [.5, .5], 'k--', lw=.8)
        ax.set_xlim((-.25, len(edges) - 1.75))
        ax.set_xticks(range(len(edges) - 1))
        xtls = [
            '{}{}'.format(bin, '+' if n_bin == len(edges) - 2 else '')
            for n_bin, bin in enumerate(edges[:-1])]
        ax.set_xticklabels(xtls)
    
    
    ## Pretty
    axa[0].set_ylabel('performance')
    axa[1].set_yticklabels([])
    f.text(.55, .075, 'number of contacts', ha='center', va='center')

    axa[0].text(3, .32, 'nothing', color='orange', ha='center', va='center', size=12)
    axa[0].text(3, .18, 'something', color='purple', ha='center', va='center', size=12)
    axa[1].text(3, .32, 'concave', color='blue', ha='center', va='center', size=12)
    axa[1].text(3, .18, 'convex', color='red', ha='center', va='center', size=12)

    plt.show()
    

    ## Save
    f.savefig('PLOT_PERFORMANCE_VS_N_CONTACTS.svg')
    f.savefig('PLOT_PERFORMANCE_VS_N_CONTACTS.png', dpi=300)
    
    
    ## Stats
    with open('STATS__PLOT_PERFORMANCE_VS_N_CONTACTS', 'w') as fi:
        det_perf = perf_by_n_contacts_bin.loc['detection'].unstack()
        disc_perf = perf_by_n_contacts_bin.loc['discrimination'].unstack()
        
        fi.write('STATS__PLOT_PERFORMANCE_VS_N_CONTACTS\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(det_perf) + len(disc_perf),
            len(det_perf),
            len(disc_perf),
            ))
        fi.write('error bars: sem (excluding null bins)\n')
        fi.write('Some mice are null in some bins:\n')
        null_ratio = perf_by_n_contacts_bin.isnull().mean(1).drop(
            'nothing', level='task_stimulus').sort_values()
        fi.write(null_ratio.to_string() + '\n')
    
    with open('STATS__PLOT_PERFORMANCE_VS_N_CONTACTS', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if BAR_PLOT_CONTACT_COUNTS:
    ## Show contact counts by whisker, and by single vs multi, for each task
    ## Extract contact counts
    contact_counts = contact_count_by_bin_excluding_nothing.copy()
    
    # Sum over time bins
    # Because this comes from logreg, it's all pre-choice
    contact_counts = contact_counts.sum(
        level=[lev for lev in contact_counts.index.names 
        if lev != 'analysis_bin'])
    
    # Drop C0
    contact_counts = contact_counts.drop('C0', 1).copy()
    
    
    ## Extract grasp counts
    grasp_counts = grasp_count_by_bin_excluding_nothing.copy()

    # Sum over time bins
    # Because this comes from logreg, it's all pre-choice
    grasp_counts = grasp_counts.sum(
        level=[lev for lev in grasp_counts.index.names 
        if lev != 'analysis_bin'])    
    
    # Drop C0 (which is separately grasped anyway)
    grasp_counts = grasp_counts.drop('C0', 1).copy()

    # Sum multi- and single-whisker, and drop the original grasp types
    grasp_counts['single'] = grasp_counts[['C1', 'C2', 'C3']].sum(1)
    grasp_counts['multi'] = grasp_counts.drop(
        ['C1', 'C2', 'C3', 'single'], axis=1).sum(1)
    grasp_counts = grasp_counts.loc[:, ['single', 'multi']].copy()
    
    
    ## Mean within session and then within mouse
    cc_by_session = contact_counts.mean(level=['task', 'mouse', 'session'])
    cc_by_mouse = contact_counts.mean(level=['task', 'mouse'])
    gc_by_session = grasp_counts.mean(level=['task', 'mouse', 'session'])
    gc_by_mouse = grasp_counts.mean(level=['task', 'mouse'])    
    
    # Concat the two kinds of metrics
    concatted = pandas.concat([cc_by_mouse, gc_by_mouse], axis=1)
    concatted.columns.name = 'metric'
    
    # Mean over mice
    mcc = concatted.mean(level='task').stack().swaplevel().sort_index()
    mcc_err = concatted.sem(level='task').stack().swaplevel().sort_index()
    
    # t-test each column (whisker/single/multi)
    pvalues = pandas.Series(
        scipy.stats.ttest_ind(
        concatted.loc['detection'], 
        concatted.loc['discrimination']).pvalue,
        index=concatted.columns)

    # Plot order
    plot_order = ['C1', 'C2', 'C3', 'single', 'multi']
    mcc = pandas.concat([mcc.loc[[ord]] for ord in plot_order])
    mcc_err = pandas.concat([mcc_err.loc[[ord]] for ord in plot_order])

    # Fig handles
    f, axa = plt.subplots(
        1, 2, figsize=(3.5, 2), sharey=True, 
        gridspec_kw={'width_ratios': [9, 6]})
    f.subplots_adjust(left=.15, bottom=.275, right=.975, top=.85)
    
    # index2plot_kwargs
    def index2plot_kwargs(idx):
        if idx['task'] == 'discrimination':
            fc = 'gray'
            ec = 'k'
        else:
            fc = 'w'
            ec = 'k'
        
        return {'fc': fc, 'ec': ec}
    
    # Plot the counts by whisker
    df1 = mcc.loc[['C1', 'C2', 'C3']]
    df1_err = mcc_err.loc[['C1', 'C2', 'C3']]
    df1.index = df1.index.remove_unused_levels()
    my.plot.grouped_bar_plot(   
        df1,
        yerrlo=(df1 - df1_err),
        yerrhi=(df1 + df1_err),
        index2plot_kwargs=index2plot_kwargs,
        ax=axa[0],
        group_name_fig_ypos=.175,
        )

    # asterisk by whisker
    for ngroup, group in enumerate(df1.index.levels[0]):
        xvals = axa[0].get_xticks()[ngroup * 2:ngroup * 2 + 2]
        axa[0].plot(xvals, [4, 4], 'k-', lw=.8)
        
        xval = np.mean(xvals)
        pvalue = pvalues.loc[group]
        sig_str = my.stats.pvalue_to_significance_string(pvalue)
        if '*' in sig_str:
            axa[0].text(xval, 4.2, sig_str, ha='center', va='center')
        else:
            axa[0].text(xval, 4.3, sig_str, ha='center', va='center', size=12)

    # Plot the counts by single/multi
    df2 = mcc.loc[['single', 'multi']]
    df2_err = mcc_err.loc[['single', 'multi']]
    df2.index = df2.index.remove_unused_levels()
    my.plot.grouped_bar_plot(   
        df2,
        yerrlo=(df2 - df2_err),
        yerrhi=(df2 + df2_err),
        index2plot_kwargs=index2plot_kwargs,
        ax=axa[1],
        group_name_fig_ypos=.175,
        )

    # asterisk by whisker
    for ngroup, group in enumerate(df2.index.levels[0]):
        xvals = axa[1].get_xticks()[ngroup * 2:ngroup * 2 + 2]
        axa[1].plot(xvals, [4, 4], 'k-', lw=.8)
        
        xval = np.mean(xvals)
        if pvalues.loc[group] < .05:
            axa[1].text(xval, 4.2, '*', ha='center', va='center')
        else:
            axa[1].text(xval, 4.3, 'n.s.', ha='center', va='center', size=12)


    ## Pretty
    for ax in axa:
        my.plot.despine(ax)
        ax.set_ylim((0, 4))
        ax.set_yticks((0, 2, 4))
        ax.set_xticks([])
    
    axa[0].set_ylabel('number of contacts')
    my.plot.despine(axa[1], which=['left'])
    axa[0].set_xlim((-1, 8))
    axa[1].set_xlim((-1, 5))
    
    
    ## Save
    f.savefig('BAR_PLOT_CONTACT_COUNTS.svg')
    f.savefig('BAR_PLOT_CONTACT_COUNTS.png', dpi=300)
    
    
    ## Stats
    with open('STATS__BAR_PLOT_CONTACT_COUNTS', 'w') as fi:
        fi.write('STATS__BAR_PLOT_CONTACT_COUNTS\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(concatted),
            len(concatted.loc['detection']),
            len(concatted.loc['discrimination']),
            ))
        fi.write('error bars: sem\n')
        
        fi.write('unpaired t-test\n')
        fi.write(pvalues.to_string() + '\n')
    
    with open('STATS__BAR_PLOT_CONTACT_COUNTS', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PLOT_TOTAL_CONTACTS_BY_STIMULUS:
    ## Sum over time and whiskers, but separate by stimulus
    # Get the contact count by trial
    this_ccb = contact_count_by_trial.copy()
    
    
    ## Mean over trials
    # Note that this includes C0
    this_ccb_otrial = this_ccb.groupby(
        ['task', 'mouse', 'stimulus', 'servo_pos'])['total'].mean()


    ## Helper plot function
    def index2plot_kwargs__shape_task__stimulus(ser):
        # The default
        res = my.plot.index2plot_kwargs__shape_task(ser)
        
        # But color by stimulus
        if 'stimulus' in ser:
            if ser['stimulus'] == 'nothing':
                res['fc'] = 'k'
            elif ser['stimulus'] == 'concave':
                res['fc'] = 'b'
            elif ser['stimulus'] == 'convex':
                res['fc'] = 'r'
        
        # Override
        if 'servo_pos' in ser:
            if ser['servo_pos'] == 1670:
                res['alpha'] = .15
            elif ser['servo_pos'] == 1760:
                res['alpha'] = .5
            elif ser['servo_pos'] == 1850:
                res['alpha'] = 1
            else:
                raise ValueError("unknown servo_pos")
        
        return res


    ## Make figure
    f, axa = my.plot.figure_1x2_standard(sharey=True,
        gridspec_kw={'width_ratios': [3, 2]})
    f.subplots_adjust(bottom=.275, left=.1, right=.975, top=.9)
    servo_pos2dist = {1670: 'far', 1760: 'med.', 1850: 'close'}
    

    ## Plot by task
    for task in task_l:
        # Slice by task
        ax = axa[task_l.index(task)]
        ax.set_title(task)
        task_this_ccb_otrial = this_ccb_otrial.loc[task]
        
        # Unstack mouse (replicates)
        topl = task_this_ccb_otrial.unstack('mouse')
        
        # Label by stimulus
        if task == 'detection':
            rewside2stimulus = {'left': 'nothing', 'right': 'something'}
        elif task == 'discrimination':
            rewside2stimulus = {'left': 'concave', 'right': 'convex'}
        else:
            1/0
    
        # Plot
        my.plot.grouped_bar_plot(
            topl,
            ax=ax,
            index2plot_kwargs=index2plot_kwargs__shape_task__stimulus,
            group_index2group_label=lambda s: s,
            index2label=lambda ser: servo_pos2dist[ser['servo_pos']],
            xtls_kwargs={'size': 'small', 'rotation': 45},
            group_name_fig_ypos=.05,
            plot_error_bars_instead_of_points=True,
        )
        
        # Pretty
        my.plot.despine(ax)
    
    # Pretty
    axa[0].set_ylabel('contacts per trial')
    my.plot.despine(axa[1], which=['left'])
    
    f.savefig('PLOT_TOTAL_CONTACTS_BY_STIMULUS.svg')


if PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY:
    ## Sum over time, but separate by stimulus and whisker
    # Get the contact count by trial
    this_ccb = contact_count_by_trial.copy()
    
    
    ## Mean over trials
    this_whisker_l = ['C1', 'C2', 'C3']
    this_ccb_otrial = this_ccb.groupby(
        ['task', 'mouse', 'stimulus', 'servo_pos'])[this_whisker_l].mean()

    # Drop 'nothing' because it's always zero
    this_ccb_otrial = this_ccb_otrial.drop('nothing', level='stimulus')
    
    # Include only discrimination
    this_ccb_otrial = this_ccb_otrial.loc['discrimination']
    this_ccb_otrial.index = this_ccb_otrial.index.remove_unused_levels()
    

    ## Make figure
    f, axa = plt.subplots(1, len(this_whisker_l),
        figsize=(6.5, 2), sharex=True, sharey=True)
    f.subplots_adjust(bottom=.225, left=.1, right=.975, top=.8,
        hspace=.4, wspace=.3)
    servo_pos2dist = {1670: 'far', 1760: 'med.', 1850: 'close'}


    ## Helper functions
    def index2plot_kwargs(ser):
        res = {}
        res['fc'] = 'gray'
        res['ec'] = 'k'
        res['lw'] = 0.8
        
        if 'servo_pos' in ser:
            if ser['servo_pos'] == 1670:
                res['alpha'] = .15
            elif ser['servo_pos'] == 1760:
                res['alpha'] = .5
            elif ser['servo_pos'] == 1850:
                res['alpha'] = 1
            else:
                raise ValueError("unknown servo_pos")
        
        return res

    ## Plot by task
    for whisker in this_whisker_l:
        # Slice by task
        ax = axa[this_whisker_l.index(whisker)]
        ax.set_title(whisker, pad=10)
        task_this_ccb_otrial = this_ccb_otrial.loc[:, whisker]

        # Unstack mouse (replicates)
        topl = task_this_ccb_otrial.unstack('mouse')
        
        # Plot
        my.plot.grouped_bar_plot(
            topl.mean(1),
            yerrhi=topl.mean(1) + topl.sem(1),
            yerrlo=topl.mean(1) - topl.sem(1),
            ax=ax,
            index2plot_kwargs=index2plot_kwargs,
            group_index2group_label=lambda s: s,
            index2label=None, #lambda ser: servo_pos2dist[ser['servo_pos']],
            xtls_kwargs={'size': 'small', 'rotation': 45},
            group_name_fig_ypos=.1,
            #~ group_name_kwargs={'size': 'small'},
            datapoint_plot_kwargs={'marker': None},
        )

        ## Pretty
        my.plot.despine(ax)
        ax.set_ylim((0, 7)) # can't get the errorbars to stay unclipped
        ax.set_yticks((0, 3, 6))
        ax.set_xticks([])
    
    
    ## Pretty
    axa[0].set_ylabel('contacts per trial')
    
    
    ## Save
    f.savefig('PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY.svg')
    f.savefig('PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_TOTAL_CONTACTS_BY_WHISKER_AND_STIMULUS_DISCONLY'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice\n'.format(
            task_this_ccb_otrial.unstack('mouse').shape[1]))
        fi.write('error bars: SEM over mice')
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

plt.show()