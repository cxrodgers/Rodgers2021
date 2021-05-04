## Descriptive plots of kappa (and duration)
"""
2F	HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY		
    STATS__HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY
    Kappa traces by whisker and duration

S2F	BAR_PLOT_KAPPA_PARAM_BY_TASK_AND_WHISKER	
    N/A
    Compare kappa params and duration by task and whisker

2I	BAR_PLOT_DURATION_BY_TASK_AND_WHISKER		
    STATS__BAR_PLOT_DURATION_BY_TASK_AND_WHISKER
    Duration of contacts by task and whisker

2G	BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER	
    STATS__BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER
    Compare kappa params by whisker, discrimination only

S2G	PLOT_DURATIONS		
    Histogram of contact durations
"""
import json
import os
import numpy as np
import pandas
import scipy.stats
import matplotlib.pyplot as plt
import whiskvid
import my
import my.plot
import matplotlib



## Fonts
my.plot.manuscript_defaults()
my.plot.font_embed()
DK = chr(916) + chr(954)


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
kp = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'kappa_parameterized'))
pck = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'peri_contact_kappa'))


## Use big_tm to slice big_ccs_df, kp, and pck
# Get included_trials
included_trials = big_tm.index

# Use those cycles to slice big_ccs_df
big_ccs_df = big_ccs_df.set_index(
    ['trial', 'whisker', 'cycle'], append=True).reorder_levels(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()
big_ccs_df = my.misc.slice_df_by_some_levels(
    big_ccs_df, C2_whisk_cycles.index)

# Return big_ccs_df to its original index
big_ccs_df = big_ccs_df.reset_index().set_index(['session', 'cluster'])

# Use those clusters to slice kp and pck
slicing_idx = pandas.MultiIndex.from_frame(
    big_ccs_df.index.to_frame().reset_index(drop=True))
kp = my.misc.slice_df_by_some_levels(kp, slicing_idx)
pck = my.misc.slice_df_by_some_levels(pck, slicing_idx)


## Append the whisker level to kp and pck
# This would be where to append "locked_t" to filter by contact time wrt rwin
whisker_ser = big_ccs_df['whisker']
pck.index = pandas.MultiIndex.from_frame(
    pck.index.to_frame().reset_index(drop=True).join(
    whisker_ser, on=['session', 'cluster'])).reorder_levels(
    ['duration_ceil', 'duration', 'session', 'whisker', 'cluster'])
pck = pck.sort_index()
kp.index = pandas.MultiIndex.from_frame(
    kp.index.to_frame().reset_index(drop=True).join(
    whisker_ser, on=['session', 'cluster'])).reorder_levels(
    ['session', 'whisker', 'cluster'])
kp = kp.sort_index()


## Insert mouse and task levels
pck = my.misc.insert_mouse_and_task_levels(
    pck, mouse2task)
kp = my.misc.insert_mouse_and_task_levels(
    kp, mouse2task)
    

## Plots
HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY = True
BAR_PLOT_KAPPA_PARAM_BY_TASK_AND_WHISKER = True
BAR_PLOT_DURATION_BY_TASK_AND_WHISKER = True
BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER = True
PLOT_DURATIONS = True


## Plot the mean trace, grouped by task, whisker, and duration
if HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY:
    ## Aggregate
    # Rebin duration more coarsely
    duration_ceil = pck.index.get_level_values('duration_ceil')
    duration_ceil2 = duration_ceil // 2
    
    # Insert level
    pck2 = pck.copy()
    idx = pck2.index.to_frame().reset_index(drop=True)
    idx['duration_ceil'] = duration_ceil2
    pck2.index = pandas.MultiIndex.from_frame(idx)
    pck2 = pck2.sort_index()
    
    # Mean
    meaned = pck2.mean(
        level=['task', 'mouse', 'duration_ceil', 'whisker'])
    meaned.columns.name = 'lag'
    
    # Size
    size = pck2.groupby(
        ['task', 'mouse', 'duration_ceil', 'whisker']).size()
    
    
    ## Drop C0 for now
    meaned = meaned.drop('C0', level='whisker')
    size = size.drop('C0', level='whisker')


    ## Let's just drop detection because it's noisy
    meaned = meaned.drop('detection').droplevel('task')
    size = size.drop('detection').droplevel('task')

    
    ## Determine which bins we have sufficient data for
    # There are mouse * whisker * duration_ceil that have no data, so either
    # way there is going to be missing data somewhere.
    # Discarding bins with <10 contacts leaves the plot virtually unaffected
    # (maybe slightly decreases noise on long contacts)
    # And is perhaps congruent with the way other parts of the same figure
    # were handled
    meaned = meaned.loc[size >= 10].copy()
    
    
    ## Get the list of duration bins
    # Extract
    duration_ceil_l = sorted(np.unique(
        pck2.index.get_level_values('duration_ceil')))
    
    # Exclude last because it contains all the long ones
    duration_ceil_l = duration_ceil_l[:-1]
    
    # Actually just stop after 60 ms
    duration_ceil_l = [dc for dc in duration_ceil_l if dc <=6]
  
    # Include only these durations
    meaned = meaned[
        meaned.index.get_level_values('duration_ceil').isin(duration_ceil_l)
        ].copy()
    
    
    ## Plot
    grouping_keys = ['duration_ceil', 'whisker']
    whisker_l = ['C1', 'C2', 'C3']
    f, axa = plt.subplots(
        len(whisker_l), len(duration_ceil_l), 
        figsize=(8.8, 2), sharex=True, sharey=True)
    f.subplots_adjust(
        left=.1, right=.9, bottom=.01, top=.8, hspace=0, wspace=.1)


    ## Iterate over duration_ceil * whisker (axes)
    for (duration_ceil, whisker), sub_meaned in meaned.groupby(['duration_ceil', 'whisker']):
        # Droplevel
        sub_meaned = sub_meaned.droplevel(['duration_ceil', 'whisker'])
        
        # Make mouse into a replicate column
        sub_meaned = sub_meaned.T
        
        # Get ax
        ax = axa[
            whisker_l.index(whisker),
            duration_ceil_l.index(duration_ceil), 
            ]
        
        # Mean over mice
        mean_over_mice = sub_meaned.mean(1).sort_index()
        err_over_mice = sub_meaned.sem(1).sort_index()
        
        # Put xvals into ms
        xvals = mean_over_mice.index.values * 5

        # Plot
        ax.plot(xvals, mean_over_mice, color='k', lw=1, clip_on=False)
        ax.fill_between(
            x=xvals,
            y1=(mean_over_mice - err_over_mice).values,
            y2=(mean_over_mice + err_over_mice).values,
            color='k', alpha=.25, lw=0, clip_on=False)

        # Plot baseline
        ax.plot(xvals, [0] * len(xvals), 'k--', lw=.8, alpha=.5)
        
        
        ## Pretty
        my.plot.despine(ax, which=('left', 'bottom', 'right', 'top'))
        
        if ax in axa[0, :]:
            # Label the duration
            # A bit weird because the first bin contains 5 and 10 ms
            # contacts, and we're calling it 10, I guess
            ax.set_title((duration_ceil + 1) * 10, size=12)
        if ax in axa[:, 0]:
            ax.set_ylabel(whisker, rotation=0, labelpad=10, size=12)
        
        ax.set_yticks([])
        ax.set_xticks([])
        
        # Limits
        ylim = (-.015, .02)
        ax.set_xlim((-30, 100))
        ax.set_ylim(ylim)


        ## Duration shade
        duration = (duration_ceil + 1) * 10 - 2.5 # the average, kind of
        ax.fill_between(
            x=[0, duration], 
            y1=[ylim[0], ylim[0]],
            y2=[ylim[1], ylim[1]],
            color='pink', alpha=.25, lw=0, clip_on=True, zorder=0)
        
    
    ## Pretty
    # Axis labels
    f.text(
        .55, .97, 'contact duration (ms)', 
        ha='center', va='center')
    f.text(
        .04, .5, 'bending ({}) of the\ncontacting whisker'.format(DK), 
        ha='center', va='center', rotation=90)
    
    #~ f.text(.95, .95, 'n = {} mice'.format(
    
    ## Scale bar
    t_start = 150
    t_len = 50
    y_start = 0
    y_len = .020
    
    # kappa scale bar
    axa[1, -1].plot(
        [t_start, t_start], [y_start, y_len], 
        'k-', clip_on=False, lw=1)
    axa[1, -1].text(
        t_start - 8, y_start + y_len / 2 + .005, '{} m-1'.format(int(y_len * 1000)), 
        ha='right', va='center', rotation=90, clip_on=False, size=12)

    # time scale bar
    axa[1, -1].plot(
        [t_start, t_start + t_len], [y_start, y_start], 
        'k-', clip_on=False, lw=1)
    axa[1, -1].text(
        t_start + t_len / 2, y_start - .004, '{} ms'.format(t_len),
        ha='center', va='top', clip_on=False, size=12)


    ## Save
    f.savefig('HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY.svg')


    ## Stats
    # Count mice
    n_mice_per_subplot = meaned.groupby(
        ['duration_ceil', 'whisker']).size().unstack().T
    
    with open('STATS__HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY', 'w') as fi:
        fi.write('STATS__HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY\n')
        fi.write('Including discrimination mice only (max: n = {})\n'.format(
            n_mice_per_subplot.max().max()))
        fi.write('error bars: sem\n')
        fi.write('n per duration and whisker:\n')
        fi.write(n_mice_per_subplot.to_string() + '\n')
    
    with open('STATS__HORIZONTAL_GRID_PLOT_MEANED_BY_WHISKER_AND_DURATION__DISC_ONLY', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


## Parameterize contacts by task * whisker
if BAR_PLOT_KAPPA_PARAM_BY_TASK_AND_WHISKER:
    # First mean within mouse
    meaned_within_mouse = kp.groupby(['task', 'mouse', 'whisker']).mean()
    
    # Drop C0 for now
    meaned_within_mouse = meaned_within_mouse.drop('C0', level='whisker')
    
    # Convert frames to ms
    meaned_within_mouse['duration'] = meaned_within_mouse['duration'] * 5

    # Convert curvature to 1/m
    meaned_within_mouse[['max', 'min', 'std']] *= 1000
    
    # Now mean and error accross mice
    mean_param = meaned_within_mouse.mean(level=['task', 'whisker'])
    err_param = meaned_within_mouse.sem(level=['task', 'whisker'])
    param_l = list(mean_param.columns)
    
    # Plotting helper functions
    def index2plot_kwargs(ser):
        if ser['task'] == 'detection':
            color = 'r'
        elif ser ['task'] == 'discrimination':
            color = 'gray'
            
        res = {'ec': 'k', 'fc': color, 'alpha': .5}
        return res

    def group_index2group_label(group_index):
        return {'detection': 'det.', 'discrimination': 'disc.'}[group_index]
    
    kappa_param_l = ['max', 'min', 'std']
    param2pretty_title = {
        'duration': 'duration (ms)',
        'max': 'max({}) (m-1)'.format(DK),
        'min': 'min({}) (m-1)'.format(DK),
        'std': 'std({}) (m-1)'.format(DK),
    }
       
    f, axa = plt.subplots(1, len(kappa_param_l), figsize=(8, 2), sharex=True)
    f.subplots_adjust(left=.1, right=.98, wspace=.7, bottom=.2, top=.875)
    for param in kappa_param_l:
        # Get ax
        ax = axa[kappa_param_l.index(param)]
        ax.set_ylabel(param2pretty_title[param])
        
        # Slice
        mean_topl = mean_param[param]
        err_topl = err_param[param]
        
        # Ordering
        mean_topl = mean_topl.reindex(['discrimination', 'detection'], level=0)
        err_topl = err_topl.reindex(['discrimination', 'detection'], level=0)
        
        # Plot
        my.plot.grouped_bar_plot(
            mean_topl,
            yerrlo=(mean_topl - err_topl),
            yerrhi=(mean_topl + err_topl),
            index2plot_kwargs=index2plot_kwargs,
            index2label=(lambda ser: ser['whisker']),
            group_index2group_label=lambda s: '',
            ax=ax,
            group_name_fig_ypos=.05
        )
        
        
        ## Pretty
        my.plot.despine(ax)
        ax.set_xticklabels(ax.get_xticklabels(), size='small')
        
    
    
    ## Pretty
    axa[kappa_param_l.index('min')].set_ylim((0, -30))
    axa[kappa_param_l.index('min')].set_yticks((0, -10, -20, -30))
    axa[kappa_param_l.index('max')].set_ylim((0, 30))
    axa[kappa_param_l.index('max')].set_yticks((0, 10, 20, 30))
    axa[kappa_param_l.index('std')].set_ylim((0, 30))
    axa[kappa_param_l.index('std')].set_yticks((0, 10, 20, 30))
    
    f.savefig('BAR_PLOT_KAPPA_PARAM_BY_TASK_AND_WHISKER.svg')
    f.savefig('BAR_PLOT_KAPPA_PARAM_BY_TASK_AND_WHISKER.png', dpi=300)


if BAR_PLOT_DURATION_BY_TASK_AND_WHISKER:
    ## Select only data
    duration_data = kp['duration'].copy()
    
    
    ## Aggregate
    # First mean within mouse
    meaned_within_mouse = duration_data.groupby(
        ['task', 'mouse', 'whisker']).mean()
    
    # Drop C0 for now
    meaned_within_mouse = meaned_within_mouse.drop('C0', level='whisker')
    
    # Convert frames to ms
    meaned_within_mouse *= 5
    
    # Unstack whisker
    meaned_within_mouse = meaned_within_mouse.unstack('whisker')
    
    # Drop mice missing data (currently 234CR missing C3)
    meaned_within_mouse = meaned_within_mouse.dropna()
    
    # Now mean and error accross mice
    mean_param = meaned_within_mouse.mean(
        level='task').stack().swaplevel().sort_index()
    err_param = meaned_within_mouse.sem(
        level='task').stack().swaplevel().sort_index() 

    
    ## t-test each whisker
    pvalues = pandas.Series(
        scipy.stats.ttest_ind(
        meaned_within_mouse.loc['detection'], 
        meaned_within_mouse.loc['discrimination'],
        ).pvalue, index=meaned_within_mouse.columns)


    ## Bar plot
    # Plotting helper functions
    def index2plot_kwargs(ser):
        if ser['task'] == 'detection':
            ec = 'k'
            fc = 'w'
        else:
            ec = 'k'
            fc = 'gray'
        
        res = {'ec': ec, 'fc': fc}
        return res

    f, ax = my.plot.figure_1x1_small()
    f.subplots_adjust(bottom=.275, right=.975, top=.85)
    
    my.plot.grouped_bar_plot(
        mean_param,
        yerrlo=(mean_param - err_param),
        yerrhi=(mean_param + err_param),
        index2plot_kwargs=index2plot_kwargs,
        index2label=None,
        group_index2group_label=None,
        ax=ax,
        group_name_fig_ypos=.175,
    )

    
    ## asterisk by whisker
    yval_line = 60
    yval_txt = 63
    for ngroup, group in enumerate(mean_param.index.levels[0]):
        xvals = ax.get_xticks()[ngroup * 2:ngroup * 2 + 2]
        ax.plot(xvals, [yval_line, yval_line], 'k-', lw=.8)
        
        xval = np.mean(xvals)
        if pvalues.loc[group] < .05:
            ax.text(xval, yval_txt, '*', ha='center', va='center')
        else:
            ax.text(xval, yval_txt, 'n.s.', ha='center', va='center', size=12)    
    
    
    ## Pretty
    my.plot.despine(ax)
    ax.set_xticks([])
    ax.set_ylim((0, 60))
    ax.set_yticks((0, 20, 40, 60))
    ax.set_ylabel('duration (ms)')
    
    
    ## Save
    f.savefig('BAR_PLOT_DURATION_BY_TASK_AND_WHISKER.svg')
    f.savefig('BAR_PLOT_DURATION_BY_TASK_AND_WHISKER.png', dpi=300)


    ## Stats
    with open('STATS__BAR_PLOT_DURATION_BY_TASK_AND_WHISKER', 'w') as fi:
        fi.write('STATS__BAR_PLOT_DURATION_BY_TASK_AND_WHISKER\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(meaned_within_mouse),
            len(meaned_within_mouse.loc['detection']),
            len(meaned_within_mouse.loc['discrimination']),
            ))
        fi.write('error bars: sem\n')
        
        fi.write('unpaired t-test\n')
        fi.write(pvalues.to_string())
    
    with open('STATS__BAR_PLOT_DURATION_BY_TASK_AND_WHISKER', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))
    

if BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER:
    ## Discrimination only, exclude minimum for clarity
    # First mean within mouse
    meaned_within_mouse = kp.groupby(['task', 'mouse', 'whisker']).mean()
    
    # Drop C0 for now
    meaned_within_mouse = meaned_within_mouse.drop('C0', level='whisker')
    
    # Convert frames to ms
    meaned_within_mouse['duration'] = meaned_within_mouse['duration'] * 5

    # Convert curvature to 1/m
    meaned_within_mouse[['max', 'min', 'std']] *= 1000

    # Discrimination only
    meaned_within_mouse = meaned_within_mouse.loc['discrimination']
    
    # duation, max, and std only
    param_l = ['min', 'max', 'std']
    meaned_within_mouse = meaned_within_mouse.loc[:, param_l]
    
    # Now mean and error accross mice
    mean_param = meaned_within_mouse.mean(level='whisker')
    err_param = meaned_within_mouse.sem(level='whisker')
    
    # Plotting helper functions
    def index2plot_kwargs(ser):
        res = {'ec': 'k', 'fc': 'w'}
        return res

    # Pretty titles
    param2pretty_title = {
        'duration': 'duration (ms)',
        'max': 'max({}) (m-1)'.format(DK),
        'min': 'min({}) (m-1)'.format(DK),
        'std': 'std({}) (m-1)'.format(DK),
    }
    
    # Handles
    f, axa = plt.subplots(1, len(param_l), figsize=(5.5, 2), sharex=True)
    f.subplots_adjust(left=.12, right=.98, wspace=.8, bottom=.275, top=.85)
    
    
    # Iterate over params
    for param in param_l:
        # Get ax
        ax = axa[param_l.index(param)]
        ax.set_ylabel(param2pretty_title[param])
        
        # Slice
        mean_topl = mean_param[param]
        err_topl = err_param[param]

        # Plot
        my.plot.grouped_bar_plot(
            mean_topl,
            yerrlo=(mean_topl - err_topl),
            yerrhi=(mean_topl + err_topl),
            index2plot_kwargs=index2plot_kwargs,
            ax=ax,
            group_name_fig_ypos=.05
        )
        
        # Pretty
        my.plot.despine(ax)
        ax.set_xticks(range(len(mean_topl)))
        ax.set_xticklabels(mean_topl.index.values)

    
    ## Pretty
    axa[param_l.index('max')].set_ylim((0, 30))
    axa[param_l.index('max')].set_yticks((0, 10, 20, 30))
    axa[param_l.index('std')].set_ylim((0, 30))
    axa[param_l.index('std')].set_yticks((0, 10, 20, 30))
    axa[param_l.index('min')].set_ylim((0, -30))
    axa[param_l.index('min')].set_yticks((0, -10, -20, -30))
    
    f.text(.525, .99, 'quantifications of whisker bending', ha='center', va='center')
    
    f.savefig('BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER.svg')
    f.savefig('BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER.png', dpi=300)

    ## Stats
    unstacked = meaned_within_mouse.unstack('whisker')
    
    with open('STATS__BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER', 'w') as fi:
        fi.write('STATS__BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER\n')
        fi.write('n = {} mice (discrimination only)\n'.format(
            len(unstacked),
            ))
        fi.write('error bars: sem\n\n')
        
        fi.write('mean values:\n' + unstacked.mean().unstack('whisker').to_string() + '\n\n')
        fi.write('SEM values:\n' + unstacked.sem().unstack('whisker').to_string() + '\n\n')
        
    
    with open('STATS__BAR_PLOT_DISCRIMINATION_KAPPA_PARAM_BY_WHISKER', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))
    

if PLOT_DURATIONS:
    ## Histogram and plot durations
    f, ax = my.plot.figure_1x1_standard()
    duration_counts, duration_edges = np.histogram(
        kp['duration'].values, bins=list(range(1, 21)) + [np.inf])
    duration_prop = duration_counts / float(duration_counts.sum())
   
    #~ kp2 = kp.copy()
    #~ kp2['duration'] *= 5 # convert to ms
    #~ duration_bins = np.concatenate([np.arange(0, 100, 5), [np.inf]])
    #~ duration_starts = duration_bins[:-1]
    #~ kp2['duration_bin'] = pandas.cut(
        #~ kp2['duration'], duration_bins, labels=False, right=False)
    #~ assert not kp2['duration_bin'].isnull().any()
    #~ duration_counts = kp2['duration_bin'].groupby(
        #~ ['task', 'whisker']).value_counts().unstack(
        #~ ['task', 'whisker']).reindex(range(len(duration_starts))).fillna(0)
    #~ duration_counts.index = duration_starts
    #~ duration_counts = duration_counts.divide(duration_counts.sum())
   
    ax.plot(duration_edges[:-1] * 5, duration_prop, color='k', clip_on=False)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(['0', '25', '50', '75', '100+'])
    ax.set_xlabel('contact duration (ms)')
    ax.set_ylabel('proportion')
    ax.set_ylim((0, .2))
    ax.set_yticks((0, .1, .2))
    my.plot.despine(ax)

    ## Save
    f.savefig('PLOT_DURATIONS.svg')
    f.savefig('PLOT_DURATIONS.png', dpi=300)

    
    ## Stats
    with open('STATS__PLOT_DURATIONS', 'w') as fi:
        data_ms = kp['duration'].values * 5
        
        fi.write('n = {} contacts\n'.format(len(data_ms)))
        fi.write('duration mean: {:.3}\n'.format(data_ms.mean()))
        
        tiles = np.percentile(data_ms, [25, 50, 75])
        
        fi.write('duration median: {}\n'.format(tiles[1]))
        fi.write('duration .25 tile: {}\n'.format(tiles[0]))
        fi.write('duration .75 tile: {}\n'.format(tiles[2]))
    
    with open('STATS__PLOT_DURATIONS', 'r') as fi:
        print(''.join(fi.readlines()))


plt.show()
