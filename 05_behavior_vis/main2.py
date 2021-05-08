## This directory is for making simple plots/parameterizations of the patterns
# This script is for basic statistics of whisking
# TODO: get the "n_frames" count working, somehow
"""
2A
    STATS__TOPLINE_WHISK_PARAMS
    Topline whisk parameters, like count and amplitude

S2E
    PLOT_HISTOGRAM_OF_AMPLITUDE
    Histograms of whisk amplitudes for individual mice
"""


import json
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import my
import my.plot
import whiskvid
import runner.models


## Plot flags
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)



## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(params, 
    'big_tm')
C2_whisk_cycles = my.dataload.load_data_from_patterns(params, 
    'big_C2_tip_whisk_cycles')
big_cycle_features = my.dataload.load_data_from_patterns(params, 
    'big_cycle_features')
    

## Count trials
trial_count = big_tm.groupby('session').size()
session_df['n_trials'] = trial_count
trial_count = my.misc.insert_mouse_and_task_levels(trial_count, mouse2task)


## Count frames
n_frames_l = []
for session_name in session_df.index:
    gs = runner.models.GrandSession.objects.filter(name=session_name).first()
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    n_frames = int(np.rint(my.video.get_video_duration(
        vs.data.monitor_video.get_path) * 30))
    n_frames_l.append(n_frames)
session_df['n_frames'] = n_frames_l


## Extract performance to correlate with these whisking metrics
big_tm = my.misc.insert_mouse_level(big_tm)
outcomes = big_tm.groupby(['mouse', 'session'])[
    'outcome'].value_counts().unstack()
mouse_perf = (outcomes['hit'] / outcomes.sum(1)).mean(
    level=['mouse', 'session']).mean(level='mouse')
mouse_perf_and_task = pandas.concat(
    [mouse_perf.rename('perf'), mouse2task], axis=1, sort=True).sort_values(
    ['task', 'perf'])
mouse_perf_and_task.index.name = 'mouse'


## Extract cycle_stats from C2_whisk_cycles and big_cycle_features
# Keep all global C2 whisk cycle features
# Dropping the timing-related columns, except peak_frame_wrt_rwin,
# which I'll use to lock everything up
panwhisker_cycle_features = C2_whisk_cycles.drop(
    ['start_frame_wrt_rwin', 'rwin_frame', 'peak_frame', 
    'start_frame', 'stop_frame'], axis=1)
panwhisker_cycle_features.columns = pandas.MultiIndex.from_tuples(
    [(metric, 'all') for metric in panwhisker_cycle_features.columns],
    names=['metric', 'whisker'])

# Per-whisker features like start and peak angle
perwhisker_cycle_features = big_cycle_features.loc[:, 
    ['start_tip_angle', 'peak_tip_angle']].rename(
    columns={
        'start_tip_angle': 'start_angle', 
        'peak_tip_angle': 'peak_angle', 
    })

# Concat panwhisker and perwhisker
cycle_stats = pandas.concat(
    [perwhisker_cycle_features, panwhisker_cycle_features], 
    axis=1).sort_index(axis=1)
assert not cycle_stats.isnull().any().any()


## Bin cycle_stats by the frame time
# Define frame bins
frame_bin_edges = np.linspace(-400, 200, 31)
frame_bin_centers = (frame_bin_edges[:-1] + frame_bin_edges[1:]) / 2.0

# Cut peak_frame_wrt_rwin by frame_bins
cycle_stats.loc[:, ('frame_bin', 'all')] = pandas.cut(
    cycle_stats.loc[:, ('peak_frame_wrt_rwin', 'all')], 
    bins=frame_bin_edges, labels=False, right=False)

# Drop cycles outside of the range defined by frame_bin_edges
cycle_stats = cycle_stats.loc[
    ~cycle_stats.loc[:, ('frame_bin', 'all')].isnull()].copy()
cycle_stats.loc[:, ('frame_bin', 'all')] = (
    cycle_stats.loc[:, ('frame_bin', 'all')].astype(np.int))

# Make frame_bin a level on the index
frame_bin_column = cycle_stats.pop(('frame_bin', 'all'))
midx = cycle_stats.index.to_frame()
midx.insert(midx.shape[1], 'frame_bin', frame_bin_column)
cycle_stats.index = pandas.MultiIndex.from_frame(midx.reset_index(drop=True))

# Add mouse and task levels
cycle_stats = my.misc.insert_mouse_level(cycle_stats)
cycle_stats = my.misc.insert_level(
    cycle_stats, name='task', 
    func=lambda idx: idx['mouse'].map(mouse2task))


## Identify big cycles
cycle_stats.loc[:, ('ampl_gt1deg', 'all')] = (
    cycle_stats.loc[:, ('amplitude', 'all')] > 1)

# Convert o ms
cycle_stats.loc[:, ('duration_ms', 'all')] = cycle_stats.loc[:, ('duration', 'all')] * 5


## Mean cycle_stats over trials, versus time
# Mean the various metrics across trials
trial_meaned_stats = cycle_stats.groupby(
    ['task', 'mouse', 'session', 'frame_bin']).mean().drop(
    ['peak_frame_wrt_rwin'], axis=1)

# Count the number of cycles in each task*mouse*session*frame_bin
level_names = list(cycle_stats.index.names)
n_cycles_per_frame_bin = cycle_stats.groupby(
    [lev for lev in level_names if lev not in ['trial', 'cycle']]).size()
n_big_cycles_per_frame_bin = cycle_stats[
    cycle_stats['ampl_gt1deg'].values].groupby(
    [lev for lev in level_names if lev not in ['trial', 'cycle']]).size()

# Divide this count by the number of trials in each task * mouse * session
trial_meaned_n_cycles = n_cycles_per_frame_bin.divide(trial_count)
trial_meaned_n_big_cycles = n_big_cycles_per_frame_bin.divide(trial_count)

# Append these cycle rates to trial_meaned_stats
trial_meaned_stats.loc[:, ('cycle_rate', 'all')] = (
    trial_meaned_n_cycles)
trial_meaned_stats.loc[:, ('big_cycle_rate', 'all')] = (
    trial_meaned_n_big_cycles)


## Mean over mice
# Mean over sessions
mouse_meaned_stats = trial_meaned_stats.mean(
    level=['task', 'mouse', 'frame_bin']).unstack(
    'frame_bin')
    

## Plots
STATS__TOPLINE_WHISK_PARAMS = True
PLOT_HISTOGRAMS_OF_WHISKING_PARAMS_OVERLAID_BY_TASK = True

if STATS__TOPLINE_WHISK_PARAMS:
    ## Separately mean NOT over time for some top-level stats
    # Mean over all cycles for each task * mouse
    cycle_stats_meaned_over_cycles = cycle_stats.mean(level=['task', 'mouse'])

    # Keep only things that make sense
    cycle_stats_meaned_over_cycles = cycle_stats_meaned_over_cycles.loc[:, [
        ('amplitude', 'all'),
        ('duration', 'all'),
        ('inst_frequency', 'all'),
        ('ampl_gt1deg', 'all'),
        ('C1vC2', 'all'),
        ('C3vC2', 'all'),
        ]]
    cycle_stats_meaned_over_cycles.columns = (
        cycle_stats_meaned_over_cycles.columns.droplevel('whisker'))

    # Convert some stuff to better units
    cycle_stats_meaned_over_cycles['duration_ms'] = (
        cycle_stats_meaned_over_cycles['duration'] * 5)

    # These are indistinguishable by task so pool
    # Aggregate by mouse
    topline_stats_mean_by_task = cycle_stats_meaned_over_cycles.mean(level='task')
    topline_stats_sem_by_task = cycle_stats_meaned_over_cycles.sem(level='task')
    topline_stats_mean_pooled = cycle_stats_meaned_over_cycles.mean()
    topline_stats_sem_pooled = cycle_stats_meaned_over_cycles.sem()
    topline_stats_std_pooled = cycle_stats_meaned_over_cycles.std()

    # Dump topline stats
    with open('STATS__TOPLINE_WHISK_PARAMS', 'w') as fi:
        n_mice_pooled = cycle_stats_meaned_over_cycles.shape[0]
        n_cyles_pooled = cycle_stats.shape[0]
        
        fi.write(
            'n = {} mice, {:.1f} hours, {} sessions, {} trials, {} frames\n'.format(
            n_mice_pooled,
            session_df['n_frames'].sum() / 200. / 3600.,
            len(session_df),
            session_df['n_trials'].sum(),
            session_df['n_frames'].sum(),
            ))
        fi.write('excluding opto from trial count\n')
        fi.write('n = {} cycles\n'.format(n_cyles_pooled))
        fi.write('including only cycles within (-2, 1) of rwin\n')
        fi.write('excluding opto from cycle count\n')
        fi.write('meaning first within mice and then across mice\n')
        
        for metric in topline_stats_mean_pooled.index:
            fi.write('{:16}: mean {:.4f}, sem {:.4f}, std {:.4f}\n'.format(
                metric,
                topline_stats_mean_pooled.loc[metric],
                topline_stats_sem_pooled.loc[metric],
                topline_stats_std_pooled.loc[metric],
                ))
        
        fi.write('mean duration converted to Hz: {:.2f}\n'.format(
            1000. / topline_stats_mean_pooled.loc['duration_ms']))

    with open('STATS__TOPLINE_WHISK_PARAMS', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PLOT_HISTOGRAMS_OF_WHISKING_PARAMS_OVERLAID_BY_TASK:
    # One subplot per metric
    metric_l = ['amplitude', 'protraction_speed', 'duration_ms']

    f, axa = plt.subplots(1, len(metric_l), figsize=(7.5, 2.2))
    f.subplots_adjust(left=.13, right=.925, wspace=.6, bottom=.25, top=.875)
    
    for metric in metric_l:
        
        ## Slice ax
        ax = axa[metric_l.index(metric)]

        
        ## Choose bins
        binned_key = metric + '_binned'
        if metric == 'amplitude':
            # Note that zero-amplitude bins will be null
            bins = np.logspace(-1, 2, 31)
            xticklabels = [0.1, 1, 10, 100]
            loglabels = True
            xlabel = 'whisk amplitude ({})'.format(chr(176))
        
        elif metric == 'protraction_speed':
            # Note that zero-amplitude bins will be null
            bins = np.logspace(0, 3.5, 31)
            xticklabels = [1, 10, 100, 1000,]
            loglabels = True
            xlabel = 'protraction speed ({}/s)'.format(chr(176))

        elif metric == 'duration_ms':
            bins = np.linspace(0, 150, 31)
            xticklabels = np.linspace(0, 150, 4).astype(np.int)
            loglabels = False
            xlabel = 'duration (ms)'
    
        else:
            1/0
            
        bin_centers = (bins[1:] + bins[:-1]) / 2.0
        
        
        ## Histogram
        # Bin
        cycle_stats.loc[:, (binned_key, 'all')] = pandas.cut(
            cycle_stats.loc[:, (metric, 'all')], 
            bins=bins, right=False, labels=False)

        # Count by bin, discarding values outside the range
        bin_counts = cycle_stats.loc[:, (binned_key, 'all')].rename(
            binned_key).groupby(['task', 'mouse', 'session']).value_counts(
            ).unstack().fillna(0).astype(np.int)
        bin_counts.columns = bin_counts.columns.astype(np.int)

        # Normalize
        bin_counts = bin_counts.divide(bin_counts.sum(1), axis=0)

        # Mean over sessions within mouse
        mouse_meaned_bin_counts = bin_counts.mean(level=['task', 'mouse'])

        
        ## Plot
        task_l = ['detection', 'discrimination']

        # Iterate over task (axis)
        for task in task_l:
            if task == 'detection':
                color = 'r'
                ls = '-'
            elif task == 'discrimination':
                color = 'gray'
                ls = '-'
            
            # Plot
            topl = mouse_meaned_bin_counts.loc[task].T
            xvals = bin_centers[topl.index.values].copy()
            if loglabels:
                xvals = np.log10(xvals)
            
            cbar = my.plot.generate_colorbar(topl.shape[1])
            for ncol, col in enumerate(topl.columns):
                ax.plot(xvals, topl[col], lw=1, color=color, ls=ls, alpha=.5)
        
        
        ## xticks
        if loglabels:
            ax.set_xticks(np.log10(xticklabels))
        else:
            ax.set_xticks(xticklabels)
        
        ax.set_xticklabels(xticklabels)

        
        ## Pretty
        my.plot.despine(ax)
        ax.set_ylim((0, .15))
        ax.set_xlabel(xlabel)


    ## Pretty
    axa[0].set_ylabel('fraction of whisks')

    n_detection = len(mouse_meaned_bin_counts.loc['detection'])
    n_discrimination = len(mouse_meaned_bin_counts.loc['discrimination'])
    f.text(.99, .88, 'n = {} mice (detection)'.format(n_detection), 
        color='r', ha='right', va='center', size=12)
    f.text(.99, .95, 'n = {} mice (discrimination)'.format(n_discrimination), 
        color='gray', ha='right', va='center', size=12)

    
    f.savefig('PLOT_HISTOGRAMS_OF_WHISKING_PARAMS_OVERLAID_BY_TASK.svg')
    f.savefig('PLOT_HISTOGRAMS_OF_WHISKING_PARAMS_OVERLAID_BY_TASK.png', dpi=300)
   
plt.show()