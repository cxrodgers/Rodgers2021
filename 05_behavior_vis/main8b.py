## General plots of performance

"""
1F
    PLOT_PERF_BY_TASK
    STATS__PLOT_PERF_BY_TASK
    Performance by task

1G
    PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS
    STATS__PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS
    Performance by stimulus, position, and task
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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM


## Fonts
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
big_tm = my.dataload.load_data_from_patterns(
    params, 'big_tm', mouse2task=mouse2task)

# Add correct
big_tm['correct'] = big_tm['outcome'] == 'hit'


## Plot flags
PLOT_PERF_BY_TASK = True
PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS = True


## Plots
if PLOT_PERF_BY_TASK:
    task_l = ['detection', 'discrimination']
    
    # Mean performance by task * mouse * rewside * servo_pos
    perfdf = big_tm.groupby(['task', 'mouse'])['correct'].mean()

    # Create figure
    f, ax = plt.subplots(figsize=(4, 2.25))
    f.subplots_adjust(bottom=.25, right=.8, top=.95, left=.35)
    
    for task in task_l:
        tickloc = 0 if task == 'detection' else 1
        
        # Plot bar to mean
        ax.barh( 
            y=[tickloc],
            width=[perfdf.loc[task].mean()],
            height=.5,
            fc='none',
            ec='k',
            lw=1,
        )
    
        # Plot individual mice
        ax.plot(
            perfdf.loc[task].values,
            [tickloc] * len(perfdf.loc[task]),
            marker='o', ms=4, mfc='none', mec='k', ls='none', mew=.8)
    
    # Text n
    ax.text(1.25, 1, 'n = {}'.format(len(perfdf.loc['discrimination'])), ha='center', va='center')
    ax.text(1.25, 0, 'n = {}'.format(len(perfdf.loc['detection'])), ha='center', va='center')
    ax.text(-.4, 1, 'discrimination', ha='center', va='center')
    ax.text(-.4, 0, 'detection', ha='center', va='center')

    # Pretty
    my.plot.despine(ax)
    ylim = [-.6, 1.6]
    ax.plot([.5, .5], ylim, 'k--', lw=.8)
    ax.set_ylim(ylim)
    ax.set_xlim((0, 1))
    ax.set_xticks((0, .25, .5, .75, 1))    
    ax.set_xticklabels(('0.0', '', '0.5', '', '1.0'))
    ax.set_yticks((0, 1))
    #~ ax.set_yticklabels(task_l, ha='center', labelpad=30)
    ax.set_yticklabels([])
    ax.set_xlabel('performance')
    
    f.savefig('PLOT_PERF_BY_TASK.svg')
    f.savefig('PLOT_PERF_BY_TASK.png', dpi=300)
    
    
    ## Stats
    with open('STATS__PLOT_PERF_BY_TASK', 'w') as fi:
        fi.write('STATS__PLOT_PERF_BY_TASK\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(perfdf),
            len(perfdf.loc['detection']),
            len(perfdf.loc['discrimination']),
            ))
    
    with open('STATS__PLOT_PERF_BY_TASK', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

    
if PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS:
    task_l = ['discrimination', 'detection']
    
    # This is to make detection orange/purple
    def index2plot_kwargs__detection(ser):
        res = my.plot.index2plot_kwargs__shape_task(ser)

        if 'rewside' in ser:
            if ser['rewside'] == 'left':
                res['fc'] = 'orange'
            elif ser['rewside'] == 'right':
                res['fc'] = 'purple'
            else:
                raise ValueError("unknown rewside")
                
        return res

    # Mean performance by task * mouse * rewside * servo_pos
    perfdf = big_tm.groupby(['task', 'mouse', 'rewside', 'servo_pos']
        )['correct'].mean().unstack(['task', 'mouse'])

    
    ## Run stats
    # One-way repeated-measures anova, separately on each task and rewside
    # Iterate over task
    stats_l = []
    stats_keys_l = []
    for task in task_l:
        # Get data for task
        task_perfdf = perfdf.loc[:, task]
        
        # Iterate over rewside
        for rewside in ['left', 'right']:
            # Get data for rewside
            this_perfdf = task_perfdf.loc[rewside]
            
            # Tableize
            data_table = this_perfdf.stack().rename('perf').reset_index()
            
            # OLS
            formula = 'perf ~ servo_pos + mouse'
            lm = ols(formula, data_table)
            lm_res = lm.fit()
            
            # Anova
            aov = AnovaRM(
                data_table, depvar='perf', subject='mouse', 
                within=['servo_pos'],
                )            
            aov_res = aov.fit()
            
            # Not sure which p-value to take, probably anova
            lm_pvalue = lm_res.pvalues.loc['servo_pos']
            aov_pvalue = aov_res.anova_table.loc['servo_pos', 'Pr > F']
            
            # Store
            stats_l.append((lm_pvalue, aov_pvalue))
            stats_keys_l.append((task, rewside))

    # Concat stats
    stats_df = pandas.DataFrame.from_records(
        stats_l, columns=['lm_p', 'aov_p'])
    stats_df.index = pandas.MultiIndex.from_tuples(
        stats_keys_l, names=['task', 'rewside'])

    
    ## Create figure
    f, axa = plt.subplots(1, 2, figsize=(4.2, 2.25), sharey=True)
    f.subplots_adjust(wspace=.2, bottom=.325, right=.975, top=.825, left=.175)
    
    # Plot by task
    for task in task_l:
        # Get ax
        ax = axa.flatten()[task_l.index(task)]
        ax.set_title(task, pad=15)
        
        # Extract data from this task
        task_perfdf = perfdf.loc[:, task]
        
        # Label the groups according to the task
        if task == 'detection':
            group_index2group_label = (
                lambda rewside: {'left': 'nothing', 'right': 'something'}[
                rewside])
            index2plot_kwargs = index2plot_kwargs__detection
        else:
            group_index2group_label = my.plot.group_index2group_label__rewside2shape
            index2plot_kwargs = my.plot.index2plot_kwargs__shape_task
        
        # Grouped bar plot
        servo_pos2dist = {1670: 'far', 1760: 'med.', 1850: 'close'}
        
        my.plot.grouped_bar_plot(
            task_perfdf,
            group_name_fig_ypos=.05,
            index2label=lambda ser: servo_pos2dist[ser['servo_pos']],
            index2plot_kwargs=index2plot_kwargs,
            group_index2group_label=group_index2group_label,
            xtls_kwargs={'rotation': 45, 'size': 12},
            plot_error_bars_instead_of_points=True,
            ax=ax,
        )
    
        # label n
        #~ ax.text(.99, 1.08, 'n = {}'.format(task_perfdf.shape[1]), 
            #~ ha='right', va='top', transform=ax.transAxes, size=12)

        
        ## Asterisks
        asterisk_ypos = .95
        ax.plot([0, 2], [asterisk_ypos] * 2, 'k-', lw=.8, clip_on=False)
        ax.plot([4, 6], [asterisk_ypos] * 2, 'k-', lw=.8, clip_on=False)
        
        # Doesn't matter which is taken
        sig_metric = 'aov_p'
        
        # Plot left
        pvalue = stats_df.loc[task].loc['left'].loc[sig_metric]
        sig_str = my.stats.pvalue_to_significance_string(pvalue)
        if '*' in sig_str:
            ax.text(1, asterisk_ypos - .02, sig_str, ha='center', va='bottom', size=12)
        else:
            ax.text(1, asterisk_ypos, sig_str, ha='center', va='bottom', size=12)

        # Plot right
        pvalue = stats_df.loc[task].loc['right'].loc[sig_metric]
        sig_str = my.stats.pvalue_to_significance_string(pvalue)
        if '*' in sig_str:
            ax.text(5, asterisk_ypos - .02, sig_str, ha='center', va='bottom', size=12)
        else:
            ax.text(5, asterisk_ypos, sig_str, ha='center', va='bottom', size=12)


        ## Pretty
        my.plot.despine(ax)
        xlim = ax.get_xlim()
        ax.plot(xlim, [.5, .5], 'k--', lw=.8)
        ax.set_xlim(xlim)
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_yticklabels(('0.0', '', '0.5', '', '1.0'))
    
    # Pretty
    axa[0].set_ylabel('performance')
    axa[1].texts[0].set_color('orange')
    axa[1].texts[1].set_color('purple')
    axa[0].texts[0].set_color('blue')
    axa[0].texts[1].set_color('red')
    
    f.savefig('PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS.svg')


    ## Stats
    with open('STATS__PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS', 'w') as fi:
        fi.write('STATS__PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(perfdf.T),
            len(perfdf.T.loc['detection']),
            len(perfdf.T.loc['discrimination']),
            ))
        fi.write('error bars: sem\n')
        
        fi.write('one-way anova_rm "perf ~ servo_pos" on each (task * rewside)\n')
        fi.write('p-values:\n')
        fi.write(stats_df.to_string() + '\n')
    
    with open('STATS__PLOT_PERF_BY_TASK_REWSIDE_SERVOPOS', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))
    
plt.show()