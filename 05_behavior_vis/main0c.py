## Example whisking traces
"""
2A
    INDIVIDUAL_TRIALS_BY_MOUSE
    Example whisking traces from many mice
"""

import json
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Plotting
DEGREE = chr(176)
my.plot.presentation_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(params, 
    'big_tm', mouse2task=mouse2task)
C2_whisk_cycles = my.dataload.load_data_from_patterns(params, 
    'big_C2_tip_whisk_cycles', mouse2task=mouse2task)
big_tip_pos = my.dataload.load_data_from_patterns(params, 
    'big_tip_pos', mouse2task=mouse2task)
    
    
## Slice tip angle
tip_angle = big_tip_pos['tip'].xs('angle', level='metric', axis=1)


## Plots
INDIVIDUAL_TRIALS_BY_MOUSE = True

if INDIVIDUAL_TRIALS_BY_MOUSE:
    ## For each mouse plot a set of trials
    n_choose_trials = 3
    outcome_l = ['hit']


    ## Slice out example trials
    sliced_l = []
    for mouse, sub_tm in big_tm[big_tm['outcome'].isin(outcome_l)].groupby('mouse'):
        if mouse2task[mouse] == 'detection':
            continue
        
        iidxs = 0 + my.misc.take_equally_spaced(range(len(sub_tm)), n_choose_trials)
        sliced = tip_angle.loc[sub_tm.index[iidxs]].copy()
        sliced_l.append(sliced)
    sliced_df = pandas.concat(sliced_l)

    
    ## Order the mice
    mouse_task_df = mouse2task.reset_index().sort_values(['task', 'mouse'])
    mouse_task_df = mouse_task_df[mouse_task_df['task'] != 'detection'].copy()
    mouse_l = list(mouse_task_df['mouse'])

    
    ## Plot params
    # How to space in y (bigger compresses each trial)
    y_spacing = 100

    # How to space in x (bigger compresses each mouse)
    x_spacing = 3.5

    # Where to put the mouse title
    mouse_ypos = n_choose_trials * y_spacing + 20

    
    ## Make handles
    f, ax = plt.subplots(figsize=(13.3, 2.6))
    f.subplots_adjust(left=.04, right=.99, top=.95, bottom=.05)
    ax.autoscale(False)#tight=True)


    ## Iterate over mice
    for n_mouse, mouse in enumerate(mouse_l):
        # Slice
        sliced = sliced_df.xs(mouse, level='mouse')

        # Offset in x by mouse
        x_offset = mouse_l.index(mouse) * x_spacing
        
        # Plot each trial
        for n_trial, sliced_idx in enumerate(sliced.index):
            # Unstack whisker
            trial = sliced.loc[sliced_idx].unstack('whisker')
            
            # Downsample
            trial = trial.loc[-400:200]
            trial = trial.iloc[::5]
            
            # Take C2 and demean
            C2_angle = trial['C2'] - sliced['C2'].mean().mean()
            
            # Get tvals
            tvals = trial.index / 200.
            
            # Offset in y by trial
            y_offset = n_trial * y_spacing
            
            # Plot
            ax.plot(
                tvals + x_offset, 
                trial['C2'] + y_offset, 
                color='g', lw=.8)

            # Label the trial number
            if n_mouse == 0:
                ax.text(-2.5, y_offset + 10, str(n_choose_trials - n_trial), 
                    size=12, ha='center', va='center')
        
        # title by mouse and task
        titstr = '{}'.format(n_mouse + 1)
        ax.text(
            x_offset + np.mean(tvals), mouse_ypos - 30,
            titstr, size=12, ha='center', va='top')


    ## Legend
    t_start = 32
    y_start = 260
    t_len = 1
    y_len = 45
    ax.plot(
        [t_start, t_start + t_len], [y_start, y_start], 
        'k-', lw=.8, clip_on=False)
    ax.plot(
        [t_start, t_start], [y_start, y_start + y_len], 
        'k-', lw=.8, clip_on=False)
    ax.text(
        t_start + .5, y_start - 10, '{} s'.format(t_len), 
        ha='center', va='top', size=12)
    ax.text(
        t_start, y_start + y_len / 2., '{}{}'.format(y_len, DEGREE), 
        ha='right', va='center', size=12, rotation=90)
    

    ## Pretty
    ax.set_ylim((-40, mouse_ypos - 20))
    ax.set_xlim((-2, (len(mouse_l) - 1) * x_spacing + 1))
    my.plot.despine(ax, which=('top', 'bottom', 'left', 'right'))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Pretty the trial labels for animate
    f.text(.01, .475, 'trial', rotation=90, size=14, ha='center', va='center')
    f.text(.5, .975, 'mouse', size=14, ha='center', va='center')
    
    
    ## Save
    f.savefig('INDIVIDUAL_TRIALS_BY_MOUSE.png', dpi=300)
    f.savefig('INDIVIDUAL_TRIALS_BY_MOUSE.svg')        
    
plt.show()