## Simple plots of neural that don't depend on GLM
# Pie charts of recorded neurons
"""
5C
    PIE_N_NEURONS_BY_LAYER_AND_NS
    STATS__N_NEURONS
    Pie chart of neuron laminar location and cell type
"""
import json
import pandas
import numpy as np
import my, my.plot, my.decoders
import matplotlib.pyplot as plt
import os
import whiskvid
import kkpandas
import ArduFSM


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)
big_waveform_info_df = my.misc.insert_mouse_and_task_levels(
    big_waveform_info_df, mouse2task)
big_waveform_info_df['layer'] = big_waveform_info_df['layer'].astype(str)


## Plot flags
STATS_N_NEURONS = True
PIE_N_NEURONS_BY_LAYER_AND_NS = True


## Plots
if STATS_N_NEURONS:
    n_by_task_and_mouse = big_waveform_info_df.groupby(['task', 'mouse']).size()
    n_by_task = n_by_task_and_mouse.sum(level='task')
    n_mice_by_task = n_by_task_and_mouse.groupby('task').size()
    
    with open('STATS__N_NEURONS', 'w') as fi:
        fi.write('n = {} neurons total\n'.format(n_by_task.sum()))
        fi.write('excluding L1 and L6B\n')
        fi.write('detection: {} neurons {} mice\n'.format(
            n_by_task.loc['detection'], n_mice_by_task.loc['detection'],
            ))
        fi.write('discrimination: {} neurons {} mice'.format(
            n_by_task.loc['discrimination'], n_mice_by_task.loc['discrimination'],
            ))
    
    with open('STATS__N_NEURONS', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PIE_N_NEURONS_BY_LAYER_AND_NS:
    ## Aggregate
    # Count by task * layer * NS
    # Distribution very similar for each task, except more 5a in detection
    # So pool over both 
    counted = big_waveform_info_df.groupby(['layer', 'NS']).size()
    counted.index = counted.index.remove_unused_levels()
    summed_counts = counted.sum(level='layer')


    ## Plot
    f, ax = my.plot.figure_1x1_small()
    f.subplots_adjust(left=0, right=1, bottom=0.1, top=1)
    
    # Pie separately by layer * NS
    patches, *junk = ax.pie(counted.values)

    # Pie by the sum of each layer, to draw a solid line around both NS
    # Label these larger wedges only
    patches2, junk = ax.pie(
        summed_counts.values, 
        labels=[layer.upper() for layer in summed_counts.index], 
        wedgeprops={'fc': 'none', 'ec': 'k', 'lw': 1.5, })

    # Recolor patches
    #~ colorbar = my.plot.generate_colorbar(5, start=.1, stop=.9)
    colors = ['k', 'k', 'k', 'k', 'k']
    
    for n_wedge, (wedge, NS) in enumerate(
            zip(patches, counted.index.get_level_values('NS'))):
        # Choose the color
        color = colors[n_wedge // 2]
        
        # Alpha the NS wedges lower
        if NS:
            wedge.set_fc('b')
            wedge.set_alpha(.5)
        else:
            wedge.set_fc('r')
            wedge.set_alpha(.5)

    # N
    f.text(
        .975, .05, 'n = {}'.format(summed_counts.sum()), 
        ha='right', va='center')

    f.savefig('PIE_N_NEURONS_BY_LAYER_AND_NS.svg')
    f.savefig('PIE_N_NEURONS_BY_LAYER_AND_NS.png', dpi=300)


plt.show()
