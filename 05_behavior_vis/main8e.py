## Compare performance at various levels of trims
"""
S1C
    PLOT_PERFORMANCE_BY_N_ROWS
    N/A
    Performance versus number of rows of whiskers
"""

import json
import os
import datetime
import numpy as np
import pandas
import matplotlib.pyplot as plt
import my
import my.plot


## Plotting params
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Get data
gradual_trims_dir = os.path.join(params['pipeline_input_dir'], 'gradual_trims')
tmdf = pandas.read_pickle(os.path.join(gradual_trims_dir, 'tmdf'))
session_table = pandas.read_pickle(os.path.join(gradual_trims_dir, 'session_table'))


## Count sessions per mouse * n_rows and ensure enough of each type
# 229CR has no 3-row sessions
# The others sometimes have as few as 1 or 2 sessions per n_rows
n_session_by_mouse_and_nrows = session_table.groupby(
    ['mouse', 'n_rows']).size().unstack()


## Group and aggregate perf
perf_by_session = tmdf.groupby(
    ['rewside', 'servo_pos', 'mouse', 'session'])[
    'outcome'].value_counts().unstack('outcome')
perf_by_session['perf'] = perf_by_session['hit'] / (
    perf_by_session['hit'] + perf_by_session['error'])


## Join n_rows on perf
perf_by_session = perf_by_session.join(session_table['n_rows'])
perf_by_session = perf_by_session.set_index(
    'n_rows', append=True).reorder_levels(
    ['mouse', 'n_rows', 'rewside', 'servo_pos', 'session']).sort_index()


## Aggregate 
# Over sessions within mouse
agg_perf = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev != 'session'])

# Over mice
mean_perf = agg_perf.mean(
    level=[lev for lev in agg_perf.index.names 
    if lev != 'mouse']).unstack('n_rows')
err_perf = agg_perf.sem(
    level=[lev for lev in agg_perf.index.names 
    if lev != 'mouse']).unstack('n_rows')


## Plot
PLOT_PERFORMANCE_BY_N_ROWS = True

if PLOT_PERFORMANCE_BY_N_ROWS:
    
    rewside_l = ['left', 'right']
    row2color = {1: 'b', 2: 'g', 3: 'r', 5: 'k'}
    #~ f, axa = plt.subplots(1, len(rewside_l), figsize=(6.5, 2.75))
    #~ f.subplots_adjust(wspace=.4, bottom=.225, right=.975, left=.125)
    f, axa = my.plot.figure_1x2_small(sharey=True)
    f.subplots_adjust(top=.85, bottom=.275)
    for n_rows in mean_perf.columns:
        if n_rows == 0:
            continue
        
        color = row2color[n_rows]
        linestyle = '-'
        
        for rewside in rewside_l:
            ax = axa[rewside_l.index(rewside)]
            ax.set_title({'left': 'concave', 'right': 'convex'}[rewside])

            topl = mean_perf.loc[rewside, n_rows]
            topl_err = err_perf.loc[rewside, n_rows]
            
            ax.plot(topl, linestyle=linestyle, color=color)

            ax.fill_between(
                x=topl.index, 
                y1=(topl - topl_err),
                y2=(topl + topl_err),
                lw=0, alpha=.2, color=color)

    for ax in axa:
        my.plot.despine(ax)
        ax.set_xlim((1870, 1650))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_ylim((0.5, 1))
        ax.set_xticks((1670, 1760, 1850))
        ax.set_xticklabels(('far', 'med.', 'close'))
        #~ ax.set_xlabel('stimulus position')
        #~ ax.set_ylabel('performance')
        ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.8)

    axa[0].set_ylabel('performance')
    f.text(.55, .035, 'stimulus position', ha='center', va='center')
    
    # Legend
    f.text(.55, .85, '5 rows', size=12, color=row2color[5], ha='center', va='center')
    f.text(.55, .775, '3 rows', size=12, color=row2color[3], ha='center', va='center')
    f.text(.55, .7, '2 rows', size=12, color=row2color[2], ha='center', va='center')
    f.text(.55, .625, '1 row', size=12, color=row2color[1], ha='center', va='center')


    ## Save
    f.savefig('PLOT_PERFORMANCE_BY_N_ROWS.svg')
    f.savefig('PLOT_PERFORMANCE_BY_N_ROWS.png', dpi=300)
    

    stats_filename = 'STATS__PLOT_PERFORMANCE_BY_N_ROWS'
    stats_data = agg_perf.unstack('mouse')
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice;\n{}\n'.format(stats_data.shape[1], stats_data.columns))
        fi.write('error bars: SEM over mice\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
        print('\n'.join(lines))


plt.show()