## Plot performance for flatter shapes

"""
S1A, right
    PLOT_PERFORMANCE_BY_DIFFICULTY
    STATS__PLOT_PERFORMANCE_BY_DIFFICULTY
    Performance on flatter shapes
"""
import json
import os
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
flatter_shapes_dir = os.path.join(params['pipeline_input_dir'], 'flatter_shapes')
tmdf = pandas.read_pickle(os.path.join(flatter_shapes_dir, 'tmdf'))


## Calculate perf for each mouse * session * rewside * stepper_pos * servo_pos
perf_by_session = tmdf.groupby(
    ['rewside', 'difficulty', 'servo_pos', 'mouse', 'session'])[
    'outcome'].value_counts().unstack('outcome')
perf_by_session['perf'] = perf_by_session['hit'] / (
    perf_by_session['hit'] + perf_by_session['error'])


## Aggregate over sessions within mouse
# Including rewside and difficulty
agg_perf_by_rewside_and_difficulty = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev != 'session'])

# Including only difficulty and meaning over rewside
agg_perf_by_difficulty = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev not in ['rewside', 'session']])


## Aggregate over mice
# By rewside and difficulty
agg_by_rewside_and_difficulty_mean = agg_perf_by_rewside_and_difficulty.mean(
    level=[lev for lev in agg_perf_by_rewside_and_difficulty.index.names 
    if lev != 'mouse'])
agg_by_rewside_and_difficulty_err = agg_perf_by_rewside_and_difficulty.sem(
    level=[lev for lev in agg_perf_by_rewside_and_difficulty.index.names 
    if lev != 'mouse'])

# By difficulty only
agg_by_difficulty_mean = agg_perf_by_difficulty.mean(
    level=[lev for lev in agg_perf_by_difficulty.index.names 
    if lev != 'mouse'])
agg_by_difficulty_err = agg_perf_by_difficulty.sem(
    level=[lev for lev in agg_perf_by_difficulty.index.names 
    if lev != 'mouse'])


## Plot
PLOT_PERFORMANCE_BY_DIFFICULTY = True

if PLOT_PERFORMANCE_BY_DIFFICULTY:
    f, ax = my.plot.figure_1x1_standard()
    for difficulty in ['easy', 'hard']:
        color = 'magenta' if difficulty == 'hard' else 'k'
        linestyle = '-'

        topl = agg_by_difficulty_mean.loc[difficulty]
        topl_err = agg_by_difficulty_err.loc[difficulty]
        
        ax.plot(topl, color=color, linestyle=linestyle)

        ax.fill_between(
            x=topl.index, 
            y1=(topl - topl_err),
            y2=(topl + topl_err),
            color=color, lw=0, alpha=.2)

    my.plot.despine(ax)
    ax.set_ylim((0, 1))
    ax.set_xlim((1870, 1650))
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_xticks((1670, 1760, 1850))
    ax.set_xticklabels(('far', 'med.', 'close'))
    ax.set_xlabel('stimulus position')
    ax.set_ylabel('performance')
    ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.8)


    ## Save
    f.savefig('PLOT_PERFORMANCE_BY_DIFFICULTY.svg')
    f.savefig('PLOT_PERFORMANCE_BY_DIFFICULTY.png', dpi=300)
    
    
    ## Stats
    n_mice = len(agg_perf_by_difficulty.index.get_level_values('mouse').unique())
    stats_filename = 'STATS__PLOT_PERFORMANCE_BY_DIFFICULTY'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice\n'.format(n_mice))
        fi.write(', '.join(agg_perf_by_difficulty.index.get_level_values('mouse').unique().values) + '\n')
        fi.write('mean within session, then within mice\n')
        fi.write('error bars: sem\n'.format(n_mice))
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))



plt.show()