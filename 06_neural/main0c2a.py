## Plot whisker position and contact probability locked to whisk cycle
# This takes a minute or so to run
"""
2C	
    CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER	
    STATS__CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER	
    Whisker position and contact probability versus time in cycle, discrimination only
"""

import json
import pandas
import numpy as np
import os
import my
import my.plot
import my.dataload
import matplotlib.pyplot as plt


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Set up plotting
my.plot.manuscript_defaults()
my.plot.font_embed()
this_WHISKER2COLOR = {'C0': 'gray', 'C1': 'b', 'C2': 'g', 'C3': 'r'}


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load behavioral features
behavioral_features = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features')


## Load results of main0b2 (all sessions)
big_cycle_sliced_touching = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_cycle_sliced_touching'))
big_cycle_sliced_tip_pos = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_cycle_sliced_tip_pos'))


## Plots
CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER = True

if CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER:
    ## Choose the cycles in which this contact_typ made contact
    # If all whisks are taken, then the probability distribution is too low
    # to be readable. So take any whisk with contact
    contact_typ = 'any'
    
    if True: #elif contact_typ == 'any':
        # All cycles in which any contact_typ made contact
        this_cycles = behavioral_features.index[
            behavioral_features.loc[:, 'contact_binarized'].sum(1) != 0]

    # Keep only session, trial, cycle
    this_cycles = pandas.MultiIndex.from_frame(
        this_cycles.to_frame().reset_index(drop=True)[
        ['session', 'trial', 'cycle']])

    # Continue if not enough cycles to be worth averaging
    assert len(this_cycles) > 10


    ## Slice this_cycles out of touching and tip_pos accordingly
    this_touching = my.misc.slice_df_by_some_levels(
        big_cycle_sliced_touching, this_cycles)
    this_tip_pos = my.misc.slice_df_by_some_levels(
        big_cycle_sliced_tip_pos, this_cycles)
    

    ## Slice out the time range to plot
    this_touching = this_touching.loc[
        pandas.IndexSlice[:, :, :, np.arange(-6, 7, dtype=np.int)], :].copy()
    this_tip_pos = this_tip_pos.loc[
        pandas.IndexSlice[:, :, :, np.arange(-6, 7, dtype=np.int)], :].copy()


    ## Aggregate
    # Add mouse and task level
    this_touching = my.misc.insert_mouse_and_task_levels(
        this_touching, mouse2task)
    this_tip_pos = my.misc.insert_mouse_and_task_levels(
        this_tip_pos, mouse2task)

    # Keep discrimination only, because the tasks seem to differ
    this_touching = this_touching.loc['discrimination'].copy()
    this_tip_pos = this_tip_pos.loc['discrimination'].copy()
    
    # First mean over session * trial * cycle, keeping mouse * shift
    mtouching_by_mouse = this_touching.mean(
        level=['mouse', 'shift']).sort_index()
    mtippos_by_mouse = this_tip_pos.mean(
        level=['mouse', 'shift']).sort_index()

    # Normalize within each mouse * whisker to account for arbitrary offsets
    mtippos_by_mouse = mtippos_by_mouse.sub(mtippos_by_mouse.mean(level='mouse'))
    
    # Then mean over all mouse * shift and take error bars as SEM
    mtippos = mtippos_by_mouse.mean(level='shift')
    mtippos_err = mtippos_by_mouse.sem(level='shift')

    mtouching = mtouching_by_mouse.mean(level='shift')
    mtouching_err = mtouching_by_mouse.sem(level='shift')
    
    
    ## Create figure
    f, axa = plt.subplots(1, 2, figsize=(4, 2.25), sharex=True)
    f.subplots_adjust(
        left=.08, right=.975, bottom=.25, top=.9, wspace=.6)


    ## Plot tip_pos
    whisker2offset = {'C1': 0, 'C2': 10, 'C3': 20} # to separate
    whisker2offset = {'C0': 0, 'C1': 0, 'C2': 0, 'C3': 0} # to overplot
    ax = axa[0]
    
    # Plot tip pos within cycle
    for whisker in ['C1', 'C2', 'C3']:
        color = this_WHISKER2COLOR[whisker]
        offset = whisker2offset[whisker]
        
        # Mean
        ax.plot(
            mtippos.index.values * 5,
            offset + mtippos[whisker], 
            color=color, ls='-', lw=1)

        # SEM
        ax.fill_between(
            x=mtippos.index.values * 5,
            y1=offset + mtippos[whisker] - mtippos_err[whisker],
            y2=offset + mtippos[whisker] + mtippos_err[whisker],
            color=color, ls='-', lw=0, alpha=.25)
    
    ax.set_ylabel('position ({})'.format(chr(176)))

    # Despine left
    my.plot.despine(ax, which=('left', 'top', 'right'))
    ax.set_yticks([])

    # Scale bar
    ax.plot([23, 23], [5, 10], 'k-', lw=.8)
    ax.text(24, 7, '5{}'.format(chr(176)), ha='left', va='center', size=12)
    
    
    ## Plot touching
    ax = axa[1]

    # Plot touching within cycle
    # Plot tip pos within cycle
    for whisker in ['C1', 'C2', 'C3']:
        color = this_WHISKER2COLOR[whisker]
        
        # Mean
        ax.plot(
            mtouching.index.values * 5,
            mtouching[whisker], 
            color=color, ls='-', lw=1)

        # SEM
        ax.fill_between(
            x=mtouching.index.values * 5,
            y1=mtouching[whisker] - mtouching_err[whisker],
            y2=mtouching[whisker] + mtouching_err[whisker],
            color=color, ls='-', lw=0, alpha=.25)

    ax.set_ylim((0, 1))
    ax.set_yticks((0, 1))
    ax.set_ylabel('Pr(in contact)')


    ## Pretty
    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_xlim((-30, 30))
        ax.set_xticks((-25, 0, 25))

    f.text(.55, .05, 'time from cycle peak (ms)', ha='center', va='center')

    n_mice = len(mtouching_by_mouse.index.get_level_values(
        'mouse').unique().sort_values())
    axa[1].text(15, 1.05, 'n = {} mice'.format(n_mice), ha='center', va='center')

    
    ## Save
    f.savefig(
        'CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER.svg')
    f.savefig(
        'CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER.png', 
        dpi=300)


    ## Stats
    idx0 = mtouching_by_mouse.index.get_level_values('mouse').unique().sort_values()    
    idx1 = mtippos_by_mouse.index.get_level_values('mouse').unique().sort_values()
    assert (idx0 == idx1).all()
    with open('STATS__CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER', 'w') as fi:
        fi.write('STATS__CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER\n')
        fi.write('n = {} mice (disc. only)\n'.format(len(idx0)))
        fi.write('error bars: sem over mice\n'.format(len(idx0)))
        
        n_cycles_included = len(this_touching.index.to_frame().reset_index(
            drop=True)[['session', 'trial', 'cycle']].drop_duplicates())
        fi.write('{} whisk cycles included'.format(n_cycles_included))
    
    with open('STATS__CYCLE_PLOT_ANGLE_AND_CONTACTPROB_NOSPIKES_CONTACTSONLY_SMALLER', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

    
plt.show()
