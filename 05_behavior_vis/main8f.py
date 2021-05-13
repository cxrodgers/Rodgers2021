## Lesion performance
"""
1C
    PLOT_LONGITUDINAL_LESION_EFFECT
    N/A
    Plot effect of lesion over multiple days

1D
    PLOT_QUANTIFIED_LESION_EFFECT
    STATS__PLOT_QUANTIFIED_LESION_EFFECT
    Summarize effect of lesion by averaging over days
"""

import json
import os
import pandas
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import my
import my.plot
import my.stats


## Plot stuff
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Get data
lesion_dir = os.path.join(params['pipeline_input_dir'], 'lesion')
big_peri = pandas.read_pickle(os.path.join(lesion_dir, 'big_peri'))


## Slice around locking events
locking_events = ['contra', 'ipsi']
mouse_l = big_peri.index.levels[0]

locked_l = []
keys_l = []
for locking_event in locking_events:
    for mouse in mouse_l:
        # Get mouse data
        mouse_peri = big_peri.loc[mouse]
        
        # Slice post_lock
        try:
            post_lock = mouse_peri.xs(
                locking_event, level='epoch', drop_level=False).copy()
        except KeyError:
            continue
        
        # Slice pre lock
        lock_day = post_lock.index.get_level_values('n_day')[0]
        pre_lock = mouse_peri.loc[
            pandas.IndexSlice[:, lock_day-3:lock_day-1], :].copy()
        
        # Ensure pre_lock is all from same epoch
        assert len(pre_lock.index.get_level_values('epoch').unique()) == 1
        
        # Replace n_day in pre and post
        pre_lock['n_session'] = np.arange(-len(pre_lock), 0, dtype=np.int)
        post_lock['n_session'] = np.arange(len(post_lock), dtype=np.int)
        
        # Concat pre and post
        locked = pandas.concat(
            [pre_lock, post_lock], keys=['pre', 'post'], names=['typ'])
        
        # Replace n_day with n_session on index
        locked = locked.set_index(
            'n_session', append=True).reset_index('n_day').sort_index()
        
        # Store
        locked_l.append(locked)
        keys_l.append((locking_event, mouse))

# Concat
locked_df = pandas.concat(locked_l, keys=keys_l, names=['event', 'mouse'])


## Quantify summary
locked_df2 = locked_df.copy()

# Keep only 3 days after lesion
locked_df2 = locked_df2.loc[pandas.IndexSlice[:, :, :, :, range(-3, 3)], :]

# drop epoch level
locked_df2 = locked_df2.droplevel('epoch')

# Unstack session 
quantified = locked_df2['perf_unforced'].unstack(['typ', 'n_session'])

# Check no missing days
assert not quantified.isnull().any().any()

# Mean
quantified = quantified.mean(axis=1, level='typ')


## Plots
PLOT_QUANTIFIED_LESION_EFFECT = True
PLOT_LONGITUDINAL_LESION_EFFECT = True


## Plot quantified
if PLOT_QUANTIFIED_LESION_EFFECT:
    event_l = ['ipsi', 'contra']
    #~ f, axa = plt.subplots(1, 2, figsize=(4.5, 2.5), sharex=True, sharey=True)
    #~ f.subplots_adjust(top=.85, bottom=.2, left=.2, right=.95, wspace=.4)
    
    f, axa = my.plot.figure_1x2_small(sharex=True, sharey=True)

    for event in event_l:
        # Get ax
        ax = axa[event_l.index(event)]
        
        # Slice data
        topl = quantified.loc[event, ['pre', 'post']]

        # ttest
        pvalue = scipy.stats.ttest_rel(topl['post'], topl['pre']).pvalue
        
        # Plot
        ax.plot(topl.values.T, color='k', alpha=.25)
        ax.plot(topl.mean(), color='r', lw=2.5, alpha=1)

        # Pretty
        ax.set_title(event)#, pad=15)
        ax.set_yticks((0, .25, .5, .75, 1))
        #~ ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.75)    
        my.plot.despine(ax)

        # sig
        # Not enough to do for ipsi
        if event != 'ipsi':
            ax.plot([0, 1], [.9, .9], 'k-', lw=.75)
            ax.text(.5, .93, my.stats.pvalue_to_significance_string(pvalue),
                ha='center', va='center', )

    for ax in axa:
        ax.set_ylim((.5, 1))
        ax.set_xlim((-.2, 1.2))

    axa[0].set_ylabel('performance')

    f.savefig('PLOT_QUANTIFIED_LESION_EFFECT.svg')
    f.savefig('PLOT_QUANTIFIED_LESION_EFFECT.png')

    stats_filename = 'STATS__PLOT_QUANTIFIED_LESION_EFFECT'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} contra mice: {}\n'.format(
            len(quantified.loc['contra']), quantified.loc['contra'].index.values))
        fi.write('n = {} ipsi, then contra mice: {}\n'.format(
            len(quantified.loc['ipsi']), quantified.loc['ipsi'].index.values))

        fi.write('paired t-test on contra only, p = {}\n'.format(pvalue))
        fi.write('insufficient data to test ipsi\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))


## Longitudinal plot around lesion
if PLOT_LONGITUDINAL_LESION_EFFECT:
    ## Slightly wider than typical 1x2_small
    f, axa = plt.subplots(1, 2, figsize=(4.5, 2), sharex=True, sharey=True)
    f.subplots_adjust(left=.2, right=.975, wspace=.4, bottom=.225, top=.8)

    
    locking_event_l = ['ipsi', 'contra']
    
    for locking_event in locking_event_l:
        if locking_event == 'nwt':
            continue
        
        # Slice
        locked = locked_df.loc[locking_event]
        
        # Get pre and post
        pre = locked.xs('pre', level='typ')
        post = locked.xs('post', level='typ')
        
        # droplevel
        if locking_event != 'contra':
            # Because this one can be ipsi or baseline
            # For 242CR and 243CR the contra is after ipsi
            # For the rest, the contra is after baseline
            assert len(pre.index.get_level_values('epoch').unique()) == 1
        assert len(post.index.get_level_values('epoch').unique()) == 1
        pre = pre.droplevel('epoch')
        post = post.droplevel('epoch')
        
        # unstack
        pre_perf = pre['perf_unforced'].unstack('mouse')
        post_perf = post['perf_unforced'].unstack('mouse')
        
        # Include only 3 days
        post_perf = post_perf.loc[0:2, :]
        
        # Plot
        #~ f, ax = plt.subplots()
        ax = axa[locking_event_l.index(locking_event)]
        ax.set_title(locking_event)
        
        ax.plot(pre_perf, color='k', alpha=.25)
        ax.plot(pre_perf.mean(1), color='r', lw=2.5)
        ax.plot(post_perf, color='k', alpha=.25)
        ax.plot(post_perf.mean(1), color='r', lw=2.5)
        
        # Pretty
        ax.set_ylim((.5, 1))
        ax.set_yticks((0, .25, .5, .75, 1))
        my.plot.despine(ax)
        #~ ax.plot(ax.get_xlim(), [.5, .5], 'k--')

    for ax in axa:
        ax.set_ylim((.5, 1))
        ax.set_xticks((-3, -2, -1, 0, 1, 2))
        ax.plot([-.5, -.5], [.5, 1], 'k--', lw=.75)
        ax.set_xlabel('days post lesion')
    
    axa[0].set_ylabel('performance')

    f.savefig('PLOT_LONGITUDINAL_LESION_EFFECT.svg')
    f.savefig('PLOT_LONGITUDINAL_LESION_EFFECT.png')

plt.show()
    
