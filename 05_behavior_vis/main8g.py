## Plot performance of KM100 and KM102 over gradual whisker trim
"""
4F
    PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS_grand
    N/A
    Performance for all, single, and no whiskers.

4G
    PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS
    N/A
    Performance on single whisker by stimulus and position.
"""

import json
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import my
import my.plot
import scipy.stats


## The mice with gradual trims
mouse_l = ['KM100', 'KM102']


## Plot stuff
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Get data
single_whisker_dir = os.path.join(params['pipeline_input_dir'], 'single_whisker')
big_perf_df = pandas.read_pickle(os.path.join(single_whisker_dir, 'big_perf'))


## Plots
PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS = True


if PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS:
    ## Helper aggregation function
    def add_CIs_and_perf(df):
        res = df.copy()
        
        # Recalculate perf and CIs
        CIs = pandas.DataFrame(
            [my.stats.binom_confint(n_hit, n_hit + n_error) 
            for n_hit, n_error in df[['hit', 'error']].values],
            index=df.index, columns=['lo', 'hi'])
        
        # Concatenate
        res['perf'] = res['hit'] / (res['hit'] + res['error'])
        res = pandas.concat([res, CIs], axis=1)        
        
        return res
    
    
    ## Aggregate over sessions
    this_big_perf_df = big_perf_df.copy()
    
    # Aggregagate over session, keeping mouse and n_whiskers and stimulus
    aggregated = this_big_perf_df[
        ['hit', 'error']].unstack(
        ['rewside', 'servo_pos']).sum(
        level=['mouse', 'n_whiskers']).stack(
        ['rewside', 'servo_pos'])
    
    aggregated = add_CIs_and_perf(aggregated)
    
    
    ## Calculate grand performance averages
    grand_agg = aggregated[['hit', 'error']].sum(level=['mouse', 'n_whiskers'])
    grand_agg = add_CIs_and_perf(grand_agg)
    
    # Slice out 5, 1, 0 only
    grand_agg = grand_agg.reorder_levels(['n_whiskers', 'mouse']).sort_index().loc[[5, 1, 0]]

    
    ## Plot
    # helper functions
    def index2plot_kwargs(idx):
        if idx['n_whiskers'] == 5:
            fc = 'w'
            ec = 'k'
        elif idx['n_whiskers'] == 1:
            fc = 'gray'
            ec = 'k'
        elif idx['n_whiskers'] == 0:
            fc = 'k'
            ec = 'k'
        else:
            1/0
            
        return {'fc': fc, 'ec': ec,}

    def group_index2group_label(idx):
        return {5: 'C-row', 1: 'C2', 0: 'none'}[idx]

    # Plot
    f, axa = plt.subplots(2, 1, figsize=(1.8, 3.4))
    f.subplots_adjust(wspace=.2, hspace=.7, bottom=.16, right=.9, left=.4)
    for mouse in mouse_l:
        ## Slice
        ax = axa[mouse_l.index(mouse)]
        mouse_grand_agg = grand_agg.xs(mouse, level='mouse').copy()
        

        ## sig test
        # Stats vs 1
        pval_l = []
        for n_whiskers in mouse_grand_agg.index:
            cmp = mouse_grand_agg.loc[1]
            tst = mouse_grand_agg.loc[n_whiskers]
            test_res = scipy.stats.fisher_exact([
                cmp.loc[['hit', 'error']].astype(np.int),
                tst.loc[['hit', 'error']].astype(np.int),
                ])
            pval = test_res[1]
            pval_l.append(pval)
        mouse_grand_agg['p'] = pval_l
        
        
        ## Plot
        my.plot.grouped_bar_plot(
            df=mouse_grand_agg['perf'],
            yerrlo=mouse_grand_agg['lo'],
            yerrhi=mouse_grand_agg['hi'],
            index2plot_kwargs=index2plot_kwargs,
            group_index2group_label=group_index2group_label,
            group_name_fig_ypos=.17,
            ax=ax,
        )
        
        # Plot sigstr
        for n_n_whiskers, n_whiskers in enumerate(mouse_grand_agg.index):
            if n_whiskers == 1:
                continue
            pval = mouse_grand_agg.loc[n_whiskers, 'p']
            xval = np.mean([1, ax.get_xticks()[n_n_whiskers]])
            txt = my.stats.pvalue_to_significance_string(pval)
            ax.text(xval, 1.05 + (.025 if txt == 'n.s.' else 0), txt, 
                ha='center', va='center', size=12)
            #~ print((servo_pos, xval, pval))        

    for ax in axa:
        my.plot.despine(ax)
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .5, 1))
        ax.plot([0, .9], [1, 1], 'k-', clip_on=False, lw=.75)
        ax.plot([2, 1.1], [1, 1], 'k-', clip_on=False, lw=.75)
        
    
    axa[0].set_title('mouse 1', pad=15)
    axa[1].set_title('mouse 2', pad=15)
    f.text(.125, .55, 'performance', ha='center', va='center', rotation=90)
    
    f.savefig('PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS_grand.svg')
    f.savefig('PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS_grand.png', dpi=300)
    
    ## Plot
    # Helper functions
    def group_index2group_label(idx):
        return {1670: 'far', 1760: 'med.', 1850: 'close'}[idx]
    
    # Slice
    n_whiskers_l = [5, 1]
    aggregated = aggregated.reindex(n_whiskers_l, level='n_whiskers')
    
    # Put n_whiskers last
    aggregated = aggregated.reorder_levels(
        ['mouse', 'rewside', 'servo_pos', 'n_whiskers']).sort_index()
    
    rewside_l = ['left', 'right']
    servo_pos_l = [1850, 1760, 1670]
    f, axa = plt.subplots(len(mouse_l), len(rewside_l), figsize=(5, 3.4))
    f.subplots_adjust(wspace=.2, hspace=.7, bottom=.16, right=.98, left=.15)

    for mouse in mouse_l:
        for rewside in rewside_l:
            ## Slice
            ax = axa[
                mouse_l.index(mouse),
                rewside_l.index(rewside),
                ]
            #~ ax.set_title('{} {}'.format(mouse, rewside))

            # Slice
            sliced = aggregated.loc[mouse].loc[rewside].copy()
            
            # Order
            slicing_order = pandas.MultiIndex.from_product(
                [servo_pos_l, n_whiskers_l], names=['servo_pos', 'n_whiskers'])
            sliced = sliced.loc[slicing_order]
            
            
            ## sig test
            servo_pos2pval = {}
            for servo_pos in sliced.index.levels[0]:
                test_res = scipy.stats.fisher_exact(
                    sliced.loc[servo_pos, ['error', 'hit']].values
                    )
                pval = test_res[1]
                servo_pos2pval[servo_pos] = pval
                #~ print((servo_pos, pval))

            if ax in axa[-1]:
                this_group_index2group_label = group_index2group_label
            else:
                this_group_index2group_label = lambda s: None
            
            
            ## Plot
            my.plot.grouped_bar_plot(
                df=sliced['perf'],
                yerrlo=sliced['lo'],
                yerrhi=sliced['hi'],
                index2plot_kwargs=index2plot_kwargs,
                group_index2group_label=this_group_index2group_label,
                ax=ax,
            )
            
            # Plot sigstr
            for n_servo_pos, servo_pos in enumerate(servo_pos_l):
                pval = servo_pos2pval[servo_pos]
                xval = ax.get_xticks()[n_servo_pos * 2:n_servo_pos*2 + 2].mean()
                txt = my.stats.pvalue_to_significance_string(pval)
                ax.text(xval, 1.05 + (.025 if txt == 'n.s.' else 0), txt, 
                    ha='center', va='center', size=12)
                #~ print((servo_pos, xval, pval))

    for ax in axa.flatten():
        if ax in axa[:, 1]:
            my.plot.despine(ax, which=('left', 'top', 'right'))
            ax.set_yticklabels([])
        else:
            my.plot.despine(ax)
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .5, 1))
    f.text(.32, .05, 'concave', ha='center', va='center')
    f.text(.79, .05, 'convex', ha='center', va='center')
    f.text(.56, .95, 'mouse 1', ha='center', va='center')
    f.text(.56, .52, 'mouse 2', ha='center', va='center')
    f.text(.03, .55, 'performance', ha='center', va='center', rotation=90)
    
    f.savefig('PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS.svg')
    f.savefig('PLOT_PERF_BY_STIM_FOR_SINGLE_VS_ALL_WHISKERS.png', dpi=300)

plt.show()