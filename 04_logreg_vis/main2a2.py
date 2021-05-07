## Plot weights of individual whisker contacts

"""
3G
    PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_normal
    STATS__PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_normal
    Decoding stimulus, regular dataset, regular trial balancing

S3G
    PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_choice_no_opto_normal
    STATS__PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_normal
    Decoding choice, regular dataset, regular trial balancing

S3H
    PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_no_licks1_normal
    STATS__PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_no_licks1_normal
    Decoding stimulus, no_licks1 dataset, regular trial balancing

S3I
    PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_subsampling
    STATS__PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_rewside_no_opto_subsampling
    Decoding stimulus, regular dataset, subsampling instead of balancing
    
"""
import json
import os
import pandas
import numpy as np
import scipy.stats
import my
import my.plot 
import matplotlib.pyplot as plt

my.plot.poster_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Paths
full_model_dir = os.path.join(params['logreg_dir'], 'full_model')


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))


## Load results of main2a1
big_weights_part = pandas.read_pickle('big_weights_part')

# Add task level
big_weights_part = my.misc.insert_level(
    big_weights_part, name='task', 
    func=lambda idx: idx['mouse'].map(mouse2task))   


## Plot flags
PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL = True 

# Use only this model
model = 'contact_binarized+anti_contact_count+angle'

# And plot only this metric
metric = 'contact_binarized'

# Drop C0
big_weights_part = big_weights_part.drop(
    'C0', level='label', axis=1, errors='ignore')


## Plots
if PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL:
    ## Things to iterate plots over
    # This goes: decode_label, dataset, subsampling
    iterations = [
        ('rewside', 'no_opto', False),
        ('choice', 'no_opto', False),
        ('rewside', 'no_opto_no_licks1', False),
        ('rewside', 'no_opto', True),
        ]

    # Always plot both tasks
    task_l = ['detection', 'discrimination']

    
    ## Iterate over iterations
    for decode_label, dataset, subsampling in iterations:
        ## Slice dataset
        this_big_weights_part = big_weights_part.loc[
            :, (subsampling, dataset, model, metric)].xs(
            decode_label, level='decode_label').copy()

        # Sort
        this_big_weights_part = this_big_weights_part.sort_index().sort_index(axis=1)

        
        ## Aggregate
        # Mean over analysis_bin
        this_big_weights_part = this_big_weights_part.mean(
            level=[lev for lev in this_big_weights_part.columns.names
            if lev != 'analysis_bin'], axis=1)
        
        # Mean over session within task * mouse
        this_big_weights_part = this_big_weights_part.mean(
            level=[lev for lev in this_big_weights_part.index.names
            if lev != 'session'])


        ## Drop mice that are all null
        all_null_mice = this_big_weights_part.isnull().all(1).droplevel('task')    
        all_null_mice = all_null_mice.index[all_null_mice.values]
        if len(all_null_mice) > 0:
            print("dropping {} mice for being all null".format(len(all_null_mice)))
            this_big_weights_part = this_big_weights_part.drop(
                all_null_mice, level='mouse')

        # Debug
        assert not this_big_weights_part.isnull().any().any()

        
        ## T-test each whisker's weights vs zero
        pvalues_l = []
        for task in task_l:
            # Slice
            weight_by_mouse_and_whisker = this_big_weights_part.loc[task]
            
            # t-test vs 0
            pvalues = scipy.stats.ttest_1samp(
                weight_by_mouse_and_whisker.values, popmean=0).pvalue
            
            # Store
            pvalues_l.append(
                pandas.Series(pvalues, 
                index=weight_by_mouse_and_whisker.columns))
        
        # Concat stats
        stats_df = pandas.concat(pvalues_l, keys=task_l, names=['task'], axis=1)
        

        ## Create figure handles
        f, axa = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(6, 2),
            gridspec_kw={'width_ratios': [1.5, .9, 1.5, .8]})
        f.subplots_adjust(left=.175, right=1, bottom=.225, top=.85)
        
        # These are just spacers
        axa[1].set_visible(False)
        axa[3].set_visible(False)


        ## Helper functions
        def index2plot_kwargs(ser):
            return {'ec': 'k', 'fc': 'w', 'lw': 1}

        
        ## Plot each task (axis)
        for task in task_l:
            # Slice
            topl = this_big_weights_part.loc[task].T

            # Get ax
            ax = axa[2 * task_l.index(task)]
            ax.set_title(task, pad=15)
            
            # Bar plot
            my.plot.grouped_bar_plot(
                topl,
                ax=ax,
                index2plot_kwargs=index2plot_kwargs,
                group_index2group_label=None,
                datapoint_plot_kwargs={'ms': 4},
                index2label=lambda ser: ser['label'],
                plot_error_bars_instead_of_points=True,
                elinewidth=1,
            )
            
            
            ## Asterisks
            continue
            if decode_label == 'rewside':
                sig_string_yval = .5
            else:
                sig_string_yval = .4
            
            for n_whisker, whisker in enumerate(topl.index):
                pvalue = stats_df.loc[whisker, task]
                sig_string = my.stats.pvalue_to_significance_string(pvalue)
                ax.text(
                    ax.get_xticks()[n_whisker], 
                    sig_string_yval, 
                    sig_string, ha='center', va='center', size=12)

        
        ## Pretty
        if decode_label == 'choice':
            ylim_abs = .2
        elif subsampling:
            ylim_abs = .05
        else:
            ylim_abs = .5
        xlim = ax.get_xlim()
        for ax in axa.flatten():
            my.plot.despine(ax)
            ax.plot(xlim, [0, 0], 'k-', lw=.75)
            ax.set_ylim((-ylim_abs, ylim_abs))
            ax.set_xlim(xlim)
        axa[0].set_ylabel('evidence per contact\n(logits)')
        f.text(.52, .03, 'whisker in contact', ha='center', va='center')


        ## Legend
        f.text(.475, .75, 'something', ha='center', va='center')
        f.text(.475, .3, 'nothing', ha='center', va='center')
        
        f.text(.9, .75, 'convex', ha='center', va='center')
        f.text(.9, .3, 'concave', ha='center', va='center')


        ## Save
        figname = (
            'PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_{}_{}_{}'.format(
            decode_label, dataset, 'subsampling' if subsampling else 'normal',
            ))
        f.savefig(figname + '.svg')
        f.savefig(figname + '.png', dpi=300)


        ## Stats
        stats_filename = (
            'STATS__PLOT_CONTACT_WEIGHTS_BY_WHISKER_AND_DECODELABEL_{}_{}_{}'.format(
            decode_label, dataset, 'subsampling' if subsampling else 'normal',
            ))
        
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write('n = {} mice ({} det., {} disc.)\n'.format(
                len(this_big_weights_part),
                len(this_big_weights_part.loc['detection']),
                len(this_big_weights_part.loc['discrimination']),
                ))
            fi.write('meaning over time, then over session, then over mouse\n')
            fi.write('error bars: SEM over mice\n')
            fi.write('t-test each whisker vs zero:\n')
            fi.write(stats_df.to_string() + '\n')
        
        with open(stats_filename) as fi:
            lines = fi.readlines()
        print(''.join(lines))



plt.show()