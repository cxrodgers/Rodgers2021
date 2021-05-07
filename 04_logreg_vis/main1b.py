## Plot decoding model performance
# Plot performance of individual component models
"""
S3D	
    PLOT_INDIVIDUAL_ACCURACY_VERT	
    STATS__PLOT_INDIVIDUAL_ACCURACY_VERT	
    Accuracy of all individual decoders
3B	PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED	
    STATS__PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED	
    Accuracy of a subset of individual decoders
"""
# TODO: standardize this model aggregation procedure with main1c

import json
import pandas
import numpy as np
import my.plot 
import matplotlib
import matplotlib.pyplot as plt
import os

my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Paths
indiv_model_dir = os.path.join(params['logreg_dir'], 'indiv_model_results')


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))

dataset = 'no_opto'
decode_label_l = ['rewside', 'choice']


dataset_results_dir = os.path.join(indiv_model_dir, dataset)
component_names = os.listdir(dataset_results_dir)


## Iterate over the individual components
preds_l = []
for component in component_names:
    ## Load results of component model
    model_dir = os.path.join(dataset_results_dir, component)

    # Load preds
    preds = pandas.read_pickle(os.path.join(model_dir, 'finalized_predictions'))

    # Store preds
    preds_l.append(preds)

# Concat preds
big_preds = pandas.concat(preds_l, keys=component_names, axis=0, names=['metric'])
big_preds = big_preds.reset_index()

# Add mouse name to index
big_preds['mouse'] = big_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
big_preds = pandas.merge(big_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])


## Plots
PLOT_INDIVIDUAL_ACCURACY_VERT = True
PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED = True

if PLOT_INDIVIDUAL_ACCURACY_VERT:
    ## Plot the accuracy of each individual model
    # Mean the pred_correct field over metrics
    grouped_decoder_perf = big_preds.groupby(
        ['metric', 'mouse', 'decode_label'])[
        'pred_correct'].mean().astype(np.float).unstack('mouse').T

    # Insert task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))  
    
    # Define consistent sorting index
    consistent_sort_idx = grouped_decoder_perf.mean(
        level='task').mean(
        level='metric', axis=1).mean().sort_values().index
    
    
    ## Figure handles
    task_l = ['detection', 'discrimination']
    f, axa = plt.subplots(1, len(task_l), figsize=(7.5, 6))
    f.subplots_adjust(left=.3, right=.95, bottom=.05, top=.95, wspace=.2)


    ## Iterate over tasks
    for task in task_l:
        ## Slice data
        mouse_names = task2mouse.loc[task]
        task_gdp = grouped_decoder_perf.loc[task]

        # Mean over mice
        gdp_mouse_mean = task_gdp.mean()
        
        # Unstack outcome and decode_label
        gdp_mouse_mean = gdp_mouse_mean.unstack('decode_label')
        
        # Sort consistently
        gdp_mouse_mean = gdp_mouse_mean.loc[consistent_sort_idx]
        
        # Sort by typ
        decode_label_l = ['rewside', 'choice']
        
        
        ## Slice ax
        ax = axa[task_l.index(task)]
        ax.set_title(task)
        
        # Get yvals
        yvals = pandas.Series(
            range(len(gdp_mouse_mean)), index=gdp_mouse_mean.index)
        
        # Iterate over decode_label
        for n_typ, decode_label in enumerate(decode_label_l):
            if decode_label == 'rewside':
                color = 'green'
            else:
                color = 'magenta'
            
            ax.plot(
                gdp_mouse_mean.loc[:, decode_label].values,
                yvals.values,
                marker='o', color=color, ls='none', mfc='none',
            )
        
        # Plot a horizontal line for each feature
        for yval in yvals.values:
            ax.plot([0, 1], [yval, yval], 'k-', lw=.5)

        
        ## Pretty
        # Chance line
        ax.plot([.5, .5], (-0.5, len(yvals) - 0.5), 'k-', lw=.8)
        
        # yticks and ylim
        ax.set_yticks(yvals.values)
        if ax is axa[0]:
            ax.set_yticklabels(yvals.index.values, size='x-small')
        else:
            ax.set_yticklabels([])
        ax.set_ylim((-0.5, len(yvals) - 0.5))
        
        # xticks
        ax.set_xticks((0, .25, .5, .75, 1))
        ax.set_xlim((.4, 1))
        ax.set_xlabel('decoder accuracy')

        ## Pretty
        my.plot.despine(ax)
    
    
    ## Save
    f.savefig('PLOT_INDIVIDUAL_ACCURACY_VERT.svg')
    f.savefig('PLOT_INDIVIDUAL_ACCURACY_VERT.png')
    
    
    ## Stats
    n_detection_mice = grouped_decoder_perf.loc[    
        'detection'].shape[0]
    n_discrimination_mice = grouped_decoder_perf.loc[    
        'discrimination'].shape[0]
    
    stats_filename = 'STATS__PLOT_INDIVIDUAL_ACCURACY_VERT'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            n_detection_mice + n_discrimination_mice,
            n_detection_mice,
            n_discrimination_mice,
            ))
        fi.write('meaning within mouse concatting all sessions\n')
        fi.write('no error bars\n')
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED:
    task_l = ['discrimination', 'detection']
    
    ## Plot the accuracy of each individual model
    # Mean the pred_correct field over metrics
    grouped_decoder_perf = big_preds.groupby(
        ['metric', 'mouse', 'decode_label'])[
        'pred_correct'].mean().astype(np.float)
    
    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Mean over mice
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['metric', 'task', 'decode_label'])

    # SEM over mice
    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['metric', 'task', 'decode_label'])

    # Get metric alone on index, and task * decode_label on columns
    gdp_mouse_mean = gdp_mouse_mean.unstack(['task', 'decode_label'])
    gdp_mouse_err = gdp_mouse_err.unstack(['task', 'decode_label'])
    
    # Sort
    gdp_mouse_mean = gdp_mouse_mean.sort_index(axis=0).sort_index(axis=1)

    # Sort by overall performance of metric
    gdp_mouse_mean = gdp_mouse_mean.loc[gdp_mouse_mean.mean(1).sort_values().index]
    
    
    ## Drop some metrics
    gdp_mouse_mean = gdp_mouse_mean.drop([
        'contact_count_by_whisker', 
        'contact_count_by_time',
        'anti_angle',
        'anti_angle_max',
        'cycle_duration',
        'anti_contact_duration',
        'anti_frame_start_wrt_peak',
        'velocity2_tip',
        'contact_interaction', 'contact_interaction_count_by_label',
        'frame_start_wrt_peak', 'kappa_max', 'kappa_min', 'protract_ratio',
        'contact_surplus', 
        'xw_angle', # this one is actually better than the other cross-whisker, but still not very good
        'xw_latency_off',
        'xw_duration',
        'anti_contact_count_by_whisker',
        'anti_contact_count_by_time',
        'anti_contact_count_total',
        'touching',
        ])
    gdp_mouse_err = gdp_mouse_err.loc[gdp_mouse_mean.index]
    
    # Reorder, putting contact_count_total first (last on plot)
    gdp_mouse_mean = gdp_mouse_mean.reindex(
        ['contact_count_total'] + [
        val for val in gdp_mouse_mean.index 
        if val != 'contact_count_total'])
    gdp_mouse_err = gdp_mouse_err.loc[gdp_mouse_mean.index]
    
    # Rename
    renaming_dict = {
        'contact_binarized': 'whisk with contact',
        'contact_count_total': 'total contact count',
        'anti_contact_count': 'whisk without contact',
        'angle': 'angle of contact',
        'phase': 'phase of contact',
        'whisking_spread': 'whisking spread',
        'whisking_setpoint': 'whisking set point',
        'contact_duration': 'contact duration',
        'whisking_amplitude': 'whisking amplitude',
        'cycle_duration': 'whisk duration',
        'velocity2_tip': 'contact velocity',
        'kappa_std': 'contact-induced bending',
        'C1vC2_angle': 'cross-whisker contact angle',
        'task': 'task history',
        'xw_latency_on': 'cross-whisker latencies'
        }
    gdp_mouse_mean = gdp_mouse_mean.rename(index=renaming_dict)
    gdp_mouse_err = gdp_mouse_err.rename(index=renaming_dict)
    
    
    ## Create figures
    f, axa = plt.subplots(1, len(task_l), figsize=(8, 2.5))
    f.subplots_adjust(left=.275, right=.925, bottom=.225, top=.875, wspace=.3)
    
    # The y-value of each metric
    yvals = pandas.Series(
        range(len(gdp_mouse_mean)), index=gdp_mouse_mean.index)
    
    
    ## Iterate over marker type
    for decode_label in decode_label_l:
        for task in task_l:
            # Slice task * decode_label (marker type)
            topl = gdp_mouse_mean.loc[:, (task, decode_label)]

            # Marker parameters
            if decode_label == 'rewside':
                color = 'green'
            else:
                color = 'magenta'
            
            
            ## Plot
            ax = axa[task_l.index(task)]
            ax.set_title(task)
            for metric in topl.index:
                xval = topl.loc[metric]
                yval = yvals.loc[metric]
                xerr = gdp_mouse_err.loc[metric, (task, decode_label)]
            
                # Plot the marker
                ax.plot(
                    [xval], [yval],
                    marker='o', mec=color, mfc='none', ls='none', ms=4,
                    mew=1, clip_on=False,
                )

                # Plot the error bar
                ax.plot(
                    [xval - xerr, xval + xerr], [yval, yval],
                    marker=None, ls='-', color=color, lw=1, clip_on=False,
                )

    
    ## Pretty
    for ax in axa:
        ax.set_xlim((0.45, 1))
        ax.set_xticks((.5, .75, 1))
        ax.set_yticks(yvals.values)
        if ax is axa[0]:
            ax.set_yticklabels(yvals.index.values, size=12)
        else:
            ax.set_yticklabels([])
        ax.set_ylim((-0.5, len(yvals) - 0.5))
        ax.plot([0.5, 0.5], (-0.5, len(yvals) - 0.5), 'k-', lw=.75)
        #~ ax.set_xlabel('decoder accuracy')
        my.plot.despine(ax, which=('left', 'top', 'right'))
    
    f.text(.95, .8, 'stimulus', ha='center', va='top', color='g', size=12)
    f.text(.95, .725, 'choice', ha='center', va='top', color='magenta', size=12)
    f.text(.62, .05, 'classifier accuracy', ha='center', va='center', color='k')
    
    
    ## Save
    f.savefig('PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED.svg')
    f.savefig('PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED.png', dpi=300)
    
    
    ## Stats
    n_detection_mice = grouped_decoder_perf.loc[    
        'detection'].unstack('mouse').shape[1]
    n_discrimination_mice = grouped_decoder_perf.loc[    
        'discrimination'].unstack('mouse').shape[1]
    
    stats_filename = 'STATS__PLOT_INDIVIDUAL_ACCURACY_SIMPLIFIED'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            n_detection_mice + n_discrimination_mice,
            n_detection_mice,
            n_discrimination_mice,
            ))
        fi.write('meaning within mouse concatting all sessions\n')
        fi.write('error bars: sem over mice\n')
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))



plt.show()
