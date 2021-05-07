## Plot decoding model performance
# Compare reduced models with full and individual
# Uses the following data:
#   full_model_results_sbrc
#   indiv_model_results
#   reduced_model_results_sbrc

"""
3B
    PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL
    STATS__PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL
    Headline performance summary: OBD on both tasks, both decode targets, both outcomes

3C
    COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK_discrimination
    STATS__COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK_discrimination
    "Additive" comparison of reduced models for discrimination only

S3E
    COMPARE_SUBTRACTIVE_REDUCED_MODELS_SIMPLIFIED
    STATS__COMPARE_SUBTRACTIVE_REDUCED_MODELS_SIMPLIFIED
    "Subtractive" comparison of reduced models for discrimination only

S3F
    COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_SIMPLIFIED
    STATS__COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_SIMPLIFIED
    "Whisker dropping" comparison of reduced models for discrimination only

3E
    PLOT_MOUSE_VS_MODEL_PERF_SMALLER
    STATS__PLOT_MOUSE_VS_MODEL_PERF_SMALLER
    Comparison of mouse performance and decoder accuracy
"""


import json
import pandas
import scipy.stats
import numpy as np
import my.plot 
import matplotlib.pyplot as plt
import matplotlib
import os

my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Which models to load
reduced_models = [
    'contact_binarized+anti_contact_count+angle',
    'contact_binarized+anti_contact_count',
    'full*',
    'full*-contact_binarized',
    'full*-anti_contact_count',
    'full*-angle',   
    'OBD-C0',
    'OBD-C0-C1',
    'OBD-C0-C2',
    'OBD-C0-C3',
]


## Paths
full_results_dir = os.path.join(params['logreg_dir'], 'full_model_results_sbrc')
indiv_results_dir = os.path.join(params['logreg_dir'], 'indiv_model_results')
reduced_results_dir = os.path.join(params['logreg_dir'], 'reduced_model_results_sbrc')


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))

dataset = 'no_opto'
task_l = ['discrimination', 'detection']
decode_label_l = ['rewside', 'choice']


## Load full results
dataset_full_results_dir = os.path.join(full_results_dir, dataset)

# Load the full model predictions
full_preds = pandas.read_pickle(os.path.join(dataset_full_results_dir, 
    'finalized_predictions'))
full_preds = full_preds.reset_index()

# Add mouse name to index
full_preds['mouse'] = full_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
full_preds = pandas.merge(full_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])


## Load the individual model predictions
# Where the individual results are
dataset_indiv_results_dir = os.path.join(indiv_results_dir, dataset)
component_names = os.listdir(dataset_indiv_results_dir)

preds_l = []
preds_keys_l = []
for component in component_names:
    ## Load results of component model
    model_dir = os.path.join(dataset_indiv_results_dir, component)

    # Load preds
    try:
        preds = pandas.read_pickle(
            os.path.join(model_dir, 'finalized_predictions'))
    except IOError:
        continue

    # Store preds
    preds_l.append(preds)
    preds_keys_l.append(component)

# Concat preds
big_preds = pandas.concat(preds_l, keys=preds_keys_l, axis=0, names=['metric'])
big_preds = big_preds.reset_index()

# Add mouse name to index
big_preds['mouse'] = big_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
indiv_preds = pandas.merge(big_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])


## Load the reduced model predictions
dataset_reduced_results_dir = os.path.join(reduced_results_dir, dataset)
component_names = os.listdir(dataset_reduced_results_dir)

reduced_preds_l = []
for reduced_model in reduced_models:
    # Load results of component model
    model_dir = os.path.join(dataset_reduced_results_dir, reduced_model)

    # Load preds
    preds = pandas.read_pickle(os.path.join(model_dir, 'finalized_predictions'))

    # Store
    reduced_preds_l.append(preds)

# Concat preds
reduced_preds = pandas.concat(reduced_preds_l, 
    keys=reduced_models, axis=0, names=['reduction'])

reduced_preds = reduced_preds.reset_index()

# Add mouse name to index
reduced_preds['mouse'] = reduced_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
reduced_preds = pandas.merge(reduced_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])


## Concat the indiv, full, and reduced
# Extract
indiv_preds_to_concat = indiv_preds.rename(
    {'metric': 'model_name'}, axis=1)
reduced_preds_to_concat = reduced_preds.rename(
    {'reduction': 'model_name'}, axis=1)
full_preds_to_concat = full_preds.copy()
full_preds_to_concat['model_name'] = 'full'


# Concat
all_preds = pandas.concat([
    indiv_preds_to_concat, 
    reduced_preds_to_concat, 
    full_preds_to_concat], 
    axis=0, ignore_index=True, sort=True)

# Label 'mouse_correct', for comparison with 'pred_correct'
all_preds['mouse_correct'] = all_preds['outcome'] == 'hit'


## Plots
# This one is the headline performance plot, using the optimized model for each task
PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL = True

COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK = True
COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_SIMPLIFIED = True
COMPARE_SUBTRACTIVE_REDUCED_MODELS_SIMPLIFIED = True
PLOT_MOUSE_VS_MODEL_PERF_SMALLER = True

# Human readable feature names
renaming_dict = {
    'contact_binarized': 
        'whisks with contact',
    'contact_binarized+anti_contact_count': 
        'above +\nwhisks without contact',
    'contact_binarized+anti_contact_count+angle': 
        'above +\nangle of contact',
    'contact_binarized+anti_contact_count+angle+anti_angle_max': 
        'above +\npeak angle without contact',
    'contact_count_total': 'total contact count',
    'contact_count_by_whisker': 'total contacts by each whisker',
    'full*-contact_binarized': 'whisks with contact',
    'full*-anti_contact_count': 'whisks without contact',
    'full*-angle': 'contact angle',
    'OBD-C0-C1': 'C1',
    'OBD-C0-C2': 'C2',
    'OBD-C0-C3': 'C3',
    }
    

## Plot accuracy of OBD performance by decode label and outcome
if PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL:
    
    ## Calculate model performance by task * mouse * model
    grouped_decoder_perf = all_preds.groupby(
        ['mouse', 'session', 'model_name', 'decode_label', 'outcome'])[
        'pred_correct'].mean().astype(np.float)

    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Aggregate sessions within mouse
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['model_name', 'task', 'mouse', 'decode_label', 'outcome']
        ).sort_index()

    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['model_name', 'task', 'mouse', 'decode_label', 'outcome']
        ).sort_index()

    
    ## Extract relevant model only
    sliced_mean_l = []
    sliced_err_l = []
    for task in task_l:
        # Choose model for this task
        if task == 'discrimination':
            model_name = 'contact_binarized+anti_contact_count+angle'
        elif task == 'detection':
            model_name = 'contact_binarized+anti_contact_count+angle'
        else:
            1/0        
        
        # Slice
        sliced_mean = gdp_mouse_mean.xs(
            model_name, level='model_name', drop_level=False).xs(
            task, level='task', drop_level=False)
        sliced_err = gdp_mouse_err.xs(
            model_name, level='model_name', drop_level=False).xs(
            task, level='task', drop_level=False)
            
        # Store
        sliced_mean_l.append(sliced_mean)
        sliced_err_l.append(sliced_err)
    
    # Concat and drop now redundant model level
    gdp_mouse_mean = pandas.concat(
        sliced_mean_l).sort_index().droplevel('model_name').sort_index()
    gdp_mouse_err = pandas.concat(
        sliced_err_l).sort_index().droplevel('model_name').sort_index()
    
    # Unstack decode_label * outcome
    perf_by_mouse = gdp_mouse_mean.unstack(['decode_label', 'outcome'])
    err_by_mouse = gdp_mouse_err.unstack(['decode_label', 'outcome'])
    
    
    ## Headline performance (used only in STATS)
    ## This one is different because it doesn't break out by outcome
    # Slice optimized model (Make sure it matches above)
    sliced_preds = all_preds[
        all_preds['model_name'] == 'contact_binarized+anti_contact_count+angle'].copy()
    
    # Add task level
    sliced_preds['task'] = sliced_preds['mouse'].map(mouse2task)
    
    # Mean pred_correct within task * mouse * session * decode_label
    headline_perf_by_session = sliced_preds.groupby(
        ['task', 'mouse', 'session', 'decode_label']
        )['pred_correct'].mean()
    
    # Mean across session within mouse
    headline_perf_by_mouse = headline_perf_by_session.mean(
        level=[lev for lev in headline_perf_by_session.index.names 
        if lev != 'session'])
    
    # Mean across mice within task
    headline_perf_by_task = headline_perf_by_mouse.mean(
        level=[lev for lev in headline_perf_by_mouse.index.names 
        if lev != 'mouse'])

    # STD across mice within task
    headline_std_by_task = headline_perf_by_mouse.std(
        level=[lev for lev in headline_perf_by_mouse.index.names 
        if lev != 'mouse'])
        
    # SEM across mice within task
    headline_sem_by_task = headline_perf_by_mouse.sem(
        level=[lev for lev in headline_perf_by_mouse.index.names 
        if lev != 'mouse'])
    

    ## Aggregate across mice
    perf_agg = perf_by_mouse.mean(level='task')
    
    # SEM for STATS
    perf_err = perf_by_mouse.sem(level='task')
    perf_std = perf_by_mouse.std(level='task')
    
    # Order bars in this way
    ordering_midx = pandas.MultiIndex.from_product([
        ['rewside', 'choice'], ['hit', 'error']], 
        names=['decode_label', 'outcome'])


    ## Plot handles
    f, axa = my.plot.figure_1x2_small()
    f.subplots_adjust(wspace=.3, right=.975)
    task_l = ['discrimination', 'detection']

    # Iterate over task (axis)
    for task in task_l:
        # Get ax
        ax = axa[task_l.index(task)]
        ax.set_title(task, pad=17.5)
        
        # Slice and order
        topl_agg = perf_agg.loc[task].copy()
        topl_agg = topl_agg.loc[ordering_midx]
        topl_err = perf_err.loc[task].copy()
        topl_err = topl_err.loc[ordering_midx]

        # generate colors
        bar_edge_colors = []
        bar_face_colors = []
        for idx in topl_agg.index:
            if idx[1] == 'hit':
                color = 'g' if idx[0] == 'rewside' else 'magenta'
                bar_edge_colors.append(color)
                bar_face_colors.append(color)
            
            else:
                color = 'g' if idx[0] == 'rewside' else 'magenta'
                bar_edge_colors.append(color)
                bar_face_colors.append('white')    

        # x-ticks
        # this is why we can't use grouped_bar_plot
        bar_x = [0, 1, 2.5, 3.5]
        
        # Plot the bars of mean performance
        bars = ax.bar(
            bar_x, topl_agg, lw=1,
            yerr=topl_err,
            error_kw={'linewidth': .75},
        )

        # Set bar colors
        for n_bar, bar in enumerate(bars):
            bar.set_edgecolor(bar_edge_colors[n_bar])
            bar.set_facecolor(bar_face_colors[n_bar])
        
        # Pretty
        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            topl_agg.index.get_level_values('outcome'), size=12)
        ax.set_xticklabels([])
        my.plot.despine(ax)
        ax.set_ylim((.25, 1))
        ax.set_yticks((.25, .5, .75, 1))
        ax.plot([-1, np.max(bar_x) + 1], [.5, .5], 'k--', lw=.75)
        ax.set_xlim((-.5, np.max(bar_x) + .5))

        # Label stimulus and choice
        ax.text(np.mean(ax.get_xticks()[:2]) + .1, .1, 'stimulus', 
            color='g', ha='center', va='center', size=14)
        ax.text(np.mean(ax.get_xticks()[2:]) + .1, .1, 'choice', 
            color='magenta', ha='center', va='center', size=14)

        # Label N
        ax.text(
            ax.get_xlim()[1] + .3, 1.05, 'n = {} mice'.format(len(indiv_mice)), 
            ha='right', va='center', size=12)

    # Labels
    axa[0].set_ylabel('classifier accuracy')
    axa[1].set_yticklabels([])


    ## Save
    f.savefig('PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL.svg')
    f.savefig('PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL.png', dpi=300)

    
    ## Stats
    stats_filename = 'STATS__PLOT_OBD_PERFORMANCE_BY_TASK_OUTCOME_AND_DECODELABEL'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('contact_binarized+anti_contact_count+angle model only\n')
        
        for task in ['detection', 'discrimination']:
            fi.write(task + '\n-----\n')
            fi.write('mean within session, then within mouse\n')
            fi.write('n = {} mice\n'.format(
                len(perf_by_mouse.loc[task])))

            fi.write('performance by decode_label * outcome, meaned over mice:\n')
            fi.write(perf_agg.loc[task].to_string() + '\n\n')
            
            fi.write('performance by decode_label * outcome, SEM over mice:\n')
            fi.write(perf_err.loc[task].to_string() + '\n\n')

            fi.write('performance over all trials, meaned over mice:\n')
            fi.write(headline_perf_by_task.loc[task].to_string() + '\n\n')

            fi.write('performance over all trials, SEM over mice:\n')
            fi.write(headline_sem_by_task.loc[task].to_string() + '\n\n')

            fi.write('\n')
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))
    


if COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK:
    ## This one does different reductions for each task
    grouped_decoder_perf = all_preds.groupby(
        ['mouse', 'model_name', 'decode_label'])[
        'pred_correct'].mean().astype(np.float)

    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Mean over mice
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['model_name', 'task', 'decode_label'])

    # SEM over mice
    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['model_name', 'task', 'decode_label'])

    # Get metric alone on index, and task * decode_label on columns
    gdp_mouse_mean = gdp_mouse_mean.unstack(['task', 'decode_label'])
    gdp_mouse_err = gdp_mouse_err.unstack(['task', 'decode_label'])
    
    # Sort
    gdp_mouse_mean = gdp_mouse_mean.sort_index(axis=0).sort_index(axis=1)


    ## Iterate over marker type
    for task in ['discrimination']: #task_l:
        ## Choose the models to plot
        if task == 'discrimination':
            task_gdp_mouse_mean = gdp_mouse_mean.loc[[
                'contact_binarized',
                'contact_binarized+anti_contact_count',
                'contact_binarized+anti_contact_count+angle',
                #~ 'contact_binarized+anti_contact_count+angle+anti_angle_max', # not really better
                'full',
                #~ 'contact_count_total',
                ]]
            task_gdp_mouse_err = gdp_mouse_err.loc[task_gdp_mouse_mean.index]
        

        task_gdp_mouse_mean = task_gdp_mouse_mean.rename(index=renaming_dict)
        task_gdp_mouse_err = task_gdp_mouse_err.rename(index=renaming_dict)


    
        ## Create figures
        f, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))
        f.subplots_adjust(bottom=.225, left=.45, right=.95)
        
        # The y-value of each metric
        yvals = pandas.Series(
            range(len(task_gdp_mouse_mean)), index=task_gdp_mouse_mean.index)
            
        
        ## Iterate over decode labels
        for decode_label in decode_label_l:
        
            # Slice task * decode_label (marker type)
            topl = task_gdp_mouse_mean.loc[:, (task, decode_label)]

            # Marker parameters
            if decode_label == 'rewside':
                color = 'green'
            else:
                color = 'magenta'
            
            
            ## Plot
            #~ ax = axa[task_l.index(task)]
            #~ ax.set_title(task)
            for metric in topl.index:
                xval = topl.loc[metric]
                yval = yvals.loc[metric]
                xerr = task_gdp_mouse_err.loc[metric, (task, decode_label)]
            
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
        ax.set_xlim((0.45, 1))
        ax.set_xticks((.5, .75, 1))
        ax.set_yticks(yvals.values)
        ax.set_yticklabels(yvals.index.values, linespacing=1, size=12)
        ax.set_ylim((len(yvals) - 0.5, -0.5))
        ax.plot([0.5, 0.5], (-0.5, len(yvals) - 0.5), 'k-', lw=.75)
        ax.set_xlabel('classifier accuracy')
        my.plot.despine(ax, which=('left', 'top', 'right'))
        ax.set_title('discrimination')

        f.text(1.05, .8, 'stimulus', ha='center', va='top', color='g', size=12)
        f.text(1.05, .725, 'choice', ha='center', va='top', color='magenta', size=12)
        
        
        ## Save
        f.savefig(
            'COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK_{}.svg'.format(task))
        f.savefig(
            'COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK_{}.png'.format(task), 
            dpi=300)
        
        
        ## Stats
        stats_filename = (
            'STATS__COMPARE_REDUCED_MODELS_SIMPLIFIED_BY_TASK_{}'.format(task))
        with open(stats_filename, 'w') as fi:
            stats_df = grouped_decoder_perf.loc[task]
            n_mice = len(stats_df.index.get_level_values('mouse').unique())
            fi.write(stats_filename + '\n')
            fi.write('n = {} mice\n'.format(n_mice))
            fi.write('error bars: sem over mice\n')
    
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))


if COMPARE_SUBTRACTIVE_REDUCED_MODELS_SIMPLIFIED:
    ## This one subtracts one feature at a time
    grouped_decoder_perf = all_preds.groupby(
        ['mouse', 'model_name', 'decode_label'])[
        'pred_correct'].mean().astype(np.float)

    # Normalize by full*
    # full* removes all of the features that are simple linear summations
    # and combinations of the features under test here
    # Similar results are obtained with 'full', but smaller uniquely explained
    # variance
    # Somewhat similar results can also be obtained with the OBD
    grouped_decoder_perf = -grouped_decoder_perf.sub(grouped_decoder_perf.xs(
        'full*', 
        level=1))

    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Mean over mice
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['model_name', 'task', 'decode_label'])

    # SEM over mice
    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['model_name', 'task', 'decode_label'])

    # Get metric alone on index, and task * decode_label on columns
    gdp_mouse_mean = gdp_mouse_mean.unstack(['task', 'decode_label'])
    gdp_mouse_err = gdp_mouse_err.unstack(['task', 'decode_label'])
    
    # Sort
    gdp_mouse_mean = gdp_mouse_mean.sort_index(axis=0).sort_index(axis=1)


    ## Iterate over marker type
    for task in ['discrimination']: #task_l:
        ## Choose the models to plot
        if task == 'discrimination':
            task_gdp_mouse_mean = gdp_mouse_mean.loc[[
                'full*-contact_binarized',
                'full*-anti_contact_count',
                'full*-angle',
                ]]
            task_gdp_mouse_err = gdp_mouse_err.loc[task_gdp_mouse_mean.index]

        task_gdp_mouse_mean = task_gdp_mouse_mean.rename(index=renaming_dict)
        task_gdp_mouse_err = task_gdp_mouse_err.rename(index=renaming_dict)

    
        ## Create figures
        f, ax = plt.subplots(1, 1, figsize=(3.75, 2.5))
        f.subplots_adjust(bottom=.3, left=.48, right=.95)
        
        # The y-value of each metric
        yvals = pandas.Series(
            range(len(task_gdp_mouse_mean)), index=task_gdp_mouse_mean.index)
            
        
        ## Iterate over decode labels
        for decode_label in decode_label_l:
        
            # Slice task * decode_label (marker type)
            topl = task_gdp_mouse_mean.loc[:, (task, decode_label)]

            # Marker parameters
            if decode_label == 'rewside':
                color = 'green'
            else:
                color = 'magenta'
            
            
            ## Plot
            #~ ax = axa[task_l.index(task)]
            #~ ax.set_title(task)
            for metric in topl.index:
                xval = topl.loc[metric]
                yval = yvals.loc[metric]
                xerr = task_gdp_mouse_err.loc[metric, (task, decode_label)]
            
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
        ax.set_xlim((0, .06))
        ax.set_xticks((0, .05))
        ax.set_yticks(yvals.values)
        ax.set_yticklabels(yvals.index.values, linespacing=1, size=12)
        ax.set_ylim((len(yvals) - 0.5, -0.5))
        ax.plot([0, 0], (-0.5, len(yvals) - 0.5), 'k-', lw=.75)
        ax.set_xlabel('unique contribution\n-{}(classifier accuracy)'.format(chr(916)))
        my.plot.despine(ax, which=('left', 'top', 'right'))
        
        
        ## Save
        f.savefig(
            'COMPARE_SUBTRACTIVE_REDUCED_MODELS_{}.svg'.format(task))
        f.savefig(
            'COMPARE_SUBTRACTIVE_REDUCED_MODELS_{}.png'.format(task), 
            dpi=300)
        
        
        ## Stats
        stats_filename = (
            'STATS__COMPARE_SUBTRACTIVE_REDUCED_MODELS_SIMPLIFIED_{}'.format(task))
        with open(stats_filename, 'w') as fi:
            stats_df = grouped_decoder_perf.loc[task]
            n_mice = len(stats_df.index.get_level_values('mouse').unique())
            fi.write(stats_filename + '\n')
            fi.write('n = {} mice\n'.format(n_mice))
            fi.write('error bars: sem over mice\n')
    
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))

if COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_SIMPLIFIED:
    ## This one subtracts one feature at a time
    grouped_decoder_perf = all_preds.groupby(
        ['mouse', 'model_name', 'decode_label'])[
        'pred_correct'].mean().astype(np.float)

    # Normalize by full*
    # full* removes all of the features that are simple linear summations
    # and combinations of the features under test here
    # Similar results are obtained with 'full', but smaller uniquely explained
    # variance
    # Somewhat similar results can also be obtained with the OBD
    grouped_decoder_perf = -grouped_decoder_perf.sub(grouped_decoder_perf.xs(
        'OBD-C0', 
        level=1))

    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Mean over mice
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['model_name', 'task', 'decode_label'])

    # SEM over mice
    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['model_name', 'task', 'decode_label'])

    # Get metric alone on index, and task * decode_label on columns
    gdp_mouse_mean = gdp_mouse_mean.unstack(['task', 'decode_label'])
    gdp_mouse_err = gdp_mouse_err.unstack(['task', 'decode_label'])
    
    # Sort
    gdp_mouse_mean = gdp_mouse_mean.sort_index(axis=0).sort_index(axis=1)


    ## Iterate over marker type
    for task in ['discrimination']: #task_l:
        ## Choose the models to plot
        if task == 'discrimination':
            task_gdp_mouse_mean = gdp_mouse_mean.loc[[
                'OBD-C0-C1',
                'OBD-C0-C2',
                'OBD-C0-C3',
                ]]
            task_gdp_mouse_err = gdp_mouse_err.loc[task_gdp_mouse_mean.index]

        task_gdp_mouse_mean = task_gdp_mouse_mean.rename(index=renaming_dict)
        task_gdp_mouse_err = task_gdp_mouse_err.rename(index=renaming_dict)

    
        ## Create figures
        f, ax = plt.subplots(1, 1, figsize=(3.75, 2.5))
        f.subplots_adjust(bottom=.3, left=.48, right=.95)
        
        # The y-value of each metric
        yvals = pandas.Series(
            range(len(task_gdp_mouse_mean)), index=task_gdp_mouse_mean.index)
            
        
        ## Iterate over decode labels
        for decode_label in decode_label_l:
        
            # Slice task * decode_label (marker type)
            topl = task_gdp_mouse_mean.loc[:, (task, decode_label)]

            # Marker parameters
            if decode_label == 'rewside':
                color = 'green'
            else:
                color = 'magenta'
            
            
            ## Plot
            #~ ax = axa[task_l.index(task)]
            #~ ax.set_title(task)
            for metric in topl.index:
                xval = topl.loc[metric]
                yval = yvals.loc[metric]
                xerr = task_gdp_mouse_err.loc[metric, (task, decode_label)]
            
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
        ax.set_xlim((0, .1))
        ax.set_xticks((0, .05, .1))
        ax.set_yticks(yvals.values)
        ax.set_yticklabels(yvals.index.values, linespacing=1, size=12)
        ax.set_ylim((len(yvals) - 0.5, -0.5))
        ax.plot([0, 0], (-0.5, len(yvals) - 0.5), 'k-', lw=.75)
        ax.set_xlabel('unique contribution\n-{}(classifier accuracy)'.format(chr(916)))
        my.plot.despine(ax, which=('left', 'top', 'right'))
        
        
        ## Save
        f.savefig(
            'COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_{}.svg'.format(task))
        f.savefig(
            'COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_{}.png'.format(task), 
            dpi=300)
        
        
        ## Stats
        stats_filename = (
            'STATS__COMPARE_SUBTRACTIVE_WHISKER_REDUCED_MODELS_SIMPLIFIED_{}'.format(task))
        with open(stats_filename, 'w') as fi:
            stats_df = grouped_decoder_perf.loc[task]
            n_mice = len(stats_df.index.get_level_values('mouse').unique())
            fi.write(stats_filename + '\n')
            fi.write('n = {} mice\n'.format(n_mice))
            fi.write('error bars: sem over mice\n')
    
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))

    
if PLOT_MOUSE_VS_MODEL_PERF_SMALLER:
    ## Plot model perf vs mouse perf 
    #  (for the director's cut reduced_model for that task)


    ## Calculate model performance by task * mouse * model
    # This time we also include mouse_correct for comparison to pred_correct
    # And we separately consider sessions so we can put error bars on indiv mice
    grouped_decoder_perf = all_preds.groupby(
        ['mouse', 'session', 'model_name', 'decode_label'])[
        ['mouse_correct', 'pred_correct']].mean().astype(np.float)
    grouped_decoder_perf.columns.name = 'which_correct'

    # Add a task level
    grouped_decoder_perf = my.misc.insert_level(
        grouped_decoder_perf, name='task', 
        func=lambda idx: idx['mouse'].map(mouse2task))

    # Aggregate sessions within mouse
    gdp_mouse_mean = grouped_decoder_perf.mean(
        level=['model_name', 'task', 'mouse', 'decode_label']).sort_index()

    gdp_mouse_err = grouped_decoder_perf.sem(
        level=['model_name', 'task', 'mouse', 'decode_label']).sort_index()

    
    ## Extract relevant model only
    sliced_mean_l = []
    sliced_err_l = []
    for task in task_l:
        # Choose model for this task
        if task == 'discrimination':
            model_name = 'contact_binarized+anti_contact_count+angle'
        elif task == 'detection':
            #~ model_name = 'contact_count_total'
            model_name = 'contact_binarized+anti_contact_count+angle'
        else:
            1/0        
        
        # Slice
        sliced_mean = gdp_mouse_mean.xs(
            model_name, level='model_name', drop_level=False).xs(
            task, level='task', drop_level=False)
        sliced_err = gdp_mouse_err.xs(
            model_name, level='model_name', drop_level=False).xs(
            task, level='task', drop_level=False)
            
        # Store
        sliced_mean_l.append(sliced_mean)
        sliced_err_l.append(sliced_err)
    
    # Concat and drop now redundant model level
    gdp_mouse_mean = pandas.concat(
        sliced_mean_l).sort_index().droplevel('model_name').sort_index()
    gdp_mouse_err = pandas.concat(
        sliced_err_l).sort_index().droplevel('model_name').sort_index()


    ## Stats
    stats_l = []
    stats_keys_l = []
    for task in ['detection', 'discrimination', 'both']:
        for decode_label in ['rewside', 'choice']:
            if task == 'both':
                xdata = gdp_mouse_mean.loc[:, 'mouse_correct'].xs(
                    decode_label, level='decode_label').values
                ydata = gdp_mouse_mean.loc[:, 'pred_correct'].xs(
                    decode_label, level='decode_label').values
            else:
                xdata = gdp_mouse_mean.loc[task, 'mouse_correct'].xs(
                    decode_label, level='decode_label').values
                ydata = gdp_mouse_mean.loc[task, 'pred_correct'].xs(
                    decode_label, level='decode_label').values
            
            # Linear regression
            linres = scipy.stats.linregress(xdata, ydata)
            
            # t-test
            ttest_res = scipy.stats.ttest_rel(xdata, ydata)
            
            # Store
            stats_l.append((linres.pvalue, ttest_res.pvalue))
            stats_keys_l.append((task, decode_label))
    
    stats_df = pandas.DataFrame.from_records(stats_l, columns=['linp', 'ttp'])
    stats_df.index = pandas.MultiIndex.from_tuples(
        stats_keys_l, names=['task', 'decode_label'])


    ## Plot handles
    f, axa = my.plot.figure_1x2_small(sharex=True, sharey=True)
    
    
    ## Iterate over task (axes)
    for task in task_l:
        # Choose model for this task
        if task == 'discrimination':
            marker = 'o'
            mfc = 'none'
            mec = 'k'
        elif task == 'detection':
            marker = 'x'
            mfc = 'none'
            mec = 'k'
        else:
            1/0
        
        # Iterate over decode label
        for decode_label in decode_label_l:
            # Get ax
            ax = axa[decode_label_l.index(decode_label)]
            if decode_label == 'rewside':
                ax.set_title('classifying\nstimulus', pad=8)
            else:
                ax.set_title('classifying\nchoice', pad=8)

            # Get data
            topl_mean = gdp_mouse_mean.xs(
                task, level='task').xs(
                decode_label, level='decode_label')
            topl_err = gdp_mouse_err.xs(
                task, level='task').xs(
                decode_label, level='decode_label')
            
            # Iterate over mice
            for mouse in topl_mean.index:
                # Plot
                ax.plot(
                    [topl_mean.loc[mouse, 'mouse_correct']],
                    [topl_mean.loc[mouse, 'pred_correct']],
                    #~ xerr=[topl_err.loc[mouse, 'mouse_correct']],
                    #~ yerr=[topl_err.loc[mouse, 'pred_correct']],
                    ms=6, marker=marker,
                    lw=1, mew=.8,
                    mfc=mfc, mec=mec,
                    )


    ## Pretty
    for ax in axa.flatten():
        # Unity line
        ax.plot([0, 1], [0, 1], 'k:', lw=.8)
        
        # Pretty
        ax.axis('square')
        my.plot.despine(ax)
        
        # Limits and ticks
        ax.set_xlim((.5, 1))
        ax.set_ylim((.5, 1))
        ax.set_xticks((.5, .75, 1))
        ax.set_xticklabels(('0.5', '', '1.0'))
        ax.set_yticks((.5, .75, 1))
        ax.set_yticklabels(('0.5', '', '1.0'))

    # Axis labels
    axa[0].set_ylabel('classifier\naccuracy')
    f.text(.6, .05, 'mouse performance', ha='center', va='center')
    
    f.savefig('PLOT_MOUSE_VS_MODEL_PERF_SMALLER.svg')
    f.savefig('PLOT_MOUSE_VS_MODEL_PERF_SMALLER.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_MOUSE_VS_MODEL_PERF_SMALLER'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n\n')
        fi.write('contact_binarized+anti_contact_count+angle model only\n')
        fi.write('linear regression on classifier accuracy versus mouse perf:\n')
        fi.write(stats_df['linp'].unstack('decode_label').to_string() + '\n\n')
        
        fi.write('2-sample t-test on classifier accuracy versus mouse perf:\n')
        fi.write(stats_df['ttp'].unstack('decode_label').to_string() + '\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))
    
plt.show()
