# Evaluate cross-decoder
#
# Uses the following data:
#   cross_reduced_model_results_sbs
#   reduced_model_results_sbrc

"""
3F
    COMPARE_ACCURACY_ACROSS_TASK	
    STATS__COMPARE_ACCURACY_ACROSS_TASK	
    Comparison of shape decoding across tasks
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
]


## Paths
# sbrc is the best comparison
# though sbs would be better
# sbrc is not even used for the plot just for sanity check
reduced_results_dir = os.path.join(params['logreg_dir'], 'reduced_model_results_sbrc')
cross_reduced_results_dir = os.path.join(params['logreg_dir'], 'cross_reduced_model_results_sbs')


## Load metadata about sessions
session_df = pandas.read_pickle(os.path.join(params['pipeline_dir'], 'session_df'))
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))
task2mouse = session_df.groupby('task')['mouse'].unique()
mouse2task = session_df[['task', 'mouse']].drop_duplicates().set_index('mouse')['task']


dataset = 'no_opto'


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

# Only rewside is relevant here
reduced_preds = reduced_preds.loc[reduced_preds['decode_label'] == 'rewside']


## Load the cross reduced model predictions
cross_dataset_reduced_results_dir = os.path.join(cross_reduced_results_dir, dataset)
component_names = os.listdir(cross_dataset_reduced_results_dir)

cross_reduced_preds_l = []
for reduced_model in reduced_models:
    # Load results of component model
    model_dir = os.path.join(cross_dataset_reduced_results_dir, reduced_model)

    # Load preds
    preds = pandas.read_pickle(os.path.join(model_dir, 'finalized_predictions'))

    # Store
    cross_reduced_preds_l.append(preds)

# Concat preds
cross_reduced_preds = pandas.concat(cross_reduced_preds_l, 
    keys=reduced_models, axis=0, names=['reduction'])

cross_reduced_preds = cross_reduced_preds.reset_index()

# Add mouse name to index
cross_reduced_preds['mouse'] = cross_reduced_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
cross_reduced_preds = pandas.merge(cross_reduced_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])


## Concat the reduced and cross_reduced
# Extract
reduced_preds_to_concat = reduced_preds.rename(
    {'reduction': 'model_name'}, axis=1)
cross_reduced_preds_to_concat = cross_reduced_preds.rename(
    {'reduction': 'model_name'}, axis=1)


# Concat
all_preds = pandas.concat([
    reduced_preds_to_concat, 
    cross_reduced_preds_to_concat,
    ], 
    keys=['regular', 'cross'],
    axis=0, sort=True, names=['typ', 'index'])

# Label 'mouse_correct', for comparison with 'pred_correct'
all_preds['mouse_correct'] = all_preds['outcome'] == 'hit'


## Calculate accuracy by typ 
# Performance within session
accuracy_by_session = all_preds.groupby(
    ['typ', 'session'])['pred_correct'].mean()

# Add mouse and task
accuracy_by_session = my.misc.insert_mouse_and_task_levels(
    accuracy_by_session, mouse2task)

# Mean within mouse across session
accuracy_by_mouse = accuracy_by_session.mean(level=['task', 'mouse', 'typ'])

# The "regular" decoding is not necessary, just here to check that it
# is the same as the cross-decoding for the discrimination mice
accuracy_by_mouse = accuracy_by_mouse.unstack('typ')['cross']

COMPARE_ACCURACY_ACROSS_TASK = True

if COMPARE_ACCURACY_ACROSS_TASK:
    # Run stats
    pvalue = scipy.stats.ttest_ind(
        accuracy_by_mouse.loc['detection'].values, 
        accuracy_by_mouse.loc['discrimination'].values,
        ).pvalue
    
    # Create figure
    f, ax = plt.subplots(figsize=(2.7, 2))
    f.subplots_adjust(bottom=.225, right=.9, left=.4, top=.8)
    ax.set_title('classifying\nshape identity', pad=8)


    task_l = ['detection', 'discrimination']
    for task in task_l:
        tickloc = 0 if task == 'detection' else 1
        
        # Plot bar to mean
        ax.barh( 
            y=[tickloc],
            width=[accuracy_by_mouse.loc[task].mean()],
            xerr=[accuracy_by_mouse.loc[task].sem()],
            height=.5,
            fc='none',
            ec='k',
            lw=1,
            error_kw={'lw': 1},
        )

        #~ # Plot individual mice
        #~ ax.plot(
            #~ accuracy_by_mouse.loc[task].values,
            #~ [tickloc] * len(accuracy_by_mouse.loc[task]),
            #~ marker='o', ms=4, mfc='none', mec='k', ls='none', mew=.8)

    # Stats line
    ax.plot([1.025, 1.025], [0, 1], 'k-', lw=.8, clip_on=False)
    ax.text(1.05, .4, '***', ha='left', va='center')


    # Pretty
    my.plot.despine(ax)
    ylim = [-.6, 1.6]
    ax.plot([.5, .5], ylim, 'k--', lw=.8)
    ax.set_ylim(ylim)
    ax.set_xticks((0, .25, .5, .75, 1))    
    ax.set_xlim((0.5, 1))
    ax.set_xticklabels(('0.0', '', '0.5', '', '1.0'))
    ax.set_yticks((0, 1))
    ax.set_yticklabels(task_l, size=12)
    ax.set_xlabel('classifier accuracy')

    f.savefig('COMPARE_ACCURACY_ACROSS_TASK.svg')
    f.savefig('COMPARE_ACCURACY_ACROSS_TASK.png', dpi=300)


    ## Stats
    with open('STATS__COMPARE_ACCURACY_ACROSS_TASK', 'w') as fi:
        fi.write('STATS__COMPARE_ACCURACY_ACROSS_TASK\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(accuracy_by_mouse),
            len(accuracy_by_mouse.loc['detection']),
            len(accuracy_by_mouse.loc['discrimination']),
            ))
        fi.write('accuracy mean over mice:\n')
        fi.write(accuracy_by_mouse.mean(level='task').to_string() + '\n')
        fi.write('accuracy sem over mice:\n')
        fi.write(accuracy_by_mouse.sem(level='task').to_string() + '\n')     
        fi.write('unpaired t-test pvalue: {}\n'.format(pvalue))

    with open('STATS__COMPARE_ACCURACY_ACROSS_TASK', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


plt.show()