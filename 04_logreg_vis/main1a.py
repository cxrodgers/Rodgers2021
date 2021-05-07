## Plot results for trial balancing vs no balancing on the full model
# Uses full_model_results_sbrc and full_model_results_sbnull, 
# always the 'no_opto' dataset
"""
S3A	
    COMPARE_ACCURACY_BY_BALANCE_SIMPLER	
    STATS__COMPARE_ACCURACY_BY_BALANCE_SIMPLER	
    Accuracy of full decoder, with and without trial balancing.
"""

import json
import pandas
import numpy as np
import my.plot 
import matplotlib.pyplot as plt
import os

my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))


## Load predictions from dataset
full_model_results_dir = os.path.join(
    params['logreg_dir'], 'full_model_results_sbrc')
dataset = 'no_opto'

# Path to dataset
dataset_results_dir = os.path.join(full_model_results_dir, dataset)

# Load
big_preds = pandas.read_pickle(
    os.path.join(dataset_results_dir, 'finalized_predictions'))
big_preds = big_preds.reset_index()

# Add mouse name to index
big_preds['mouse'] = big_preds['session'].map(session_df['mouse'])

# Merge with trial outcome
big_preds = pandas.merge(big_preds, 
    big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
    on=['session', 'trial'])

# Merge on task
big_preds = pandas.merge(big_preds, session_df['task'], on='session')


## Plots
COMPARE_ACCURACY_BY_BALANCE_SIMPLER = True


if COMPARE_ACCURACY_BY_BALANCE_SIMPLER:
    task_l = ['detection', 'discrimination']
    
    ## Choice errors only
    
    
    ## Load unbalanced predictions from dataset
    full_model_results_dir_null = os.path.join(
        params['logreg_dir'], 'full_model_results_sbnull')
    dataset = 'no_opto'

    # Path to dataset
    dataset_results_dir_null = os.path.join(full_model_results_dir_null, dataset)

    # Load
    big_preds_null = pandas.read_pickle(
        os.path.join(dataset_results_dir_null, 'finalized_predictions'))
    big_preds_null = big_preds_null.reset_index()

    # Add mouse name to index
    big_preds_null['mouse'] = big_preds_null['session'].map(session_df['mouse'])

    # Merge with trial outcome
    big_preds_null = pandas.merge(big_preds_null, 
        big_tm[['rewside', 'servo_pos', 'outcome']].reset_index(), 
        on=['session', 'trial'])

    # Merge on task
    big_preds_null = pandas.merge(big_preds_null, session_df['task'], on='session')


    ## Aggregate by mouse
    # Mean pred_correct within task * mouse * session * decode_label * outcome
    perf_by_session = big_preds.groupby(
        ['task', 'mouse', 'session', 'decode_label', 'outcome']
        )['pred_correct'].mean()

    # Mean across session within mouse
    perf_by_mouse = perf_by_session.mean(
        level=[lev for lev in perf_by_session.index.names if lev != 'session'])

    # Unstack decode_label to get replicates on index
    perf_by_mouse = perf_by_mouse.unstack(['decode_label', 'outcome'])



    ## Aggregate by mouse
    # Mean pred_correct within task * mouse * session * decode_label * outcome
    perf_by_session_null = big_preds_null.groupby(
        ['task', 'mouse', 'session', 'decode_label', 'outcome']
        )['pred_correct'].mean()

    # Mean across session within mouse
    perf_by_mouse_null = perf_by_session_null.mean(
        level=[lev for lev in perf_by_session_null.index.names if lev != 'session'])

    # Unstack decode_label to get replicates on index
    perf_by_mouse_null = perf_by_mouse_null.unstack(['decode_label', 'outcome'])

    
    ## Plot
    f, ax = my.plot.figure_1x1_square()
    decode_label = 'choice'
    outcome = 'error'

    for task in task_l:
        if task == 'detection':
            marker = 'x'
        else:
            marker = 'o'
        
        ax.plot(
            perf_by_mouse_null.loc[task][(decode_label, outcome)].values,
            perf_by_mouse.loc[task][(decode_label, outcome)].values, 
            marker=marker, mec='k', mfc='none', ls='none')
        
    ax.plot([0, 1], [0, 1], 'k-', lw=.8)
    ax.plot([0, 1], [0.5, 0.5], 'k-', lw=.8)
    ax.plot([0.5, 0.5], [0, 1], 'k-', lw=.8)
    
    ax.axis('square')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel('accuracy without balancing')
    ax.set_ylabel('accuracy with balancing')
    
    
    ## Save
    f.savefig('COMPARE_ACCURACY_BY_BALANCE_SIMPLER.svg')
    f.savefig('COMPARE_ACCURACY_BY_BALANCE_SIMPLER.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__COMPARE_ACCURACY_BY_BALANCE_SIMPLER'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('accuracy of full model, with and without balancing\n')
        fi.write('accuracy on decoding choice from error trials only\n')
        fi.write('aggregating first within session, then within mouse\n')
        fi.write('n = {} mice ({} det., {} disc.)\n'.format(
            len(perf_by_mouse),
            len(perf_by_mouse.loc['detection']),
            len(perf_by_mouse.loc['discrimination']),
            ))
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))
    
plt.show()