## Plot the results of main1c1 and main1c2

import json
import os
import matplotlib.pyplot as plt
import pandas
import numpy as np
import my.plot
import statsmodels.stats.proportion
import scipy.stats


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Paths
neural_decoding_dir = os.path.join(params['neural_dir'], 'decoding')


## Plots
my.plot.manuscript_defaults()
my.plot.font_embed()


## Helper function for calculating error bars
def aggregate_accuracy(accuracy, method=0):
    # Estimate 95% CI using +/- 2 stdev over boots
    # This would fail near 1.0 where the error bars become asymmetric
    if method == 0:
        # Unstack
        unstacked = accuracy.unstack(['decode_label', 'bin'])
        mean = unstacked.mean()
        
        # Std
        std = unstacked.std()
        
        # 95% of the mass is contained within +/- 1.96 sigma of the mean
        # Here is how to calculate that threshold exactly
        # Check: norm_rv.cv(include_sigma) - norm_rv.cv(-include_sigma) = 0.95
        norm_rv = scipy.stats.norm()
        include_sigma = norm_rv.ppf(.975)
        
        # Calculate confidence bounds
        lo = mean - include_sigma * std
        hi = mean + include_sigma * std 
        
        # Concat
        res = pandas.concat(
            [mean, std, lo, hi], keys=['mean', 'std', 'lo', 'hi'], axis=1)

    # Actually calculate prctiles over boots
    # To be precise, this would require many more boots
    # In practice, basically the same as above method
    elif method == 2:
        unstacked = accuracy.unstack(['decode_label', 'bin'])
        mean = unstacked.mean()

        lo = unstacked.quantile(.025)
        hi = unstacked.quantile(.975)

        res = pandas.concat(
            [mean, lo, hi], keys=['mean', 'lo', 'hi'], axis=1)
    
    return res


## Iterate over regular analysis and no-lick analysis
for analysis_typ in ['all_trials', 'no_licks1']:

    ## Load
    # Form paths
    all_choice_path = os.path.join(neural_decoding_dir, 
        '{}-{}-{}'.format(analysis_typ, 'all', 'choice'))
    all_rewside_path = os.path.join(neural_decoding_dir, 
        '{}-{}-{}'.format(analysis_typ, 'all', 'rewside'))
    all_both_path = os.path.join(neural_decoding_dir, 
        '{}-{}-{}'.format(analysis_typ, 'all', 'both'))
    nfc_both_path = os.path.join(neural_decoding_dir, 
        '{}-{}-{}'.format(analysis_typ, 'no_frontier_crossings', 'both'))

    # Load
    acc_all_choice = pandas.read_pickle(
        os.path.join(all_choice_path, 'meaned_accuracy'))
    acc_all_rewside = pandas.read_pickle(
        os.path.join(all_rewside_path, 'meaned_accuracy'))
    acc_all_both = pandas.read_pickle(
        os.path.join(all_both_path, 'meaned_accuracy'))
    acc_nfc_both = pandas.read_pickle(
        os.path.join(nfc_both_path, 'meaned_accuracy'))
    
    # Load bins (assume same for all)
    bins = my.misc.pickle_load(os.path.join(all_both_path, 'bins'))
    bincenters = (bins[:-1] + bins[1:]) / 2.0
    bincenters_t = bincenters / 200.
    
    # Load meaned weights, just to get the number of neurons
    meaned_weights = pandas.read_pickle(
        os.path.join(all_both_path, 'meaned_weights'))
    n_neurons = meaned_weights.shape[1]


    ## Aggregate
    agged_all_choice = aggregate_accuracy(acc_all_choice)
    agged_all_rewside = aggregate_accuracy(acc_all_rewside)
    agged_all_both = aggregate_accuracy(acc_all_both)
    agged_nfc_both = aggregate_accuracy(acc_nfc_both)


    ## Plot
    f, axa = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(5.5, 2.25))
    f.subplots_adjust(left=.125, bottom=.225, wspace=.3, right=.975, top=.75)


    ## First ax: the indiv
    ax = axa[0]
    ax.plot(
        bincenters_t,
        agged_all_rewside.loc['rewside', 'mean'],
        color='g')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_all_rewside.loc['rewside', 'lo'],
        y2=agged_all_rewside.loc['rewside', 'hi'],
        color='g',
        alpha=.3, lw=0)
        
    # Plot choice
    ax.plot(
        bincenters_t,
        agged_all_choice.loc['choice', 'mean'],
        color='magenta')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_all_choice.loc['choice', 'lo'],
        y2=agged_all_choice.loc['choice', 'hi'],
        color='magenta',
        alpha=.3, lw=0)
        

    ## Second ax: the joint
    # Plot rewside
    ax = axa[1]
    ax.plot(
        bincenters_t,
        agged_all_both.loc['rewside', 'mean'],
        color='g')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_all_both.loc['rewside', 'lo'],
        y2=agged_all_both.loc['rewside', 'hi'],
        color='g',
        alpha=.3, lw=0)
        
    # Plot choice
    ax.plot(
        bincenters_t,
        agged_all_both.loc['choice', 'mean'],
        color='magenta')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_all_both.loc['choice', 'lo'],
        y2=agged_all_both.loc['choice', 'hi'],
        color='magenta',
        alpha=.3, lw=0)


    ## Third ax: without frontier
    # Plot rewside
    ax = axa[2]
    ax.plot(
        bincenters_t,
        agged_nfc_both.loc['rewside', 'mean'],
        color='g')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_nfc_both.loc['rewside', 'lo'],
        y2=agged_nfc_both.loc['rewside', 'hi'],
        color='g',
        alpha=.3, lw=0)
        
    # Plot choice
    ax.plot(
        bincenters_t,
        agged_nfc_both.loc['choice', 'mean'],
        color='magenta')
    ax.fill_between(
        x=bincenters_t,
        y1=agged_nfc_both.loc['choice', 'lo'],
        y2=agged_nfc_both.loc['choice', 'hi'],
        color='magenta',
        alpha=.3, lw=0)


    ## Pretty
    for ax in axa:
        ax.plot(bins[[0, -1]] / 200., [.5, .5], 'k--', lw=0.8)
        ax.set_ylim((.4, 1))
        ax.set_xlim((-2, 0.475))
        ax.set_xticks((-2, -1, 0))
        my.plot.despine(ax)
        ax.set_xlabel('time in trial (s)')
        
        if ax is axa[0]:
            ax.set_ylabel('decoding accuracy')

    axa[0].set_title('naive\ndecoding', pad=15)
    axa[1].set_title('balanced\ndecoding', pad=15)
    axa[2].set_title('sampling whisks\nremoved', pad=15)

    f.savefig('COMPARISON_PLOT_NEURAL_DECODING_{}.svg'.format(analysis_typ))
    f.savefig('COMPARISON_PLOT_NEURAL_DECODING_{}.png'.format(analysis_typ), dpi=300)


    ## Stats
    stats_filename = 'STATS__COMPARISON_PLOT_NEURAL_DECODING_{}'.format(analysis_typ)
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons\n'.format(n_neurons))
        fi.write('discrimination only; excluding sessions with too few errors\n')
        fi.write('error bars: 95% CIs from bootstrapping\n')

    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

plt.show()