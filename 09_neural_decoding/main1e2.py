## Plot whisker responses, broken by decoding weights
"""
8J
    PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP
    STATS__PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP
    Contact coefficients, broken by decoding preference
"""
import json
import os
import numpy as np
import pandas
import scipy.stats
import matplotlib.pyplot as plt
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Paths
neural_decoding_dir = os.path.join(params['neural_dir'], 'decoding')
    

## Set up plotting
my.plot.poster_defaults()
my.plot.font_embed()


this_WHISKER2COLOR = {'C1': 'b', 'C2': 'g', 'C3': 'r'}
DELTA = chr(916)


## Load the minimal model
model_name = 'minimal+permute_whisks_with_contact'
model_name = 'minimal'


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)

    
## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load results from main4b
glm_results_dir = os.path.join(params['glm_dir'], 'results')
model_results_dir = os.path.join(glm_results_dir, model_name)
coef_wscale_df = pandas.read_pickle(os.path.join(
    model_results_dir, 'coef_wscale_df'))
fitting_results_df = pandas.read_pickle(os.path.join(
    model_results_dir, 'fitting_results_df'))

# Normalize likelihood to the null, and for the amount of data
fitting_results_df['ll_per_whisk'] = (
    (fitting_results_df['likelihood'] - fitting_results_df['null_likelihood']) / 
    fitting_results_df['len_ytest'])

# Convert nats to bits
fitting_results_df['ll_per_whisk'] = (
    fitting_results_df['ll_per_whisk'] / np.log(2))

    
## Include only those results left in big_waveform_info_df 
# (e.g., after dropping 1 and 6b)
coef_wscale_df = my.misc.slice_df_by_some_levels(
    coef_wscale_df, big_waveform_info_df.index)
fitting_results_df = my.misc.slice_df_by_some_levels(
    fitting_results_df, big_waveform_info_df.index)


## Include only discrimination
coef_wscale_df = coef_wscale_df.loc['discrimination']
fitting_results_df = fitting_results_df.loc['discrimination']


## Load decoding results
# Path to the 'all_trials' 'all' 'both' analysis
all_both_path = os.path.join(neural_decoding_dir, 
    '{}-{}-{}'.format('all_trials', 'all', 'both'))

# Load bins (assume same for all)
bins = my.misc.pickle_load(os.path.join(all_both_path, 'bins'))
bincenters = (bins[:-1] + bins[1:]) / 2.0
bincenters_t = bincenters / 200.

# Load meaned weights, just to get the number of neurons
meaned_weights = pandas.read_pickle(
    os.path.join(all_both_path, 'meaned_weights'))


## Process weights
# Mean over boots
meaned_weights = meaned_weights.mean(level=['decode_label', 'bin'])

# Mask only the last 1s of the trial
bin_mask = np.where((
    (bincenters_t > -1) & 
    (bincenters_t < 0)
    ))[0]

# Mean the weights within this mask
mmw = meaned_weights.loc[
    pandas.IndexSlice[:, bin_mask], :].mean(
    level='decode_label', axis=0).T


## Include only encoding coefficients from those neurons used for decoding
coef_wscale_df = my.misc.slice_df_by_some_levels(
    coef_wscale_df, mmw.index)
fitting_results_df = my.misc.slice_df_by_some_levels(
    fitting_results_df, mmw.index)


## Break the neurons into concave- and convex- preferring
# Distribution is kind of lumpy
# .05 seems to capture the central part
# The mode is slightly positive and the tail is slightly heavier on negative
decode_by = 'choice'
mmw['pref'] = 'neither'
mmw.loc[mmw[decode_by] < -.05, 'pref'] = 'concave'
mmw.loc[mmw[decode_by] > .05, 'pref'] = 'convex'


## Slice out only scaled_coef_single from contact_binarized
contact_coefs = coef_wscale_df.xs(
    'contact_binarized', level='metric')['scaled_coef_single']

# Join on pref group
# This also drops any encoding data for which we don't have decoding data
# (e.g., sessions without sufficient errors)
contact_coefs = contact_coefs.to_frame().join(mmw['pref'])
contact_coefs = contact_coefs.set_index('pref', append=True).reorder_levels(
    ['pref', 'session', 'neuron', 'label']).sort_index()

# Unstack label
contact_coefs = contact_coefs['scaled_coef_single'].unstack('label')

# Drop C0
contact_coefs = contact_coefs.drop('C0', axis=1)

# Drop ones missing any coef (this is currently none since C0 is dropped)
assert not contact_coefs.isnull().any().any()

# Evaluate C1vC3
contact_coefs['C1vC3'] = contact_coefs['C1'] - contact_coefs['C3']

# Drop neither for simplicity
contact_coefs = contact_coefs.drop('neither')


## Aggregate over session * neuron
# Mean, SEM, and count
mcc = contact_coefs.mean(level='pref')
ecc = contact_coefs.sem(level='pref')
ncc = contact_coefs.groupby('pref').size()

# Concat and stack
agg = pandas.concat(
    [mcc.stack(), ecc.stack()], axis=1, keys=['mean', 'err']).reorder_levels(
    ['label', 'pref']).sort_index()


## Plots
PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP = True

if PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP:
    ## Plot coefficients for each whisker
    # Plot
    f, axa = plt.subplots(
        1, 2, figsize=(4.5, 2.5), gridspec_kw={'width_ratios': [4.1, 1]}, sharey=True)
    f.subplots_adjust(left=.15, bottom=.25, wspace=.2, right=.97)

    def index2plot_kwargs(ser):
        if ser['pref'] == 'concave':
            color = 'b'
        elif ser['pref'] == 'convex':
            color = 'r'
        else:
            color = 'k'
        
        return {'fc': color, 'ec': 'k'}

    # Plot each whisker
    mtopl = agg.loc[['C1', 'C2', 'C3'], 'mean']
    etopl = agg.loc[['C1', 'C2', 'C3'], 'err']
    my.plot.grouped_bar_plot(
        mtopl,
        yerrlo=mtopl - etopl,
        yerrhi=mtopl + etopl,
        index2plot_kwargs=index2plot_kwargs,
        ax=axa[0], elinewidth=1,
        group_name_fig_ypos=.15,
        group_name_kwargs={'size': 12},
        )

    # Plot C1vC3
    mtopl = agg.loc[['C1vC3'], 'mean']
    etopl = agg.loc[['C1vC3'], 'err']
    my.plot.grouped_bar_plot(
        mtopl,
        yerrlo=mtopl - etopl,
        yerrhi=mtopl + etopl,
        index2plot_kwargs=index2plot_kwargs,
        group_index2group_label=lambda s: '',
        ax=axa[1], elinewidth=1,
        )


    ## Pretty
    for ax in axa:
        my.plot.despine(ax)
        ax.set_xticks([])
        xlim = ax.get_xlim()
        ax.plot(ax.get_xlim(), [0, 0], 'k-', lw=.8)
        ax.set_xlim(xlim)
        
        # Ticks
        fold_ticks = np.array([1, 1.1, 1.2, 1.3])
        ax.set_yticks(np.log(fold_ticks))
        ax.set_yticklabels(fold_ticks)
        ax.set_ylim(np.log([.95, 1.3]))

    my.plot.despine(axa[1], which=('left', 'top', 'right'))
    axa[0].set_ylabel('firing rate gain')
    axa[0].set_xlabel('contacting whisker', labelpad=28)
    axa[0].set_xticklabels(axa[0].get_xticklabels(), size=12)
    axa[1].set_xlabel('C1 - C3', labelpad=9, size=12)

    # Legends
    f.text(.99, .95, 'concave-preferring population', color='b', ha='right', va='center', size=12)
    f.text(.99, .88, 'convex-preferring population', color='r', ha='right', va='center', size=12)

    
    ## Save
    f.savefig('PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP.svg')
    f.savefig('PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_CONTACT_COEF_EACH_WHISKER_BY_DECODING_SUBPOP'
    with open(stats_filename, 'w') as fi:
        ttest_cv_vs_cc = scipy.stats.ttest_ind(
            contact_coefs.loc['concave', 'C1vC3'], 
            contact_coefs.loc['convex', 'C1vC3'],
            ).pvalue
        ttest_cv_vs_zero = scipy.stats.ttest_1samp(
            contact_coefs.loc['convex', 'C1vC3'], 
            popmean=0,
            ).pvalue
        ttest_cc_vs_zero = scipy.stats.ttest_1samp(
            contact_coefs.loc['concave', 'C1vC3'], 
            popmean=0,
            ).pvalue
            
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons\n'.format(len(mmw)))
        fi.write('discrimination only\n')
        fi.write('distribution of preferences:\n')
        fi.write(mmw.groupby('pref').size().to_string() + '\n')
        fi.write('\n')
        fi.write(
            'unpaired t-test on (C1 - C3) for concave vs convex: {}\n'.format(
            ttest_cv_vs_cc))
        fi.write(
            '1-sample t-test on (C1 - C3) for concave vs zero: {}\n'.format(
            ttest_cc_vs_zero))
        fi.write(
            '1-sample t-test on (C1 - C3) for convex vs zero: {}\n'.format(
            ttest_cv_vs_zero))
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))

plt.show()

