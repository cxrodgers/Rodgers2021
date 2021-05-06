## Plot neural response versus whisking amplitude
"""
5G
    PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM
    STATS__PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM
    Normalized neural response versus whisking amplitude
"""

import json
import pandas
import numpy as np
import my, my.plot, my.decoders
import matplotlib.pyplot as plt
import os


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()

MIN_CYCLES_TO_INCLUDE_NEURON_IN_BIN = 3
MIN_FRAC_NEURONS_INCLUDED_TO_PLOT_BIN = .5


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)

    
## Load trial matrix that went into the spikes
big_tm = pandas.read_pickle(os.path.join(params['neural_dir'], 'neural_big_tm'))


## Load patterns and whisk cycles
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')


## Load features
unbinned_features = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features')


## Load spiking data
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'spikes_by_cycle'))


## Ensure consistent session * trial
# Check that same session * trial are in big_tm and spikes_by_cycle
st1 = spikes_by_cycle.index.to_frame()[
    ['session', 'trial']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial']).reset_index(drop=True)
st2 = big_tm.index.to_frame().reset_index(drop=True).sort_values(
    ['session', 'trial']).reset_index(drop=True)
assert st1.equals(st2)

# Choose these same session * trial in unbinned_features
session_trial_midx = pandas.MultiIndex.from_frame(
    spikes_by_cycle.index.to_frame().reset_index(drop=True)[
    ['session', 'trial']].drop_duplicates())

# Slice these trials
neural_unbinned_features = my.misc.slice_df_by_some_levels(
    unbinned_features, session_trial_midx)
neural_unbinned_features.index = neural_unbinned_features.index.remove_unused_levels()

# Check that they now match exactly
st1 = neural_unbinned_features.index.to_frame()[
    ['session', 'trial']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial']).reset_index(drop=True)
st2 = big_tm.index.to_frame().reset_index(drop=True).sort_values(
    ['session', 'trial']).reset_index(drop=True)
assert st1.equals(st2)


## Ensure consistent session * trial * cycle
# Check that same session * trial * cycle are in neural_unbinned_features and spikes_by_cycle
# Check that they now match exactly
st1 = neural_unbinned_features.index.to_frame()[
    ['session', 'trial', 'cycle']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial', 'cycle']).reset_index(drop=True)
st2 = spikes_by_cycle.index.to_frame()[
    ['session', 'trial', 'cycle']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial', 'cycle']).reset_index(drop=True)
assert st1.equals(st2)


## Slice neurons according to the remaining features
sessions = sorted(
    neural_unbinned_features.index.get_level_values('session').unique())
big_waveform_info_df = big_waveform_info_df.loc[sessions].copy()


## Extract some features to correlate with FR
# Extract from C2_whisk cycles
wampl = C2_whisk_cycles.loc[
    neural_unbinned_features.index, 'amplitude'].rename('wampl')
assert not wampl.isnull().any()

# Extract has_contact
has_contact = (neural_unbinned_features[
    'contact_binarized'] > 0).any(1).rename('has_contact')

# Concat these
some_nuf = pandas.concat([
    wampl, 
    has_contact,
    ], axis=1, verify_integrity=True)


## Bin simply
wampl_bins = np.linspace(0, 200, 26)
some_nuf['wampl_bin'] = pandas.cut(
    some_nuf['wampl'], bins=wampl_bins, labels=False, right=False)
assert not some_nuf['wampl_bin'].isnull().any()


## Join these features with the spike count
merged = pandas.merge(
    spikes_by_cycle.reset_index(), 
    some_nuf.reset_index(), 
    on=['session', 'trial', 'cycle'])

# Normalize spike count
mean_spike_count_by_neuron = spikes_by_cycle.groupby(
    ['session', 'neuron']).mean()
normalized_spike_count = spikes_by_cycle.divide(
    mean_spike_count_by_neuron).rename('norm_spikes').reorder_levels(
    ['session', 'neuron', 'cycle', 'trial']).sort_index()

# Join on merged and check for null
merged = merged.join(
    normalized_spike_count, on=['session', 'neuron', 'cycle', 'trial'])
assert not merged['norm_spikes'].isnull().any()


## Choose metric
metric_to_aggregate = 'wampl_bin'

# Use these names to access bin and standardizing info
metric_name_without_bin = metric_to_aggregate.replace('_bin', '')
raw_metric_name = metric_name_without_bin


## Aggregate
# Aggregate the spikes from each neuron by wampl_bin
resp_vs_wampl_by_neuron = merged.groupby(
    ['session', 'neuron', 'has_contact', metric_to_aggregate])[
    'norm_spikes'].mean()

# Count how many entries go into each aggregate
# session is redundant here
resp_vs_wampl_by_neuron_sz = merged.groupby(
    ['session', 'neuron', 'has_contact', metric_to_aggregate])[
    'norm_spikes'].size()

# Unstack replicates
resp_vs_wampl_by_neuron = resp_vs_wampl_by_neuron.unstack(
    ['session', 'neuron'])
resp_vs_wampl_by_neuron_sz = resp_vs_wampl_by_neuron_sz.unstack(
    ['session', 'neuron']).fillna(0).astype(np.int)


## Deal with missing data
# TODO: consider dropping neurons instead of bins
# Null the aggregate where there's not enough replicates
resp_vs_wampl_by_neuron.values[
    resp_vs_wampl_by_neuron_sz.values < MIN_CYCLES_TO_INCLUDE_NEURON_IN_BIN
    ] = np.nan


## Summarize over neurons
# Mean and SEM
resp_vs_wampl_mean_neuron = resp_vs_wampl_by_neuron.mean(1)
resp_vs_wampl_sem_neuron = resp_vs_wampl_by_neuron.sem(1)

# Null the overall response where >50% of the neurons have nulls
nullfrac = resp_vs_wampl_by_neuron.isnull().mean(1)
resp_vs_wampl_mean_neuron.values[
    nullfrac.values > MIN_FRAC_NEURONS_INCLUDED_TO_PLOT_BIN] = np.nan
resp_vs_wampl_sem_neuron.values[
    nullfrac.values > MIN_FRAC_NEURONS_INCLUDED_TO_PLOT_BIN] = np.nan


## Plot flags
PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM = True


## Plots
if PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM:
    ## Group neurons by NS * stratum (TODO: also task)
    grouping_keys = ['NS', 'stratum']
    NS_l = [False, True]
    stratum_l = ['deep', 'superficial']
    grouped_neurons = big_waveform_info_df.groupby(grouping_keys)


    ## Plot
    #~ f, ax = my.plot.figure_1x1_small()
    #~ f.subplots_adjust(left=.2)

    f, ax = plt.subplots(1, 1, figsize=(2.6, 2.4))
    f.subplots_adjust(bottom=.28, left=.3, right=.95, top=.95)
    
    # Iterate over subpops
    for grouped_keys, sub_bwid in grouped_neurons:
        ## Zip up the keys
        grouped_keys_d = dict(zip(grouping_keys, grouped_keys))
        stratum = grouped_keys_d['stratum']
        NS = grouped_keys_d['NS']
        
        
        ## Set color
        color = 'b' if NS else 'r'
        ls = '--' if stratum == 'superficial' else '-'
        
        
        ## Slice, including only has_contact == False
        sliced = resp_vs_wampl_by_neuron.loc[False, sub_bwid.index]


        ## Summarize over neurons
        # Mean and SEM
        mean_neuron = sliced.mean(1)
        sem_neuron = sliced.sem(1)

        # Null the overall response where >50% of the neurons have nulls
        nullfrac = sliced.isnull().mean(1)
        mean_neuron.values[nullfrac.values > .2] = np.nan
        sem_neuron.values[nullfrac.values > .2] = np.nan
        

        ## Get xticks
        # Get the bincenters corresponding to the data in topl
        #bincenters = metric2bincenters[metric_name_without_bin]
        bincenters = (wampl_bins[1:] + wampl_bins[:-1]) / 2.0
        xticks = bincenters[mean_neuron.index.values]

        
        ## Plot
        ax.plot(xticks, mean_neuron.values, color=color, ls=ls)
        ax.fill_between(
            x=xticks,
            y1=(mean_neuron.values - sem_neuron.values),
            y2=(mean_neuron.values + sem_neuron.values),
            color=color, alpha=.3, lw=0)

    
    ## Pretty
    my.plot.despine(ax)

    # Pretty
    ax.set_ylim((0, 3))
    ax.set_yticks((0, 1, 2, 3))
    ax.set_xlim((0, 50))
    ax.set_xticks((0, 25, 50))

    # Plot line at no effect
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], 'k-', lw=.5, ls='-')
    ax.set_xlim(xlim)
    
    ax.set_xlabel('whisking amplitude ({})'.format(chr(176)))
    ax.set_ylabel('firing rate gain', labelpad=12)

    
    ## Save
    f.savefig('PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM.svg')
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_FR_VS_WHISKING_AMPLITUDE_NOCONTACT_BY_NS_AND_STRATUM'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons total\n'.format(grouped_neurons.size().sum()))
        fi.write('of the following types:\n')
        fi.write(grouped_neurons.size().to_string() + '\n')
        fi.write('combining across tasks\n')
        fi.write('error bars: SEM over included neurons in each bin\n')        
        fi.write('Excluding 1 and 6B completely\n')
        fi.write('Excluding cycles with any contact\n')
        fi.write('Excluding neurons from each bin if < {} cycles\n'.format(
            MIN_CYCLES_TO_INCLUDE_NEURON_IN_BIN))
        fi.write(
            'Only plotting bins with >{}% of neural subpopulation included\n'.format(
            int(np.rint(MIN_FRAC_NEURONS_INCLUDED_TO_PLOT_BIN * 100))))
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


plt.show()