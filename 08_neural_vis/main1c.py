## Simple plots of neural that don't depend on GLM
"""
S8G
    PLOT_FR_VS_WIC_AND_BENDING
    N/A
    Neural response versus whisker in contact and bending  
    
"""
# Plot the FR versus whisker in contact AND kappa (or duration)
# Show that the difference between tasks cannot be due just to changes
# in kappa (or duration)


import json
import pandas
import numpy as np
import my, my.plot, my.decoders
import matplotlib.pyplot as plt
import os
import matplotlib_venn


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
    
    
## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)

    
## Load trial matrix that went into the spikes
big_tm = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'neural_big_tm'))


## Load features
glm_features = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'features', 'full', 
    'neural_unbinned_features')).sort_index()


## Load spiking data
# The version in the neural_dir contains a broader temporal range of cycles
spikes_by_cycle = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'aligned_spikes_by_cycle'))

# Ensure consistent session * trial with big_tm
my.misc.assert_index_equal_on_levels(
    big_tm, glm_features, levels=['session', 'trial'])

# Ensure consistent session * trial * cycle with glm_features
my.misc.assert_index_equal_on_levels(
    spikes_by_cycle, glm_features, levels=['session', 'trial', 'cycle'])
    

## Extract some features to correlate with FR
# Of the kappas, I like kappa_std (most linear). 
# But kappa_max accentuates the difference between whiskers the most
# kappa_min is weird because C1 is completely insensitive to this param
# velocity and duration look good
# they don't seem that sensitive to angle (esp C1)
# quite sensitive to anti_ange, probably reflecting whisking signals

# Join these features
feature_names = [
    'kappa_std',# 'kappa_max', 'kappa_min', 
    'velocity2_tip', #'contact_duration',
    #~ 'anti_angle', 'angle', 
    ]

# Extract contact_binarized
features_to_join = glm_features.loc[:, feature_names].copy()

# Unit conversion
features_to_join['kappa_std'] *= 1000 # 1/mm to 1/m
features_to_join['velocity2_tip'] *= 200 # deg/frame to deg/s

# Drop C0 because it's like 10x less common and often zero
features_to_join = features_to_join.drop('C0', axis=1, level='label')
label_l = sorted(features_to_join.columns.get_level_values('label').unique())
    

## Join these features with the spike count
# Squash before merging
features_to_join.columns = ['-'.join(cols) for cols in features_to_join.columns]

#~ # Standardize angle within session
#~ cols_to_standardize = [
    #~ 'angle-C1', 'angle-C2', 'angle-C3', 
    #~ 'anti_angle-C1', 'anti_angle-C2', 'anti_angle-C3', 
    #~ ]
#~ for colname in cols_to_standardize:
    #~ tostand = features_to_join[colname]
    #~ standardized = tostand.sub(tostand.mean(level=0)).div(tostand.std(level=0))
    #~ features_to_join.loc[:, colname] = standardized

# Merge features on spikes
merged = pandas.merge(
    spikes_by_cycle.reset_index(), 
    features_to_join.reset_index(), 
    on=['session', 'trial', 'cycle'])


## Bin the continuous variables
consistent_bins_d = {}
for feature_name in feature_names:
    
    # Squashed feature with label
    squashed_names_l = [
        '{}-{}'.format(feature_name, label) for label in label_l]

    # Consistent bins
    if feature_name == 'contact_duration':
        consistent_bins = np.array([1, 2, 3, 4, 5, 6, 1000])
    elif feature_name == 'kappa_std':
        consistent_bins = np.array([0, 2, 4, 6, 8, 10])
    elif feature_name == 'velocity2_tip':
        consistent_bins = np.array([0, 2, 4, 6, 8, 10]) * 200
    else:
        quantiles = np.linspace(0, 1, 7)
        consistent_bins = merged.loc[
            :, squashed_names_l].stack().dropna().quantile(quantiles).values
    
    # Store those bins
    consistent_bins_d[feature_name] = consistent_bins
    
    
    ## Bin each label (whisker)
    for squashed_name in squashed_names_l:
        # This will be the name of the binned column
        binned_column_name = squashed_name + '-bin'
        
        # bin
        merged[binned_column_name] = pandas.cut(
            merged[squashed_name], bins=consistent_bins, right=False, labels=False)

        # fillna with -1
        merged[binned_column_name] = merged[binned_column_name].fillna(-1)


## Normalize spike count
mean_spike_count_by_neuron = spikes_by_cycle.groupby(
    ['session', 'neuron']).mean()
normalized_spike_count = spikes_by_cycle.divide(
    mean_spike_count_by_neuron).rename('norm_spikes')
merged = merged.join(
    normalized_spike_count, on=normalized_spike_count.index.names)


## Aggregate
agged_l = []
agged_keys_l = []

for feature_name in feature_names:
    
    # Squashed feature with label
    squashed_bin_names_l = [
        '{}-{}-bin'.format(feature_name, label) for label in label_l]

    for feature in squashed_bin_names_l:
    
        # Group by session * neuron * feature
        grouped = merged.groupby(['session', 'neuron', feature])
        
        # Count, mean, err; and unstack values of feature
        grouped_count = grouped.size().unstack(feature).fillna(0).astype(np.int)
        grouped_mean = grouped['norm_spikes'].mean().unstack(feature)
        grouped_err = grouped['norm_spikes'].sem().unstack(feature)
        
        # Aggregate
        agged = pandas.concat(
            [grouped_count, grouped_mean, grouped_err], 
            keys=['count', 'mean', 'err'], axis=1, names=['metric', 'value'],
            ).sort_index(axis=1)
        
        # Store
        agged_l.append(agged)
        agged_keys_l.append(feature)
    
# Concat
resp_by_neuron = pandas.concat(
    agged_l, keys=agged_keys_l, names=['feature'], axis=1).sort_index(axis=1)


## Unsquash feature -> metric * label
# Extract tuples
tup_l = []
for squashed_feature, agg_metric, label_value in resp_by_neuron.columns:
    # Unsquash
    metric, label, thewordbin = squashed_feature.split('-')
    assert thewordbin == 'bin'
    tup_l.append((metric, label + '-bin', agg_metric, label_value))

# MultiIndex
resp_by_neuron.columns = pandas.MultiIndex.from_tuples(
    tup_l, names=['metric', 'label', 'agg_metric', 'label_value'])
resp_by_neuron = resp_by_neuron.sort_index(axis=1)


## Insert mouse and task levels
resp_by_neuron = my.misc.insert_mouse_and_task_levels(
    resp_by_neuron, mouse2task).sort_index()

# Untangle these
count_by_neuron = resp_by_neuron.xs('count', level='agg_metric', axis=1)
mean_by_neuron = resp_by_neuron.xs('mean', level='agg_metric', axis=1)

# Blank out data with too few events
too_few = count_by_neuron < 10
mean_by_neuron.values[too_few.values] = np.nan


## Plot
whisker2color = {'C1': 'b', 'C2': 'g', 'C3': 'r'}
features_topl_l = ['kappa_std']#, 'velocity2_tip']
task_l = ['detection', 'discrimination']
feature2pretty = {
    'kappa_std': 'bending during contact (1/m)', 
    'velocity2_tip': 'pre-contact velocity (deg/s)'
    }


for feature in features_topl_l:
    f, axa = my.plot.figure_1x2_small(sharey=True)
    f.subplots_adjust(left=.175, right=.925, wspace=.4, bottom=.25)
    
    # Get the bins
    consistent_bins = consistent_bins_d[feature]
    consistent_bin_centers = (consistent_bins[1:] + consistent_bins[:-1]) / 2
    
    pretty_feature = feature2pretty[feature]
    
    for task in task_l:
        # Slice
        topl = mean_by_neuron.loc[task, feature]
        topl.columns = topl.columns.remove_unused_levels()
        
        # Get ax
        ax = axa[
            task_l.index(task),
            ]
        ax.set_title(task)
        
        # Iterate over labels
        for label in topl.columns.levels[0]:
            color = whisker2color[label.split('-')[0]]
            sliced = topl.loc[:, label]
            
            # Split out no-contact and contact
            no_contact = sliced.loc[:, -1]
            with_contact = sliced.drop(-1, axis=1)
            
            ax.plot(
                consistent_bin_centers,
                with_contact.mean(),
                color=color, clip_on=False,)
            ax.fill_between(
                consistent_bin_centers,
                with_contact.mean() - with_contact.sem(),
                with_contact.mean() + with_contact.sem(),
                color=color, clip_on=False, alpha=.2)

        # Set xlim
        my.plot.despine(ax)
        ax.set_xlim(consistent_bins[[0, -1]])
        ax.set_ylim((1, 3))
        ax.set_yticks((1, 2, 3))

    axa[0].set_ylabel('normalized\nfiring rate')
    f.text(.55, .05, feature2pretty[feature], ha='center', va='center')
    f.text(.52, .75, 'C1', color='b', ha='center', va='center')
    f.text(.52, .67, 'C2', color='g', ha='center', va='center')
    f.text(.52, .59, 'C3', color='r', ha='center', va='center')
    
    f.savefig('PLOT_FR_VS_WIC_AND_BENDING.svg')
    f.savefig('PLOT_FR_VS_WIC_AND_BENDING.png', dpi=300)

plt.show()