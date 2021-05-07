## Heatmap the contacts and anti-contacts in the warped plot by their evidence

# Uses data:
#   big_weights_part
#   reduced_model_results_sbrc/no_opto/contact_binarized+anti_contact_count+angle+anti_angle_max


"""
4A, bottom; S4A, bottom
    PLOT_EDGE_SUMMARY_ONLY
    Image of the different shape positions in the consensus space

4B
    PLOT_OCCUPANCY_DISCONLY
    Locations of whisks with contact and whisks without contact

4C; S4B
    PLOT_EVIDENCE_DISCONLY_REWSIDEONLY
    Evidence for stimulus in both whisks with and without contact

S4C
    PLOT_EVIDENCE_DISCONLY_CHOICEONLY
    Evidence for choice in both whisks with and without contact
"""

import json
import os
import pandas
import numpy as np
import my.plot 
import matplotlib.pyplot as plt
import matplotlib
import extras


## Plot flags
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))

# Insert mouse and task levels into big_tm
big_tm = my.misc.insert_mouse_and_task_levels(
    big_tm, mouse2task, level=0, sort=True)

# Count the number of trials per session
n_trials_per_session = big_tm.groupby(['task', 'mouse', 'session']).size()

# Count the number of trials per mouse
n_trials_per_mouse = n_trials_per_session.sum(level=['task', 'mouse'])


## Load warping data
transformation_df = pandas.read_pickle(
    os.path.join(params['scaling_dir'], 'transformation_df'))
consensus_edge_summary = pandas.read_pickle(
    os.path.join(params['scaling_dir'], 'consensus_edge_summary'))

# ces to plot
cv_ces = consensus_edge_summary.xs(50, level='stepper_pos').max(level='row')
cc_ces = consensus_edge_summary.xs(150, level='stepper_pos').max(level='row')
all_ces = consensus_edge_summary.max(level='row')

# fillna for transparent plotting
cv_ces[cv_ces == 0] = np.nan
cc_ces[cc_ces == 0] = np.nan
all_ces[all_ces == 0] = np.nan


## Load data
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))
big_cycle_features = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_cycle_features'))

# This is just to plot follicle position
mean_follicle = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'mean_follicle'))

# Transform follicle
transformed_mean_follicle = my.misc.transform(
    mean_follicle, transformation_df).mean(level='whisker')


## Load the original features for plotting
# Need the original bins ('analysis_bin') to interpret the weights
ouf = pandas.read_pickle(os.path.join(params['logreg_dir'], 
    'obliviated_unaggregated_features_with_bin'))

# Insert mouse and task levels into features
ouf = my.misc.insert_mouse_and_task_levels(
    ouf, mouse2task, level=0, sort=True)

# Add a new bin for this analysis
bin_edges_frames = np.linspace(-300, 100, 5)
bin_centers_frames = (bin_edges_frames[1:] + bin_edges_frames[:-1]) / 2.0
bin_ser = pandas.cut(
    C2_whisk_cycles['peak_frame_wrt_rwin'],
    bin_edges_frames, labels=False, right=True).rename('bin')

# Append bin_ser to index
idxdf = ouf.index.to_frame().reset_index(drop=True)
idxdf = idxdf.join(bin_ser, on=['session', 'trial', 'cycle'])
idxdf['bin'] = idxdf['bin'].fillna(-1).astype(np.int)
ouf.index = pandas.MultiIndex.from_frame(idxdf)

# Drop null bins and reorder levels
ouf = ouf.drop(-1, level='bin')
ouf = ouf.reorder_levels(
    ['task', 'mouse', 'session', 'trial', 'bin', 'analysis_bin', 'cycle']
    ).sort_index()

# Extract features of interest
contact_binarized = ouf['contact_binarized']
anti_contact_count = ouf['anti_contact_count']


## Load results of main2a1
big_weights_part = pandas.read_pickle('big_weights_part')

# Choose the reduced_model
reduced_model = 'contact_binarized+anti_contact_count+angle+anti_angle_max'

# Use these weights
use_weights = big_weights_part[False]['no_opto'][reduced_model]

# normalizing stuff for features that aren't raw
normalizing_mu = pandas.read_pickle(os.path.join(
    params['logreg_dir'], 'reduced_model_results_sbrc', 'no_opto', reduced_model, 
    'big_normalizing_mu'))
normalizing_sigma = pandas.read_pickle(os.path.join(
    params['logreg_dir'], 'reduced_model_results_sbrc', 'no_opto', reduced_model, 
    'big_normalizing_sigma'))

# Remove redundant
normalizing_mu = normalizing_mu.xs(
    'rewside', level='decode_label').rename('mu').copy()
normalizing_sigma = normalizing_sigma.xs(
    'rewside', level='decode_label').rename('sigma').copy()


## Extract the locations of each contact, to be weighted by weights
# Extract contact presence and angle onto the columns, one row per contact
stacked_contacts = ouf[
    ['anti_angle_max', 'anti_contact_count', 'contact_binarized', 'angle']
    ].stack('label')

# Drop the rows that have neither anti- nor actual contact
stacked_contacts = stacked_contacts.loc[
    (stacked_contacts['anti_contact_count'] != 0) |
    (stacked_contacts['contact_binarized'] != 0)
    ].copy()

# Join on whisk location (this is where it will be plotted)
# TODO: join on contact location, not peak location, but probably the same
to_join = big_cycle_features[
    ['peak_tip_x', 'peak_tip_y']].stack('whisker')
to_join.index = to_join.index.rename('label', level='whisker')
stacked_contacts = stacked_contacts.join(
    to_join, on=['session', 'trial', 'cycle', 'label']).sort_index()
assert not stacked_contacts.index.duplicated().any()


## Apply the standardization to the non-raw features
# Only standardize these
standardized_features = ['anti_angle_max', 'angle']

# Extract and join on sigma and mu
to_standardize = stacked_contacts[
    standardized_features].stack().rename('value').to_frame()
to_standardize = to_standardize.join(
    normalizing_mu,
    on=['session', 'metric', 'label', 'analysis_bin']
    )
to_standardize = to_standardize.join(
    normalizing_sigma,
    on=['session', 'metric', 'label', 'analysis_bin']
    )
to_standardize['standardized'] = to_standardize['value'].sub(
    to_standardize['mu']).divide(to_standardize['sigma'])

# Drop ones that go to infinity
to_standardize = to_standardize.loc[
    ~np.isinf(to_standardize['standardized']) &
    ~to_standardize['standardized'].isnull() &
    (to_standardize['standardized'].abs() < 10)
    ]

# Put back into stacked_contacts
# This will insert nulls where standardized angle was messed up
to_rejoin = to_standardize['standardized'].unstack('metric')
stacked_contacts = stacked_contacts.drop(standardized_features, axis=1)
stacked_contacts = stacked_contacts.join(to_rejoin)


## Transform contact location into the warped space
to_transform = stacked_contacts[['peak_tip_x', 'peak_tip_y']]
transformed_contacts = my.misc.transform(
    to_transform, transformation_df).rename(
    columns={'peak_tip_x': 'transformed_x', 'peak_tip_y': 'transformed_y'})


## Calculate the evidence of each contact
# Stack contacts again, so that each metric (e.g. angle) is a row
to_weight = stacked_contacts[
    ['anti_contact_count', 'contact_binarized', 'angle', 'anti_angle_max']
    ].stack().rename('value')

# Get decode_label alone on columns
flattened_weights = use_weights.stack().stack().stack().unstack('decode_label')

# Rename weights
flattened_weights = flattened_weights.rename(
    columns={'choice': 'choice_weight', 'rewside': 'rewside_weight'})

# Join the weights onto the contacts
joined = to_weight.to_frame().join(
    flattened_weights, on=flattened_weights.index.names)
#~ assert not joined.isnull().any().any()
assert len(joined) == len(to_weight)

# Shouldn't be any nulls because they would have been dropped by stacking
#~ assert not joined.isnull().any().any()

# Apply weight
joined['choice_evidence'] = joined['value'] * joined['choice_weight']
joined['rewside_evidence'] = joined['value'] * joined['rewside_weight']
evidence = joined[['choice_evidence', 'rewside_evidence']].copy()

# Sum over metric
evidence = evidence.sum(
    level=[lev for lev in evidence.index.names if lev != 'metric']
    )


## Concat data about contacts, their transformed position, and their evidence
contact_evidence = pandas.concat(
    [stacked_contacts, transformed_contacts, evidence], 
    axis=1, sort=True, verify_integrity=True).sort_index(axis=1)


## Bin the contacts spatially
# How to bin
bins_x = np.linspace(-300, 300, 26)
bincenters_x = (bins_x[1:] + bins_x[:-1]) / 2.0
bins_y = np.linspace(-200, 400, 26)
bincenters_y = (bins_y[1:] + bins_y[:-1]) / 2.0

# Histogram the points
contact_evidence['bin_x'] = pandas.cut(
    contact_evidence['transformed_x'],
    bins=bins_x,
    labels=False, right=True)
contact_evidence['bin_y'] = pandas.cut(
    contact_evidence['transformed_y'],
    bins=bins_y,
    labels=False, right=True)

# Drop ones outside bins
# TODO: check this doesn't happen too much
contact_evidence = contact_evidence.dropna(subset=['bin_x', 'bin_y'])
contact_evidence['bin_x'] = contact_evidence['bin_x'].astype(np.int)
contact_evidence['bin_y'] = contact_evidence['bin_y'].astype(np.int)

# This is used to reindex various quantities below to evenly tile the frame
full_spatial_bincenter_midx = pandas.MultiIndex.from_product([
    pandas.Index(range(len(bincenters_x)), name='bin_x'),
    pandas.Index(range(len(bincenters_y)), name='bin_y'),
    ], names=['bin_x', 'bin_y'])


## Rename label to whisker
contact_evidence.index = contact_evidence.index.rename('whisker', level='label')


## Drop C0 for now
contact_evidence = contact_evidence.drop('C0', level='whisker')


## Split the evidence by contact vs no-contact whisks
# A contact occurred
yes_contact_evidence = contact_evidence.loc[
    (contact_evidence['contact_binarized'] > 0) &
    (contact_evidence['anti_contact_count'] == 0)
    ]

# No contact occurred
non_contact_evidence = contact_evidence.loc[
    (contact_evidence['contact_binarized'] == 0) &
    (contact_evidence['anti_contact_count'] > 0)
    ]

# On ~1.5% of whisks some double pump happened where both a contact 
# and an anti-contact happened on the same whisker
# Those are dropped

# Add this as a level
contact_evidence = pandas.concat([
    yes_contact_evidence, non_contact_evidence],
    axis=0, sort=True, verify_integrity=True, keys=['yes', 'non'], 
    names=['contact_typ'])


## Aggregate the evidence by spatial bins
# Mean evidence
gobj = contact_evidence.groupby(
    ['contact_typ', 'task', 'mouse', 'whisker', 'bin_x', 'bin_y'])
aggregated_evidence_spatial = gobj[
    ['choice_evidence', 'rewside_evidence']].mean()

# Count the number of whisks that went into this mean
n_whisks = gobj.size().rename('n_whisks')
assert n_whisks.sum() == len(contact_evidence)
aggregated_evidence_spatial = aggregated_evidence_spatial.join(n_whisks)

# Calculate whisks per trial in each bin
# This is more appropriate for comparing across conditions
aggregated_evidence_spatial['n_whisks_per_trial'] = (
    aggregated_evidence_spatial['n_whisks'].divide(
    n_trials_per_mouse)).reorder_levels(
    aggregated_evidence_spatial.index.names)

# Also normalize this, so that it sums to 1 over all spatial bins
# This is more appropriate for just looking at relative spatial distributions
normalizing_factor = aggregated_evidence_spatial['n_whisks'].sum(
    level=[lev for lev in aggregated_evidence_spatial.index.names 
    if lev not in ['bin_x', 'bin_y']])
aggregated_evidence_spatial['norm_whisks_per_trial'] = (
    aggregated_evidence_spatial['n_whisks'].divide(
    normalizing_factor).reorder_levels(
    aggregated_evidence_spatial.index.names)
    )


## Aggregate the evidence by spatiotemporal bins
## TODO: normalize like above
# Mean evidence
gobj = contact_evidence.groupby(
    ['contact_typ', 'task', 'mouse', 'bin', 'whisker', 'bin_x', 'bin_y'])
aggregated_evidence_spatiotemporal = gobj[
    ['choice_evidence', 'rewside_evidence']].mean()

# Sum occupancy
occupancy = gobj.size().rename('n_contacts')
assert occupancy.sum() == len(contact_evidence)
aggregated_evidence_spatiotemporal = aggregated_evidence_spatiotemporal.join(occupancy)

# Normalize the occupancy to sum to 1 over the spatial bins
contacts_per_bin = aggregated_evidence_spatiotemporal['n_contacts'].sum(
    level=[lev for lev in aggregated_evidence_spatiotemporal.index.names 
    if lev not in ['bin_x', 'bin_y']])
aggregated_evidence_spatiotemporal['occupancy'] = aggregated_evidence_spatiotemporal['n_contacts'].divide(
    contacts_per_bin).reorder_levels(aggregated_evidence_spatiotemporal.index.names)

# Replace bin with bincenter
idxdf = aggregated_evidence_spatiotemporal.index.to_frame().reset_index(drop=True)
idxdf['frame_bin'] = idxdf['bin'].map(
    pandas.Series(bin_centers_frames, index=range(len(bin_centers_frames))))
aggregated_evidence_spatiotemporal.index = pandas.MultiIndex.from_frame(
    idxdf[['contact_typ', 'task', 'mouse', 'frame_bin', 
    'whisker', 'bin_x', 'bin_y']])
aggregated_evidence_spatiotemporal = aggregated_evidence_spatiotemporal.sort_index()


## Plot flags
PLOT_EDGE_SUMMARY_ONLY = True
PLOT_OCCUPANCY_DISCONLY = True
PLOT_EVIDENCE_DISCONLY_REWSIDEONLY = True
PLOT_EVIDENCE_DISCONLY_CHOICEONLY = True


## Plot
if PLOT_EDGE_SUMMARY_ONLY:
    ## Simple single axis with edge summary, for demonstration
    # Figure handle
    f, ax = plt.subplots(figsize=(3, 2.5))
    f.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Plot edge summary
    extras.plot_warped_edge_summary(
        ax, cv_ces=cv_ces, cc_ces=cc_ces, typ='color_by_stimulus')

    # Follicle
    ax.plot(
        [transformed_mean_follicle['x'].values.mean()],
        [transformed_mean_follicle['y'].values.mean()],
        marker='x', color='k', ls='none')

    # Pretty
    ax.axis('image')
    ax.set_xlim((-300, 300))
    ax.set_ylim((300, -200))
    ax.set_xticks([])
    ax.set_yticks([])    

    # Scale bar
    # 2.7mm = 60px, so 45um per px, or 222.2px per 10mm
    ax.plot([-200, -200+111.1], [275, 275], 'k-', lw=.8)
    ax.text(-200 + 55.55, 275, '5 mm', ha='center', va='bottom', size=12)
    
    # Save
    f.savefig('PLOT_EDGE_SUMMARY_ONLY.svg')
    f.savefig('PLOT_EDGE_SUMMARY_ONLY.png', dpi=300)


if PLOT_OCCUPANCY_DISCONLY:
    ## Parameters
    # Metric to plot
    metric_topl = 'norm_whisks_per_trial'

    # Iterate over whisk type (rows of figure)
    whisk_typ_l = ['yes', 'non']
    
    # Do only discrimination
    task = 'discrimination'
    
    # Binning
    mouse_thresh = 4
    nwpt_thresh = .02
    
    # Plotting
    edge_alpha = .3
    occupancy_vmin = 0
    occupancy_vmax = .03
    
    
    ## Aggregrate
    # Slice by task and group by whisk type
    figure_gobj = aggregated_evidence_spatial.xs(
        task, level='task').groupby(
        'contact_typ')

    
    ## Make handles
    f, axa = plt.subplots(
        len(whisk_typ_l), 1,
        figsize=(3, 6.5), sharex=True, sharey=True)
    
    f.subplots_adjust(left=0, right=1, bottom=0, top=.925, hspace=.3)
    
    
    ## Iterate over whisk types (rows)
    for whisk_typ, sub_ae in figure_gobj:
        
        ## Slice
        # Droplevel
        sub_ae = sub_ae.droplevel('contact_typ')

        # Slice data (evidence)
        axis_data = sub_ae[metric_topl]

        # Get ax
        ax = axa[
            whisk_typ_l.index(whisk_typ)
            ]

        # Set title
        if whisk_typ == 'yes':
            ax.set_title('whisks with contact\n(location)')
        
        elif whisk_typ == 'non':
            ax.set_title('whisks without contact\n(location)')
        

        ## Spatialize occupancy
        # Mean over mice, separately by whisker
        spatialized = axis_data.mean(
            level=['whisker', 'bin_x', 'bin_y'])
        

        # Combine to rgb
        occupancy_rgb = extras.combine_whisker_occupancy_to_rgb(
            spatialized, full_spatial_bincenter_midx, 
            bins_x, bins_y,
            x_index=all_ces.columns, y_index=all_ces.index,
            vmin=occupancy_vmin, vmax=occupancy_vmax)
        

        ## Calculate edge_data
        edge_data = all_ces.values

        # Mask the edge_data, so that it has no effect where it is null
        # Actually, this just avoids warnings about null in normalizing
        masked_edge_data = np.ma.masked_array(
            edge_data, np.isnan(edge_data))

        # Normalize edge data to (0, 1) and colormap in black and white
        # This replaces masked data with the colormap's "bad value"
        edge_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        edge_data_rgba = plt.cm.gray_r(edge_norm(masked_edge_data))

    
        ## Blend occupancy_data and edge_data
        blended_rgba = my.plot.alpha_blend_with_mask(
            edge_data_rgba, 
            occupancy_rgb, 
            edge_alpha,
            masked_edge_data.mask,
            )

        
        ## Plot
        im = my.plot.imshow(
            blended_rgba, ax=ax, 
            x=all_ces.columns.values, y=all_ces.index.values)
        

        ## Pretty
        # Plot follicle and ellipses
        extras.plot_follicle_and_ellipses(
            ax, transformed_mean_follicle, label_ellipses=True)
        
        # Limits
        extras.consistent_limits(ax)


    f.savefig('PLOT_OCCUPANCY_DISCONLY.svg')
    f.savefig('PLOT_OCCUPANCY_DISCONLY.png', dpi=300)    


if PLOT_EVIDENCE_DISCONLY_REWSIDEONLY:
    ## Parameters
    # Metric to plot
    metric_topl = 'rewside_evidence'

    # Iterate over whisk type (rows of figure)
    whisk_typ_l = ['yes', 'non']
    
    # Do only discrimination
    task = 'discrimination'
    
    # Binning
    mouse_thresh = 4
    nwpt_thresh = .02
    
    # Plotting
    edge_alpha = .3
    evidence_vmin = -1
    evidence_vmax = 1
    
    
    ## Aggregrate
    # Slice by task and group by whisk type
    figure_gobj = aggregated_evidence_spatial.xs(
        task, level='task').groupby(
        'contact_typ')
    
    
    ## Make handles
    f, axa = plt.subplots(
        len(whisk_typ_l), 1,
        figsize=(4.25, 6.5), sharex=True, sharey=True)
    
    f.subplots_adjust(left=0, right=.7, bottom=0, top=.925, hspace=.3)

    # Axis for colorbar
    cb_ax = f.add_axes((.77, .27, .03, .4))
    cb = f.colorbar(
        matplotlib.cm.ScalarMappable(
        matplotlib.colors.Normalize(vmin=evidence_vmin, vmax=evidence_vmax),
        cmap=plt.cm.RdBu_r), cax=cb_ax)
    cb.set_ticks((evidence_vmin, 0, evidence_vmax))
    cb.ax.tick_params(labelsize=12)
    
    
    ## Iterate over whisk types (rows)
    for whisk_typ, sub_ae in figure_gobj:
        
        ## Slice
        # Droplevel
        sub_ae = sub_ae.droplevel('contact_typ')

        # Slice data (evidence)
        axis_data = sub_ae[metric_topl]

        # Get ax
        ax = axa[
            whisk_typ_l.index(whisk_typ)
            ]

        # Set title
        if whisk_typ == 'yes':
            ax.set_title('whisks with contact\n(evidence)')
        
        elif whisk_typ == 'non':
            ax.set_title('whisks without contact\n(evidence)')
        

        ## Identify spatial bins with enough whisks to be worth plotting
        keep_mask = extras.threshold_bins_by_n_whisks(
            sub_ae, mouse_thresh=mouse_thresh, nwpt_thresh=nwpt_thresh)
        
        
        ## Spatialize evidence
        evidence_data = extras.spatialize_evidence(
            axis_data, keep_mask, full_spatial_bincenter_midx,
            bins_x, bins_y,
            x_index=all_ces.columns, y_index=all_ces.index,
            )

        # Use only raw data
        evidence_data = evidence_data.values


        ## Calculate edge_data
        edge_data = all_ces.values

        # Mask the edge_data, so that it has no effect where it is null
        # Actually, this just avoids warnings about null in normalizing
        masked_edge_data = np.ma.masked_array(
            edge_data, np.isnan(edge_data))


        ## Normalize and blend plot
        extras.normalize_and_blend_plot(
            masked_edge_data, evidence_data, edge_alpha=edge_alpha, ax=ax,
            evidence_vmin=evidence_vmin, evidence_vmax=evidence_vmax,
            x_index=all_ces.columns.values, y_index=all_ces.index.values,
            )


        ## Pretty
        # Plot follicle and ellipses
        extras.plot_follicle_and_ellipses(ax, transformed_mean_follicle)
        
        # Limits
        extras.consistent_limits(ax)

    
    ## Save
    f.savefig('PLOT_EVIDENCE_DISCONLY_REWSIDEONLY.svg')
    f.savefig('PLOT_EVIDENCE_DISCONLY_REWSIDEONLY.png', dpi=300)    


if PLOT_EVIDENCE_DISCONLY_CHOICEONLY:
    ## Parameters
    # Metric to plot
    metric_topl = 'choice_evidence'

    # Iterate over whisk type (rows of figure)
    whisk_typ_l = ['yes', 'non']
    
    # Do only discrimination
    task = 'discrimination'
    
    # Binning
    mouse_thresh = 4
    nwpt_thresh = .02
    
    # Plotting
    edge_alpha = .3
    evidence_vmin = -.5
    evidence_vmax = .5
    
    
    ## Aggregrate
    # Slice by task and group by whisk type
    figure_gobj = aggregated_evidence_spatial.xs(
        task, level='task').groupby(
        'contact_typ')
    
    
    ## Make handles
    f, axa = plt.subplots(
        len(whisk_typ_l), 1,
        figsize=(4.25, 6.5), sharex=True, sharey=True)
    
    f.subplots_adjust(left=0, right=.7, bottom=0, top=.925, hspace=.3)
    
    # Axis for colorbar
    cb_ax = f.add_axes((.77, .27, .03, .4))
    cb = f.colorbar(
        matplotlib.cm.ScalarMappable(
        matplotlib.colors.Normalize(vmin=evidence_vmin, vmax=evidence_vmax),
        cmap=plt.cm.RdBu_r), cax=cb_ax)
    cb.set_ticks((evidence_vmin, 0, evidence_vmax))
    cb.ax.tick_params(labelsize=12)
    
    
    ## Iterate over whisk types (rows)
    for whisk_typ, sub_ae in figure_gobj:
        
        ## Slice
        # Droplevel
        sub_ae = sub_ae.droplevel('contact_typ')

        # Slice data (evidence)
        axis_data = sub_ae[metric_topl]

        # Get ax
        ax = axa[
            whisk_typ_l.index(whisk_typ)
            ]
        
        # Set title
        if whisk_typ == 'yes':
            ax.set_title('whisks with contact\n(evidence)')
        
        elif whisk_typ == 'non':
            ax.set_title('whisks without contact\n(evidence)')


        ## Identify spatial bins with enough whisks to be worth plotting
        keep_mask = extras.threshold_bins_by_n_whisks(
            sub_ae, mouse_thresh=mouse_thresh, nwpt_thresh=nwpt_thresh)
        
        
        ## Spatialize evidence
        evidence_data = extras.spatialize_evidence(
            axis_data, keep_mask, full_spatial_bincenter_midx,
            bins_x, bins_y,
            x_index=all_ces.columns, y_index=all_ces.index,
            )

        # Use only raw data
        evidence_data = evidence_data.values


        ## Calculate edge_data
        edge_data = all_ces.values

        # Mask the edge_data, so that it has no effect where it is null
        # Actually, this just avoids warnings about null in normalizing
        masked_edge_data = np.ma.masked_array(
            edge_data, np.isnan(edge_data))


        ## Normalize and blend plot
        extras.normalize_and_blend_plot(
            masked_edge_data, evidence_data, edge_alpha=edge_alpha, ax=ax,
            evidence_vmin=evidence_vmin, evidence_vmax=evidence_vmax,
            x_index=all_ces.columns.values, y_index=all_ces.index.values,
            )


        ## Pretty
        # Plot follicle and ellipses
        extras.plot_follicle_and_ellipses(ax, transformed_mean_follicle)
        
        # Limits
        extras.consistent_limits(ax)

    
    
    
    ## Save
    f.savefig('PLOT_EVIDENCE_DISCONLY_CHOICEONLY.svg')
    f.savefig('PLOT_EVIDENCE_DISCONLY_CHOICEONLY.png', dpi=300)    

    
plt.show()