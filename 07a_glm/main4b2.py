## A bunch of summary plots for the minimal model

"""
7D, right	
    PLOT_FIT_QUALITY_VS_DEPTH_effect_ll_per_whisk	
    STATS__PLOT_FIT_QUALITY_VS_DEPTH_ll_per_whisk
    Bar plot of fit quality by cell type and depth

7D, left	
    PLOT_FIT_QUALITY_VS_DEPTH_vdepth_ll_per_whisk	
    N/A
    Depth plot of fit quality by cell type

S7C, right	
    PLOT_FIT_QUALITY_VS_DEPTH_effect_score
    STATS__PLOT_FIT_QUALITY_VS_DEPTH_score
    Bar plot of fit quality by cell type and depth

S7C, left	
    PLOT_FIT_QUALITY_VS_DEPTH_vdepth_score
    N/A
    Depth plot of fit quality by cell type
    
S7A
    PLOT_FEATURE_CORRELATION_MAP
    STATS__PLOT_FEATURE_CORRELATION_MAP
    Heatmap of correlation between features in the GLM

7E, top	
    PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER	
    STATS__PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER	
    Fraction of significantly modulated neurons by each feature family in each task

7E, bottom
    VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER	
    STATS__VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER	
    Venn diagram of significantly modulated neurons by each feature family in each task

7F	
    PLOT_TASK_COEFS_OVER_TIME_both_regular
    STATS__PLOT_TASK_COEFS_OVER_TIME_both_regular
    Plot fraction significantly modulated neurons by each task variable over time in trial
    
S7D	
    PLOT_TASK_COEFS_OVER_TIME_optodrop
    STATS__PLOT_TASK_COEFS_OVER_TIME_optodrop
    Plot fraction significantly modulated neurons by each task variable over time in trial, dropping opto sessions
    
8A	
    PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_fracsig	
    STATS__PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM	
    Fraction of significantly modulated neurons by whisking amplitude in each cell type
    
8B	
    PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_scaled_coef_single	
    STATS__PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM	
    Bar plot of whisking amplitude modulation by cell type
    
8C, left	
    PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_vdepth_scaled_coef_single	
    N/A	
    Depth plot of whisking amplitude modulation by cell type

8C, right	
    PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_vdepth_scaled_coef_single_hz	
    N/A	
    Depth plot of whisking amplitude modulation by cell type, in Hz

S8C	
    PLOT_PCA_CONTACT_COEFS	
    STATS__PLOT_PCA_CONTACT_COEFS	
    Plot of PCA components of contact coefficients

8I	
    HEATMAP_CONTACT_COEF_BY_RECLOC_discrimination	
    STATS__HEATMAP_CONTACT_COEF_BY_RECLOC_discrimination	
    Heatmap of contact responses for each neuron during discrimination

S8F	
    BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS	
    STATS__BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS	
    Bar plot of contact coefficients by whisker, task, and stratum

8H	
    BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK	
    STATS__BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK	
    Bar plot of contact coefficients by whisker and task
    
6D	
    PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_fracsig	
    STATS__PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM	
    Fraction of significantly modulated neurons by contact in each cell type
    
8E	
    PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_scaled_coef_single	
    STATS__PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM	
    Bar plot of contact modulation by cell type
    
8F, left	    
    PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_vdepth_scaled_coef_single	
    N/A	
    Depth plot of contact modulation by cell type

8F, right	
    PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM_vdepth_scaled_coef_single_hz	
    N/A	
    Depth plot of contact modulation by cell type, in Hz
    
S8D, E	
    BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM	
    STATS__BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM	
    Bar plot of contact coefficients by whisker, task, recording location, and stratum
    
S8B, left	
    PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_strength	
    STATS__PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY	
    Bar plot of best-whisker contact modulation by cell type
    
S8B, right	
PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_selectivity	
    STATS__PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY	
    Bar plot of best/worst contact response ratio by cell type
"""

import json
import os
import copy
import pandas
import numpy as np
import matplotlib.pyplot as plt
import my
import my.plot
import my.bootstrap
import sklearn.decomposition
import sklearn.preprocessing
import statsmodels.stats.multitest
import matplotlib_venn
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM
import scipy.stats
import extras


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Set up plotting
my.plot.poster_defaults()
my.plot.font_embed()

this_WHISKER2COLOR = {'C1': 'b', 'C2': 'g', 'C3': 'r'}
DELTA = chr(916)


## Tasks to iterate over
model_name = 'minimal'
task_l = [
    'detection', 
    'discrimination',
]


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params, drop_1_and_6b=True)

    
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

# Count the neurons remaining
fit_neurons = coef_wscale_df.index.to_frame()[
    ['task', 'session', 'neuron']].drop_duplicates().reset_index(drop=True)
print("n = {} neurons fit in this model".format(len(fit_neurons)))
print("by task:")
print(fit_neurons.groupby('task').size().to_string() + '\n')


## Load the baseline firing rates
FR_overall = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'FR_overall'))
    

## Load the raw features
neural_unbinned_features = pandas.read_pickle(
    os.path.join(params['glm_dir'], 'features', model_name, 
    'neural_unbinned_features'))


## Plot flags
# Fitting results
PLOT_FIT_QUALITY_VS_DEPTH = True
PLOT_FEATURE_CORRELATION_MAP = True

# Summary plots
PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER = True
VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER = True

# Task responses
PLOT_TASK_COEFS_OVER_TIME = True

# Whisking responses
PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM = True

# Contact responses
PLOT_PCA_CONTACT_COEFS = True
HEATMAP_CONTACT_COEF_BY_RECLOC = True
BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS = True
BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK = True
PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM = True
BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM = True
PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY = True


## Plots
if PLOT_FIT_QUALITY_VS_DEPTH:
    for fitting_metric in ['ll_per_whisk', 'score']:
        ## Plot fit quality versus depth (and by stratum * NS)
        analysis = 'actual_best'

        # Extract scores
        scores = fitting_results_df[fitting_metric].xs(
            analysis, level='analysis').droplevel(
            ['n_reg_lambda', 'n_permute']).mean(
            level=['task', 'session', 'neuron'])


        ## Join on metadata
        data = scores.to_frame().join(
            big_waveform_info_df[['stratum', 'NS', 'layer', 'Z_corrected']], 
            on=['session', 'neuron'])

        # Check for nulls here
        assert not data[
            ['stratum', 'NS', 'layer', 'Z_corrected']].isnull().any().any()
        

        ## Aggregate over neurons
        agg_res = extras.aggregate_coefs_over_neurons(data, fitting_metric)
        
        
        ## Bar plot the effect size
        f, ax = my.plot.figure_1x1_small()
        ax_junk, bar_container = my.plot.grouped_bar_plot(
            df=agg_res['mean'],
            index2plot_kwargs=extras.index2plot_kwargs__NS2color,
            yerrlo=agg_res['lo'],
            yerrhi=agg_res['hi'],
            ax=ax,
            elinewidth=1.5,
            group_name_fig_ypos=.175,
            )

        # Pretty
        my.plot.despine(ax)
        ax.set_ylim((0, .3))
        ax.set_yticks((0, .1, .2, .3))
        ax.set_ylabel('{}goodness-of-fit\n(bits / whisk)'.format(DELTA))

        if fitting_metric == 'll_per_whisk':
            ax.set_ylabel('{}goodness-of-fit\n(bits / whisk)'.format(DELTA))
        
        elif fitting_metric == 'score':
            ax.set_ylabel('{}goodness-of-fit\n(pseudo R2)'.format(DELTA))
        
        # Error bar pretty
        lc = bar_container.lines[2][0]
        lc.set_clip_on(False)
        
        # Save
        f.savefig('PLOT_FIT_QUALITY_VS_DEPTH_effect_{}.svg'.format(fitting_metric))
        f.savefig('PLOT_FIT_QUALITY_VS_DEPTH_effect_{}.png'.format(fitting_metric), dpi=300)
        
        
        ## Plot versus depth
        f, ax = my.plot.figure_1x1_small()
        
        if fitting_metric == 'll_per_whisk':
            my.plot.smooth_and_plot_versus_depth(
                data, fitting_metric, ax=ax, layer_boundaries_ylim=(-.001, 1.1))
            
            # Pretty y
            ax.set_ylim((-.001, 1.1))
            ax.set_ylabel('{}goodness-of-fit\n(bits / whisk)'.format(DELTA))
            ax.set_yticks((0, .5, 1))
        
        elif fitting_metric == 'score':
            my.plot.smooth_and_plot_versus_depth(
                data, fitting_metric, ax=ax, layer_boundaries_ylim=(-.001, .6))
            
            # Pretty y
            ax.set_ylim((-.001, .6))
            ax.set_ylabel('{}goodness-of-fit\n(pseudo R2)'.format(DELTA))
            ax.set_yticks((0, .3, .6))
        
        
        ## Save
        f.savefig(
            'PLOT_FIT_QUALITY_VS_DEPTH_vdepth_{}.svg'.format(fitting_metric))
        f.savefig(
            'PLOT_FIT_QUALITY_VS_DEPTH_vdepth_{}.png'.format(fitting_metric), 
            dpi=300)
        
        
        ## Stats
        stats_filename = 'STATS__PLOT_FIT_QUALITY_VS_DEPTH_{}'.format(fitting_metric)
        with open(stats_filename, 'w') as fi:
            assert analysis == 'actual_best'
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(len(data)))
            fi.write('including both tasks\n')
            fi.write('fitting metric: {}\n'.format(fitting_metric))
            if fitting_metric == 'll_per_whisk':
                fi.write('log-likelihood per whisk in bits, versus a complete null '
                    '(not +drift-population)\n')
            fi.write('using the best reg for each neuron (not single reg)\n')
            fi.write('counts by stratum and NS:\n')
            fi.write(agg_res['agg'].size().to_string() + '\n')
            fi.write('errorbars: 95% CIs from bootstrapping\n')
        
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))


if PLOT_FEATURE_CORRELATION_MAP:
    # Results are almost identical regardless of method but pearson is fastest
    corrdf = neural_unbinned_features.corr(method='pearson')

    # For simplicity drop task indicators
    # TODO: mean these over time, or something
    # They're mainly correlated with each other, not with sensorimotor, anyway
    corrdf = corrdf.drop('task', axis=0).drop('task', axis=1)

    # Drop nuisance
    # Mildly neg corr with set point and pos corr with the contacts and ampl
    #~ corrdf = corrdf.drop(
        #~ 'log_cycle_duration', axis=0).drop(
        #~ 'log_cycle_duration', axis=1)

    # Actually C0 is a nice contrast
    #~ # Drop C0 which isn't particularly correlated
    #~ corrdf = corrdf.drop(
        #~ 'C0', axis=0, level=1).drop(
        #~ 'C0', axis=1, level=1)

    # Mask out the self-similarity
    mask = np.eye(len(corrdf)).astype(np.bool)
    corrdf.values[mask] = 0#np.nan

    # Pretty labels
    pretty_labels = []
    for metric, label in corrdf.index:
        if metric == 'contact_binarized':
            pretty_labels.append('{} contact'.format(label))
        elif metric == 'whisking_indiv_set_point_smoothed':
            pretty_labels.append('{} set point'.format(label))
        elif 'whisking' in metric:
            pretty_labels.append(label)
        elif metric == 'log_cycle_duration':
            pretty_labels.append('cycle duration')
        else:
            pretty_labels.append('unk')
            1/0

    # Plot
    cmap = copy.copy(plt.cm.RdBu_r)
    cmap.set_bad('k') # can't get this to work
    f, ax = plt.subplots(figsize=(4.75, 3.5))
    f.subplots_adjust(left=.2, bottom=.35, top=.975, right=.9)
    im = my.plot.imshow(
        corrdf.fillna(1), ax=ax, axis_call='image', cmap=cmap, 
        )
    
    
    ## Pretty
    ax.set_xticks(range(len(corrdf)))
    ax.set_xticklabels(pretty_labels, rotation=90, size=12)
    ax.set_xlim((-.5, len(corrdf) - .5))
    ax.set_yticks(range(len(corrdf)))
    ax.set_yticklabels(pretty_labels, size=12)
    ax.set_ylim((len(corrdf) - .5, -.5))
    
    # Colorbar
    cb = my.plot.colorbar(ax=ax, shrink=.8)
    cb.mappable.set_clim((-1, 1))
    cb.set_ticks((-1, 0, 1))
    cb.set_label("correlation\nPearson's {}".format(chr(961)))
    
    
    ## Save
    f.savefig('PLOT_FEATURE_CORRELATION_MAP.svg')
    f.savefig('PLOT_FEATURE_CORRELATION_MAP.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_FEATURE_CORRELATION_MAP'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('feature matrix shape: {}\n'.format(neural_unbinned_features.shape))
        fi.write('concatting over all sessions\n')
        fi.write('excluding task indicator variables\n')
        
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

if PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER:
    # Aggregate task coefficients by time, and contact coefficients by whisker
    # This is probably okay, because these coefficients are unlikely to have
    # opposite sign
    # Set point needs to be considered separately because it definitely
    # can have opposite sign over whiskers
    # But that's okay because set point / pose is whisker-specific anyway
    
    
    ## Extract coef_single_p for all, because coef_best_z was optimized without CV
    coef_p = coef_wscale_df['coef_single_p'].copy()
    
    # Drop nuisance
    coef_p = coef_p.drop(['drift', 'log_cycle_duration'], level='metric')
    
    # Drop C0
    # This barely changes anything, though it slightly lowers frac sig
    # for detection set point
    # Let's keep it since it's in the model
    #~ coef_p = coef_p.drop('C0', level='label')
    
    
    ## Split task into separate variables
    # Convert index to DataFrame
    idxdf = coef_p.index.to_frame().reset_index(drop=True)
    
    # Replace each task variable in turn
    sw_flag_l = [
        'current_choice', 'prev_choice', 'previous_outcome', 'current_outcome']
    for sw_flag in sw_flag_l:
        # Mask
        mask = idxdf['label'].str.startswith(sw_flag).fillna(False)
        
        # Replace
        idxdf.loc[mask, 'metric'] = sw_flag
        idxdf.loc[mask, 'label'] = [
            int(label.split('_')[-1]) 
            for label in idxdf.loc[mask, 'label'].values]
    
    # Return index
    coef_p.index = pandas.MultiIndex.from_frame(idxdf)
    coef_p = coef_p.sort_index()
    
    
    ## Check for dropped coefficients
    assert not coef_p.isnull().any()
    # This is probably fine as long as it's not too extreme
    # And as long as an entire family of coefficients isn't missing
    #~ n_neurons_by_metric_label = coef_p.groupby(['metric', 'label']).size()
    #~ n_missing_fit_neurons = len(fit_neurons) - n_neurons_by_metric_label
    #~ if (n_missing_fit_neurons != 0).any():
        #~ print("some neurons missing coefficients:")
        #~ print(n_missing_fit_neurons[n_missing_fit_neurons != 0])
    
    # Count the number of neurons with at least one coefficient per family
    # It's okay if individual labels are missing (e.g., C0 contact) as long
    # as there is at least one label per family
    n_fit_neurons_by_metric = coef_p.index.to_frame()[
        ['session', 'neuron', 'metric']].drop_duplicates().reset_index(
        drop=True).groupby('metric').size()
    assert (n_fit_neurons_by_metric == len(fit_neurons)).all()
    
    
    ## Aggregate significance over time (task variables) or whisker 
    ## (set point and contact variables)
    # Correct by group
    # Bonferroni here to control FWER within the family (metric)
    coef_p2 = coef_p.groupby(['task', 'session', 'neuron', 'metric']).apply(
        lambda ser: my.stats.adjust_pval(ser, 'bonferroni'))
    
    # Take minimum p within each metric (e.g., over whiskers and time)
    coef_minp_by_metric = coef_p2.min(
        level=[lev for lev in coef_p2.index.names if lev != 'label'])
    
    # Threshold
    coef_sig = coef_minp_by_metric < .05
    

    ## Aggregate over neurons
    # Mean over session * neuron within task * metric
    mfracsig = coef_sig.groupby(['metric', 'task']).mean().unstack('task')
    
    # Sort by mean
    mfracsig = mfracsig.loc[
        mfracsig.mean(1).sort_values().index
        ].copy()
    
    
    ## Rename and reorder
    # Rename
    mfracsig = mfracsig.rename(index={
        'contact_binarized': 'contact',
        'current_choice': 'choice',
        'current_outcome': 'outcome',
        'prev_choice': 'previous choice',
        'previous_outcome': 'previous outcome',
        'whisking_global_smoothed': 'whisking amplitude',
        'whisking_indiv_set_point_smoothed': 'whisking set point'
        })

    # Plot position and tick position
    tick2label = pandas.Series({
        0: 'previous choice',
        1: 'choice',
        2: 'previous outcome',
        3: 'outcome',
        4.75: 'contact',
        6.5: 'whisking amplitude',
        7.5: 'whisking set point',
        }).sort_index()
    
    colors = ['r'] * 4 + ['b'] + ['g'] * 2

    # Force sort (note: increasing order of importance)
    mfracsig = mfracsig.loc[tick2label.values]
    assert not mfracsig.isnull().any().any()
    
    # Positions, from smallest (0) to largest
    tick_pos = tick2label.index.values
    
    
    ## Plot
    f, axa = plt.subplots(1, len(task_l), 
        figsize=(5.5, 2.25), sharex=True, sharey=True)
    f.subplots_adjust(left=.275, wspace=.4, bottom=.225, top=.9, right=.97)
    for task in task_l:
        
        ## Slice ax by task
        ax = axa[task_l.index(task)]
        ax.set_title(task)


        ## Plot signif
        bar_container = ax.barh(
            y=tick_pos,
            left=0,
            width=mfracsig[task].values,
            ec='k', fc='lightgray', alpha=.4, lw=0,
            )
        
        for color, patch in zip(colors, bar_container.patches):
            patch.set_facecolor(color)

        # Plot positive again to get the edge alpha correct
        ax.barh(
            y=tick_pos,
            left=0,
            width=mfracsig[task].values,
            ec='k', fc='none', lw=.75,
            )    

        # Plot nonsig
        ns_frac = 1 - mfracsig[task].values
        ax.barh(
            y=tick_pos,
            left=(1 - ns_frac),
            width=ns_frac,
            ec='k', fc='w', lw=.75,
            clip_on=False,
            )    
    
    
    ## Pretty
    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(mfracsig.index.values, size=12)
        ax.set_ylim((-.5, np.max(tick_pos) + 0.5))
        ax.set_xlim((0, 1))
        ax.set_xticks((0, .25, .5, .75, 1))
        ax.set_xticklabels(('0.0', '', '0.5', '', '1.0'))
    
    f.text(.625, .05, 'fraction of recorded neurons that are significant', 
        ha='center', va='center')
    
    
    ## Save
    f.savefig('PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER.svg')
    f.savefig('PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER.png', dpi=300)

    
    ## Stats
    stats_filename = 'STATS__PLOT_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons\n'.format(len(fit_neurons)))
        fi.write('by task:\n{}\n'.format(fit_neurons.groupby('task').size()))
        fi.write('significance taken from coef_single_p, bonferroni'
            ' corrected within metric family, and meaned over neurons\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))


if VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER:
    ## Calculate fracsig within TWC families

    ## Extract coef_single_p for all, because coef_best_z was optimized without CV
    coef_p = coef_wscale_df['coef_single_p'].copy()

    
    ## Convert to TWC family representation
    # Rename metric to task, whisking, or contact
    idx = coef_p.index.to_frame().reset_index(drop=True)
    idx['metric'] = idx['metric'].replace({
        'whisking_indiv_set_point_smoothed': 'whisking',
        'whisking_global_smoothed': 'whisking',
        'contact_binarized': 'contact',
        })
    coef_p.index = pandas.MultiIndex.from_frame(
        idx[['metric', 'label', 'task', 'session', 'neuron']])
    coef_p = coef_p.sort_index()
    assert not coef_p.index.duplicated().any()
    
    # Include only TWC families
    coef_p = coef_p.loc[['task', 'whisking', 'contact']].sort_index()
    assert not coef_p.isnull().any()
    
    # Reorder in standard order
    coef_p = coef_p.reorder_levels(
        ['task', 'session', 'neuron', 'metric', 'label']).sort_index()
    
    
    ## Check for dropped coefficients
    # Count the number of neurons with at least one coefficient per family
    # It's okay if individual labels are missing (e.g., C0 contact) as long
    # as there is at least one label per family
    n_fit_neurons_by_metric = coef_p.index.to_frame()[
        ['session', 'neuron', 'metric']].drop_duplicates().reset_index(
        drop=True).groupby('metric').size()
    assert (n_fit_neurons_by_metric == len(fit_neurons)).all()
    
    
    ## Aggregate significance over time (task variables) or whisker 
    ## (set point and contact variables)
    # Correct by group
    # Bonferroni here to control FWER within the family (metric)
    coef_p2 = coef_p.groupby(['task', 'session', 'neuron', 'metric']).apply(
        lambda ser: my.stats.adjust_pval(ser, 'bonferroni'))
    
    # Take minimum p within each metric (e.g., over whiskers and time)
    coef_minp_by_metric = coef_p2.min(
        level=[lev for lev in coef_p2.index.names if lev != 'label'])
    
    # Threshold
    coef_sig = coef_minp_by_metric < .05

    # Unstack metric to get replicates on rows
    coef_sig = coef_sig.unstack('metric')


    ## Aggregate over neurons
    # Mean over session * neuron within task * metric
    mfracsig = coef_sig.mean(level='task').T
    
    # Sort by mean
    mfracsig = mfracsig.loc[
        mfracsig.mean(1).sort_values().index
        ].copy()
    
    
    ## Venn by task
    task_l = ['detection', 'discrimination']
    
    f, axa = plt.subplots(1, 2, figsize=(4.75, 2))
    f.subplots_adjust(left=.05, right=.9, bottom=.05, top=.8, wspace=.6)
    for task in task_l:
        # Get ax
        ax = axa[task_l.index(task)]
        ax.set_title(task, pad=10)
        
        # Count every combination
        task_coef_sig = coef_sig.loc[task]
        sets = [ 
            set(task_coef_sig.index[task_coef_sig['task'].values]), 
            set(task_coef_sig.index[task_coef_sig['whisking'].values]), 
            set(task_coef_sig.index[task_coef_sig['contact'].values]), 
            ]
        
        venn_res = matplotlib_venn.venn3_unweighted(
            sets, 
            ['task', 'whisking', 'contacts'], 
            ax=ax,
            subset_label_formatter=lambda val: '{}%'.format(int(100 * val / len(task_coef_sig))),
            )

        # Pretty
        for label in venn_res.set_labels + venn_res.subset_labels:
            label.set_fontsize(12)

    
    ## Save
    f.savefig('VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER.svg')
    f.savefig('VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__VENN_FRAC_SIG_COEFS_BY_TASK_AGGREGATING_TIME_AND_WHISKER'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons\n'.format(len(fit_neurons)))
        fi.write('by task:\n{}\n'.format(fit_neurons.groupby('task').size()))
        fi.write('significance taken from coef_single_p, bonferroni'
            ' corrected within TWC family\n')
        fi.write('# of neurons not signif for any family:\n')
        fi.write(
            (coef_sig == False).all(1).astype(np.int).sum(
            level='task').to_string() + '\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))
    

if PLOT_TASK_COEFS_OVER_TIME:
    ## Whether to pool
    # Results are similar across tasks
    include_task = 'both' # 'discrimination'

    ## Plot with and without dropping sessions with any non-sham opto trials
    for drop_opto in [False, True]:
        ## Slice task
        task_coefs = coef_wscale_df.xs('task', level='metric', drop_level=False).copy()

        # Pool or not
        if include_task == 'both':
            task_coefs = task_coefs.droplevel('task')
        else:
            task_coefs = task_coefs.loc[include_task]
        task_coefs.index = task_coefs.index.remove_unused_levels()
        

        ## Drop opto
        if drop_opto:
            include_session = ~(session_df['opto'] & ~session_df['sham'])
            include_session = include_session.index[include_session.values]
            task_coefs = task_coefs.loc[include_session.intersection(task_coefs.index.levels[0])]
            task_coefs.index = task_coefs.index.remove_unused_levels()


        ## Split task into separate variables
        # Convert index to DataFrame
        idxdf = task_coefs.index.to_frame().reset_index(drop=True)
        
        # Replace each task variable in turn
        sw_flag_l = [
            'current_choice', 'prev_choice', 'previous_outcome', 'current_outcome']
        for sw_flag in sw_flag_l:
            # Mask
            mask = idxdf['label'].str.startswith(sw_flag).fillna(False)
            
            # Replace
            idxdf.loc[mask, 'metric'] = sw_flag
            idxdf.loc[mask, 'label'] = [
                int(label.split('_')[-1]) 
                for label in idxdf.loc[mask, 'label'].values]
        
        # Return index
        task_coefs.index = pandas.MultiIndex.from_frame(idxdf)
        task_coefs = task_coefs.sort_index()

        
        ## Futher parameterize
        # Define signif
        # Could correct here. If not, expect 5% positives in each bin
        task_coefs['signif'] = task_coefs['coef_single_p'] < .05
        
        # abs coef
        task_coefs['abs_coef_single_z'] = task_coefs['coef_single_z'].abs()
        task_coefs['abs_scaled_coef_single'] = task_coefs['scaled_coef_single'].abs()
        
        
        ## Aggregate
        coef_metric_to_agg_l = ['signif', 'coef_single_z', 'scaled_coef_single', 
            'abs_coef_single_z', 'abs_scaled_coef_single',]
        
        # By metric * label
        aggmean = task_coefs.groupby(['metric', 'label'])[coef_metric_to_agg_l].mean()
        aggerr = task_coefs.groupby(['metric', 'label'])[coef_metric_to_agg_l].sem()
        
        # Binomial the significant fraction
        signif = task_coefs['signif'].unstack(['session', 'neuron'])
        assert not signif.isnull().any().any()
        signif_err = pandas.DataFrame(
            [my.stats.binom_confint(data=row) 
            for row in signif.astype(np.int).values],
            index=signif.index, columns=['lo', 'hi'],
            )
        signif_err = signif_err.unstack('metric').swaplevel(
            axis=1).sort_index(axis=1)
        
        
        ## Plot
        # left axis: choice; right axis: outcome
        f, axa = plt.subplots(1, 2, figsize=(5.5, 2.25), sharex=True, sharey=True)
        f.subplots_adjust(bottom=.225, left=.15, right=.95, wspace=.2)
        
        for coef_metric in ['signif']: #coef_metric_to_agg_l:
            
            ## Slice
            coef_mean = aggmean[coef_metric].unstack('metric')
            coef_err = aggerr[coef_metric].unstack('metric')
            
            # Account for bin center
            assert (np.diff(coef_mean.index.values) == 100).all()
            assert (np.diff(coef_err.index.values) == 100).all()
            coef_mean.index += 50
            coef_err.index += 50
            
            
            ## Plot
            for metric in coef_mean.columns:
                topl_mean = coef_mean[metric]
                topl_err = coef_err[metric]
                
                # Plot kwargs
                if metric == 'current_choice':
                    ax = axa[1]
                    color = 'k' #'purple'
                    linestyle = '-'
                    zorder = 1
                elif metric == 'prev_choice':
                    ax = axa[1]
                    color = 'k' #'purple'
                    linestyle = '--'            
                    zorder = 1
                elif metric == 'current_outcome':
                    ax = axa[0]
                    color = 'k' #'orange'
                    linestyle = '-'
                    zorder = 0
                elif metric == 'previous_outcome':
                    ax = axa[0]
                    color = 'k' #'orange'
                    linestyle = '--'
                    zorder = 0
                else:
                    1/0
            
                # Plot
                ax.plot(
                    topl_mean.index.values / 200., 
                    topl_mean, 
                    color=color, linestyle=linestyle, zorder=zorder)
                
                if coef_metric == 'signif':
                    topl_err_lo = signif_err.loc[:, (metric, 'lo')]
                    topl_err_hi = signif_err.loc[:, (metric, 'hi')]
                    ax.fill_between(
                        x=topl_mean.index.values / 200., 
                        y1=(topl_err_lo).values,
                        y2=(topl_err_hi).values,
                        color=color, alpha=.2, lw=0)
                
                else:
                    ax.fill_between(
                        x=topl_mean.index.values / 200., 
                        y1=(topl_mean - topl_err).values,
                        y2=(topl_mean + topl_err).values,
                        color=color, alpha=.2, lw=0)


        ## Pretty
        for ax in axa:
            my.plot.despine(ax)
            ax.set_xlim((-2, 0.5))
            ax.set_xticks((-2, -1, 0))
            #~ ax.set_ylim((1, 3))
            #~ ax.set_xlabel('time in trial (s)')
            #~ ax.set_title(coef_metric)
            
            if coef_metric == 'signif':
                ax.set_ylim((0, .55))
                ax.set_yticks((0, .25, .5))
                ax.plot(ax.get_xlim(), [.05, .05], 'k--', lw=.8)

        ## Labels
        axa[1].set_title('choice')
        axa[0].set_title('outcome')
        axa[0].set_ylabel('fraction significant')
        f.text(.55, .04, 'time in trial (s)', ha='center', va='center')

        
        ## Legend
        axa[0].plot([-1.3, -.95], [.1, .1], 'k--')
        axa[0].plot([-1.3, -.95], [.18, .18], 'k-')
        axa[0].text(-.8, .1, 'previous trial', ha='left', va='center', size=12)
        axa[0].text(-.8, .18, 'current trial', ha='left', va='center', size=12)
        
        
        ## Save
        trailing_string = '{}_{}'.format(include_task, 'optodrop' if drop_opto else 'regular')
        f.savefig('PLOT_TASK_COEFS_OVER_TIME_{}.svg'.format(trailing_string))
        f.savefig('PLOT_TASK_COEFS_OVER_TIME_{}.png'.format(trailing_string), dpi=300)


        ## Stats
        stats_filename = 'STATS__PLOT_TASK_COEFS_OVER_TIME_{}'.format(trailing_string)
        with open(stats_filename, 'w') as fi:
            session_neuron_list = task_coefs.index.to_frame()[
                ['session', 'neuron']].drop_duplicates().reset_index(drop=True)
            
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(len(session_neuron_list)))
            fi.write('using coef_single_p < .05 to evaluate signif without correction\n')
            fi.write('error bars: 95% CIs from binom\n')
        
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))
    

if PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM:

    ## Get data
    # Slice whisking_amplitude
    data = coef_wscale_df.xs('amplitude', level='label').copy()

    # Drop useless label level
    data = data.droplevel('metric')

    # Keep only discrimination for now
    #~ data = data.loc['discrimination'].copy()
    
    # But the results basically look the same if pooled
    data = data.droplevel('task')
    
    
    ## Slice coef_metric
    data = data[['scaled_coef_single', 'coef_single', 'coef_single_p']]
    
    # Convert scaled to FR gain / 10 degrees of whisking
    data['scaled_coef_single'] = data['scaled_coef_single'] * 10

    # Convert gain to delta Hz
    data['scaled_coef_single_hz'] = np.exp(
        data['scaled_coef_single']).mul(
        FR_overall.loc[data.index]).sub(
        FR_overall.loc[data.index])        
    
    # Assess sign and significance
    data['positive'] = data['coef_single'] > 0
    data['signif'] = data['coef_single_p'] < .05
    data['pos_sig'] = data['positive'] & data['signif']
    data['neg_sig'] = ~data['positive'] & data['signif']

    
    ## Join on metadata
    data = data.join(
        big_waveform_info_df[['stratum', 'NS', 'layer', 'Z_corrected']], 
        on=['session', 'neuron'])


    ## Aggregate over neurons within stratum * NS
    # coef_metric to aggregate
    coef_metric_l = [
        'scaled_coef_single', 'scaled_coef_single_hz', 'pos_sig', 'neg_sig',
        ]

    # Aggregate
    agg_res = extras.aggregate_coefs_over_neurons(data, coef_metric_l)
    

    ## Bar plot the effect size
    for effect_sz_col in ['scaled_coef_single']:
        # Figure handle
        f, ax = my.plot.figure_1x1_small()
        
        # Grouped bar plot
        my.plot.grouped_bar_plot(
            df=agg_res['mean'][effect_sz_col],
            index2plot_kwargs=extras.index2plot_kwargs__NS2color,
            yerrlo=agg_res['lo'][effect_sz_col],
            yerrhi=agg_res['hi'][effect_sz_col],
            ax=ax,
            group_name_fig_ypos=.175,
            elinewidth=1.5,
            )

        # Pretty
        my.plot.despine(ax)
        ax.set_xticks([])
        
        # Set the ylim in either firing rate gain or spikes
        if effect_sz_col == 'scaled_coef_single':
            # Deal with log-scale on yaxis
            coef_ticklabels = np.array([1, 1.2, 1.4])
            coef_ticks = np.log(coef_ticklabels)
            ax.set_ylim(np.log((1, 1.4)))
            ax.set_yticks(coef_ticks)
            ax.set_yticklabels(coef_ticklabels)
        
            ax.set_ylabel('firing rate gain\n(fold change / 10{})'.format(chr(176)))
            
        elif effect_sz_col == 'scaled_coef_single_hz':
            # Not a log scale
            coef_ticklabels = np.array([0, 2, 4])
            coef_ticks = coef_ticklabels
            ax.set_ylim((-.5, 4.5))
            ax.set_yticks(coef_ticks)
            ax.set_yticklabels(coef_ticklabels)
            
            # Line at zero
            xlim = ax.get_xlim()
            ax.plot(xlim, [0, 0], 'k-', lw=.75)
            ax.set_xlim(xlim)
            
            ax.set_ylabel('evoked firing rate\n(Hz / 10{})'.format(chr(176)))
        
        else:
            ax.set_ylabel(effect_sz_col)
            

        # Save
        f.savefig(
            'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_'
            'BY_NS_AND_STRATUM_{}.svg'.format(effect_sz_col)
            )
        f.savefig(
            'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_'
            'BY_NS_AND_STRATUM_{}.png'.format(effect_sz_col), dpi=300,
            )
    
    
    ## Pie chart the fraction significant
    # Extract pos_sig and neg_sig
    mfracsig = agg_res['mean'][['pos_sig', 'neg_sig']]
    
    # Pie chart
    f, ax = my.plot.figure_1x1_small()
    extras.horizontal_bar_pie_chart_signif(mfracsig, ax=ax)
    
    # Save
    f.savefig(
        'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
        '_fracsig.svg')
    f.savefig(
        'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
        '_fracsig.png', dpi=300)

    
    ## Stats on fracsig
    # Only do stats on this effect_sz_col
    effect_sz_col = 'scaled_coef_single'

    stats_filename = (
        'STATS__PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_'
        'MEAN_BY_NS_AND_STRATUM'
        )
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons total\n'.format(len(data)))
        fi.write('pooling across tasks because similar\n')
        fi.write('error bars: 95% CIs by bootstrapping\n')
        
        for (stratum, NS) in agg_res['mean'].index:
            n_neurons = agg_res['agg'].size().loc[(stratum, NS)]
            frac_pos_sig = agg_res['mean'].loc[(stratum, NS), 'pos_sig']
            frac_neg_sig = agg_res['mean'].loc[(stratum, NS), 'neg_sig']
            effect_mean = agg_res['mean'].loc[(stratum, NS), effect_sz_col]
            
            # Currently this is from bootstrapping
            # Make sure it changes in sync with the above
            effect_errlo = agg_res['lo'].loc[(stratum, NS), effect_sz_col]
            effect_errhi = agg_res['hi'].loc[(stratum, NS), effect_sz_col]
            
            # Dump
            fi.write('{} {}\n'.format(stratum, 'NS' if NS else 'RS'))
            fi.write('effect_sz_col: {}\n'.format(effect_sz_col))
            fi.write('n = {} neurons\n'.format(n_neurons))
            fi.write(
                'effect of whisking: mean {:.3f}, CI {:.3f} - {:.3f}\n'.format(
                effect_mean,
                effect_errlo,
                effect_errhi,
                ))
            fi.write(
                'effect of whisking: exp(mean) {:.3f}, exp(CI) {:.3f} - {:.3f}\n'.format(
                np.exp(effect_mean),
                np.exp(effect_errlo),
                np.exp(effect_errhi),
                ))                
            fi.write('frac pos sig: {} / {} = {:.4f}\n'.format(
                int(n_neurons * frac_pos_sig),
                n_neurons,
                frac_pos_sig,
                ))
            
            fi.write('\n')
    
    # Print
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))


    ## Plot versus depth
    for effect_sz_col in ['scaled_coef_single', 'scaled_coef_single_hz']:
        # Make figure handle
        f, ax = my.plot.figure_1x1_small()
        
        # These will be yaxis
        if effect_sz_col == 'scaled_coef_single':
            ylim = np.log((.4, 2.5))
            coef_ticklabels = np.array([.5, 1, 2])
            coef_ticks = np.log(coef_ticklabels)
            
        else:
            ylim = (-8, 20)
            coef_ticklabels = np.array([0, 10, 20])
            coef_ticks = coef_ticklabels

        # Plot
        my.plot.smooth_and_plot_versus_depth(
            data, effect_sz_col, ax=ax, layer_boundaries_ylim=ylim)
        
        # Set y-axis
        ax.set_yticks(coef_ticks)
        ax.set_yticklabels(coef_ticklabels)
        ax.set_ylim(ylim)
        
        
        if effect_sz_col == 'scaled_coef_single':
            ax.set_ylabel('firing rate gain\n(fold change / 10{})'.format(chr(176)))
        else:
            ax.set_ylabel('{} firing rate\n(Hz / 10{})'.format(DELTA, chr(176)))

        # Plot unity line
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], 'k-', lw=.8)
        ax.set_xlim(xlim)


        ## Save
        f.savefig(
            'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
            '_vdepth_{}.svg'.format(effect_sz_col))
        f.savefig(
            'PLOT_DISCRIMINATION_WHISKING_AMPL_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
            '_vdepth_{}.png'.format(effect_sz_col), dpi=300)


## Conventional way to load contact responses to each whisker
def extract_contact_responses_to_each_whisker(
    coef_wscale_df,
    big_waveform_info_df=None,
    coef_metric='scaled_coef_single',
    include_task=('discrimination',),
    floor_response_at_zero=False,
    drop_always_suppressed=False,
    drop_C0=True,
    drop_neurons_missing_whisker=True,
    combine_off_target=True,
    ):
    """Extract contact responses of each whisker
    
    big_waveform_info_df : DataFrame or None
        Joins on metadata, or does nothing if None
    
    coef_metric : which coef_metric to use
        Using the "scaled" metrics puts the results in a per-contact basis
        Probably better to use _single rather than _best, because otherwise
        different regularization by neuron contaminates the results
        coef_best_z is used for significance
    
    drop_neurons_missing_whisker : bool
        If True, drop neurons missing a coefficient for any of the whiskers
        C1-C3 (or C0-C3 if drop_C0 is False)
    
    combine_off_target : bool
        If True, replace all strings in 'recording_location' that are not
        in the C-row with the string 'off'
    """
    ## Get data
    # Slice contact_binarized
    data = coef_wscale_df.xs('contact_binarized', level='metric').copy()
    
    # Slice coef_metric
    data = data[coef_metric]

    # Floor negative responses to zero
    if floor_response_at_zero:
        data[data < 0] = 0
    
    # Unstack label
    data = data.unstack('label')

    if drop_C0:
        # Drop C0 which is often missing
        data = data.drop('C0', 1)
    
    # Drop any neuron missing C1-C3
    if drop_neurons_missing_whisker:
        data = data.dropna()
    
    # Include task
    # Using list prevents droplevel
    data = data.loc[list(include_task)]

    # Drop neurons that are suppressed by every whisker
    if drop_always_suppressed:
        data = data[(data > 0).any(1)]


    ## Join on metadata
    if big_waveform_info_df is not None:
        data = data.join(
            big_waveform_info_df[['NS', 'layer', 'stratum', 'Z_corrected', 
                'recording_location', 'crow_recording_location']], 
            on=['session', 'neuron'])

        # Replace all off-target locations with the string 'off'
        if combine_off_target:
            off_mask = (
                data['recording_location'] != data['crow_recording_location']
                )
            data.loc[off_mask, 'recording_location'] = 'off'

    return data


if PLOT_PCA_CONTACT_COEFS:
    ## Get data
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df, big_waveform_info_df,
        include_task=('detection', 'discrimination',),
        )

    
    ## PCA params
    # Setting both to False does no standardization
    # With standardization: PC0 is response strength, PC1 is topographic
    # Without standardization: PC0 already shows the C1 preference, 
    #   just not as clean in general
    # Standardizing each neuron to mean zero gets rid of PC0, but otherwise
    #   basically the same.
    standardize_mean = True
    standardize_std = True


    ## Standardize
    whiskers_to_pca = ['C1', 'C2', 'C3']
    # Standardize each feature (whisker), leaving each neuron with its mean
    scaler = sklearn.preprocessing.StandardScaler(
        with_mean=standardize_mean, with_std=standardize_std)
    scaled_arr = scaler.fit_transform(
        data[whiskers_to_pca].values)
    
    # Keep track of the index
    to_pca = pandas.DataFrame(    
        scaled_arr, index=data.index, columns=whiskers_to_pca)
    
    # Do PCA
    pca = sklearn.decomposition.PCA()
    transformed = pca.fit_transform(to_pca)
    transdf = pandas.DataFrame(transformed, index=to_pca.index)

    
    ## Extract components
    components_df = pandas.DataFrame(pca.components_, 
        columns=to_pca.columns).T
    
    # Make this one a consistent sign
    if components_df.loc['C1', 1] < components_df.loc['C3', 1]:
        components_df.loc[:, 1] = -components_df.loc[:, 1].values
        
    
    ## Plot the PCs
    f, ax = my.plot.figure_1x1_small()
    line0, = ax.plot(components_df[0], label='PC1')
    line1, = ax.plot(components_df[1], label='PC2')
    #~ line2, = ax.plot(components_df[2], label='PC3')
    f.text(.85, .9, 'PC1', color=line0.get_color())
    f.text(.85, .82, 'PC2', color=line1.get_color())
    #~ f.text(.85, .74, 'PC3', color=line2.get_color())
    my.plot.despine(ax)
    ax.set_ylim((-1, 1))
    ax.set_yticks((-1, 0, 1))
    ax.plot([0, 2], [0, 0], 'k--', lw=.8)
    ax.set_xlabel('contacting whisker')
    ax.set_ylabel('weight')
    
    
    ## Save
    f.savefig('PLOT_PCA_CONTACT_COEFS.svg')
    f.savefig('PLOT_PCA_CONTACT_COEFS.png', dpi=300)


    ## Stats
    n_by_task = data.reset_index().groupby('task').size()
    
    stats_filename = 'STATS__PLOT_PCA_CONTACT_COEFS'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('# neurons by task:\n' + n_by_task.to_string() + '\n\n')
        fi.write(
            'PCA standardizing both mean and std over features (whiskers), '
            'leaving each neuron with its mean\n')

        fi.write('including only neurons for which we have coefficients for each whisker\n')
        fi.write('pooling over tasks because similar\n')
        fi.write('PC1 explained variance: {:0.2f}\n'.format(pca.explained_variance_ratio_[0]))
        fi.write('PC2 explained variance: {:0.2f}\n'.format(pca.explained_variance_ratio_[1]))
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))
    

if HEATMAP_CONTACT_COEF_BY_RECLOC:
    ## Get data
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df, big_waveform_info_df, 
        include_task=('detection', 'discrimination'))

    
    ## Separately plot each task (figure)
    # The detection one actually looks pretty cool, but just do discrimination
    task_l = ['detection', 'discrimination']
    task_l = ['discrimination']
    for task in task_l:
        task_data = data.loc[task]
    
    
        ## Get numbers of axes and their relative widths
        # Number of neurons in each recloc, strict split
        split_metric = 'recording_location'
        recloc2N = task_data[split_metric].value_counts()
        
        # Arrange the axes this way
        recording_location_l1 = ['off', 'C1', 'C2',]
        recording_location_l2 = ['fill', 'C3', 'fill',]
        
        # Each plot will have 4 axes, the last of which is a colorbar
        # Calculate empty 'fill' axes for the second one
        fill = recloc2N[['off', 'C1', 'C2']].sum() - recloc2N['C3']
        recording_location_N1 = np.array(
            [recloc2N['off'], recloc2N['C1'], recloc2N['C2']])
        recording_location_N2 = np.array(
            [fill / 4, recloc2N['C3'], fill * 3 / 4])
        
        # width_ratios
        width_ratios1 = recording_location_N1 / recording_location_N1.sum()
        width_ratios2 = recording_location_N2 / recording_location_N2.sum()

        ## Create figure handles
        f1, axa1 = plt.subplots(
            1, len(width_ratios1), figsize=(7, 1.1), 
            gridspec_kw={'width_ratios': width_ratios1})
        f1.subplots_adjust(left=.05, right=.95, bottom=.25, top=.7, wspace=.35)

        f2, axa2 = plt.subplots(
            1, len(width_ratios2), figsize=(7, 1.1), 
            gridspec_kw={'width_ratios': width_ratios2})
        f2.subplots_adjust(left=.0, right=.9, bottom=.25, top=.7, wspace=.35)
        axa2[0].set_visible(False)
        axa2[2].set_visible(False)

        # Add the cbar ax
        cbar_ax = f2.add_axes([.875, .27, .008, .4])
        
        ## Iterate over recloc (axis)
        for recloc in recording_location_l1 + recording_location_l2:
            # Get ax unless 'fill'
            if recloc == 'fill':
                continue
            elif recloc in recording_location_l1:
                ax = axa1[recording_location_l1.index(recloc)]
                f = f1
            else:
                ax = axa2[recording_location_l2.index(recloc)]
                f = f2
            recloc_data = task_data[task_data[split_metric] == recloc]
            
            # Title by recloc
            if recloc == 'off':
                ax.set_title('outside C-row')
            else:
                ax.set_title('{} column'.format(recloc))
            
            # Imshow the data sorted by the component
            topl = recloc_data[['C1', 'C2', 'C3']].T
            
            # Sort by C1-C3
            topl = topl.reindex(
                (topl.loc['C1'] - topl.loc['C3']).sort_values().index,
                axis=1)

            # Imshow
            my.plot.imshow(topl.values, ax=ax)
        
            # Lims
            ax.set_xticks((0, topl.shape[1] - 1))
            ax.set_xticklabels((1, topl.shape[1]))
            ax.tick_params(labelsize=12)
            ax.set_xlim((-.5, topl.shape[1] - .5))
            ax.set_ylim((topl.shape[0] - .5, -.5))
            if recloc in ['off', 'C3']:
                ax.set_yticks(range(topl.shape[0]))
                ax.set_yticklabels(topl.index.values, size=12)
            else:
                ax.set_yticklabels([])
                ax.set_yticks([])
        
        
        ## Pretty
        for f, axa in zip([f1, f2], [axa1, axa2]):
            # Set color limits
            my.plot.harmonize_clim_in_subplots(fig=f, clim=[-1, 1])
            
            # colorbar in the last ax
            if f is f2:
                cb = f.colorbar(mappable=f.axes[1].images[0], cax=cbar_ax)
                ticklabels = [.5, 1, 2]
                cb.set_ticks(np.log(ticklabels))
                cb.set_ticklabels(ticklabels)
                cb.ax.tick_params(labelsize=12)

        # Label only the second one
        #~ f2.text(.5, .01, 'neurons sorted by whisker preference', 
            #~ ha='center', va='bottom')
        #~ f1.text(.015, .5, 'contacting\nwhisker', ha='center', va='center', rotation=90)        
        f2.text(.825, .46, 'firing\nrate\ngain', ha='center', va='center')        


        ## Save
        f1.savefig('HEATMAP_CONTACT_COEF_BY_RECLOC_{}_top.svg'.format(task))
        f1.savefig('HEATMAP_CONTACT_COEF_BY_RECLOC_{}_top.png'.format(task), dpi=300)
        f2.savefig('HEATMAP_CONTACT_COEF_BY_RECLOC_{}_bottom.svg'.format(task))
        f2.savefig('HEATMAP_CONTACT_COEF_BY_RECLOC_{}_bottom.png'.format(task), dpi=300)

    
        ## Stats
        stats_filename = 'STATS__HEATMAP_CONTACT_COEF_BY_RECLOC_{}'.format(task)
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(len(data.loc[task])))
            fi.write('{} only\n'.format(task))
            fi.write('excluding L1 and L6b and neurons without C1-C3 coef\n')
            fi.write('using scaled_coef_single to permit comparison\n')
            fi.write('splitting based on recording_location (strict)\n')
            fi.write('# neurons per recloc:{}\n'.format(
                data.loc[task].groupby('recording_location').size().to_string()))

        with open(stats_filename) as fi:
            lines = fi.readlines()
        print(''.join(lines))

if BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS:
    ## Get data
    # Dropping neurons missing any whisker (default) loses all 200CR sessions
    # because of C3
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df,
        big_waveform_info_df,
        include_task=('detection', 'discrimination',),
        )
    
    
    ## Plot
    # axes are stratum
    # Each axis is vs wic and NS
    NS_l = [False, True]
    whisker_names = ['C1', 'C2', 'C3']
    stratum_l = ['superficial', 'deep']

    def index2plot_kwargs(ser):
        color = 'b' if ser['NS'] else 'r'
        return {'fc': color}
    
    def index2label(ser):
        return ser['wic']
    
    def group_index2group_label(NS):
        if NS:
            return 'inhib'
        else:
            return 'excit'


    ## Figure handles
    task_l = ['detection', 'discrimination']
    f, axa = plt.subplots(
        len(task_l),
        len(stratum_l), 
        figsize=(5, 4.3),
        sharey='row', sharex=True,
    )
    f.subplots_adjust(wspace=.4, hspace=.6)
    
    ## Iterate over task
    for task in task_l:
        # Group by stratum
        gobj = data.loc[task].groupby('stratum')
        
        # Iterate over stratum (axis)
        for stratum, this_data in gobj:
            ## Slice
            try:
                ax = axa[
                    stratum_l.index(stratum),
                    task_l.index(task),
                ]
            except ValueError:
                continue
            
            ax.set_title(stratum)
            
            
            ## Aggregate
            agg = this_data.groupby('NS')
            meaned = agg[whisker_names].mean().stack()
            agg_sem = agg[whisker_names].sem().stack()
            meaned.index.names = ['NS', 'wic']
            
            
            ## Plot
            my.plot.grouped_bar_plot(
                meaned, 
                yerrlo=(meaned - agg_sem), #agg_err['lo'], #(meaned - agg_sem),
                yerrhi=(meaned + agg_sem), #agg_err['hi'], #(meaned + agg_sem),
                ax=ax,
                index2label=index2label,
                index2plot_kwargs=index2plot_kwargs,
                group_index2group_label=lambda s: None,
                group_name_fig_ypos=-.1,
                xtls_kwargs={'size': 12},
                )
        
        
    ## Pretty
    for ax in axa.flatten():
        if ax in axa[0]:
            # Superficial
            yticklabels = np.array([1, 2])
            yticks = np.log(yticklabels)
            ylim = np.log([1, 2])
        else:
            # Deep
            yticklabels = np.array([1, 1.3])
            yticks = np.log(yticklabels)
            ylim = np.log([1, 1.3])
        
        my.plot.despine(ax)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(ylim)

    # Label columns by task
    f.text(.289, .98, task_l[0], ha='center', va='center')
    f.text(.74, .98, task_l[1], ha='center', va='center')
    f.text(.025, .5, 'firing rate gain', ha='center', va='center', rotation=90)


    ## Save
    f.savefig(
        'BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS.svg')
    f.savefig(
        'BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS.png', dpi=300)


    ## Stats
    n_by_cell_type = data.reset_index().groupby(
        ['task', 'stratum', 'NS']).size().unstack('task').T
    
    stats_filename = 'STATS__BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK_STRATUM_AND_NS'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons ({} det., {} disc.)\n'.format(
            len(data),
            len(data.loc['detection']),
            len(data.loc['discrimination']),
            ))
        fi.write('finer-grained N:\n' + n_by_cell_type.to_string() + '\n')
        fi.write('error bars: SEM\n')
        fi.write('including only neurons for which we have contacts by each whisker\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))


if BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK:
    ## Get data
    # Dropping neurons missing any whisker (default) loses 181212_200CR and 
    # 181213_200CR because of C3
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df,
        big_waveform_info_df,
        include_task=('detection', 'discrimination',),
        )
    
    
    ## Plot
    # axes are stratum
    # Each axis is vs wic and NS
    NS_l = [False, True]
    whisker_names = ['C1', 'C2', 'C3']
    stratum_l = ['superficial', 'deep']

    def index2plot_kwargs(ser):
        return {'fc': 'w', 'ec': 'k'}
    
    def index2label(ser):
        return ser['wic']
    
    def group_index2group_label(NS):
        if NS:
            return 'inhib'
        else:
            return 'excit'


    ## Plot handles
    task_l = ['detection', 'discrimination']

    # Figure handles
    #~ f, axa = my.plot.figure_1x2_small(sharey=True, sharex=True)
    f, axa = plt.subplots(1, 2, figsize=(5, 3), sharex=True, sharey=True)
    f.subplots_adjust(bottom=.2, left=.18, right=.95, wspace=.4, top=.825)


    ## Aggregate
    groupby_l = ['task']
    agg = data.groupby(groupby_l)
    
    # Mean and SEM
    agg_mean = agg[whisker_names].mean()
    agg_err = agg[whisker_names].sem()
    agg_mean.columns.name = 'wic'
    agg_err.columns.name = 'wic'


    ## Stats
    REPEATED_MEASURES = False
    
    # Extract variables to consider
    data_table = data.reset_index()[
        ['task', 'session', 'neuron', 'C1', 'C2', 'C3', 'NS', 'stratum', 
        'crow_recording_location']
        ]
    
    # Shorten this name
    data_table = data_table.rename(columns={'crow_recording_location': 'recloc'})
    
    # Construct subject id
    data_table['subject'] = [
        '{}-{}'.format(session, neuron) for session, neuron in 
        zip(data_table['session'], data_table['neuron'])]
    data_table = data_table.drop(['session', 'neuron'], 1)
    
    if REPEATED_MEASURES:
        # Ideally, treat response to each whisker as a repeated measure
        # But this requires between-subjects factors not yet implemented
        # Stack the whisker
        data_table = data_table.set_index(
            [col for col in data_table.columns if col not in ['C1', 'C2', 'C3']])
        data_table.columns.name = 'whisker'
        data_table = data_table.stack().rename('response').reset_index()
    
    else:
        # Simplify as C3 - C1
        data_table['response'] = data_table['C1'] - data_table['C3']
        
    # Define formula
    formula = (
        "response ~ task + stratum + NS + "
        "C(recloc, levels=['off','C1','C2','C3'])"
        )
    
    # Build linear model
    lm = ols(formula, data_table).fit()
    
    # Run ANOVA
    # This reveals that task, stratum, and recloc are the most important factors
    # NS only slightly matters
    # discrimination, superficial, and C1 recloc are most associated with C1>C3 effect
    aov = anova_lm(lm)
    

    ## Simple AnovaRM on whisker, separately by task only
    # Extract variables to consider
    data_table = data.reset_index()[
        ['task', 'session', 'neuron', 'C1', 'C2', 'C3']
        ]
    
    # Construct subject id
    data_table['subject'] = [
        '{}-{}'.format(session, neuron) for session, neuron in 
        zip(data_table['session'], data_table['neuron'])]
    data_table = data_table.drop(['session', 'neuron'], 1)
    
    # Separate AnovaRM on each task, leaving only whisker
    data_table = data_table.set_index(['task', 'subject'])
    data_table.columns.name = 'whisker'
    data_table = data_table.stack().rename('response')
    
    # Separately for each task
    task_posthoc_l = []
    task_aov_l = []
    for task in task_l:
        # Slice data for this task
        task_data = data_table.loc[task].reset_index()
        
        # Run AnovaRM on task_data
        aov = AnovaRM(
            task_data, depvar='response', subject='subject', 
            within=['whisker'])
        res = aov.fit()
        
        # Extract pvalue
        aov_pvalue = res.anova_table.loc['whisker', 'Pr > F']
        task_aov_l.append((task, aov_pvalue))
    
        # Post-hoc
        for w0 in ['C1', 'C2', 'C3']:
            for w1 in ['C1', 'C2', 'C3']:
                # Avoid double-testing or self-testing
                if w0 >= w1:
                    continue
                
                # Extract data from each whisker
                data0 = task_data.loc[task_data['whisker'] == w0, 'response'].values
                data1 = task_data.loc[task_data['whisker'] == w1, 'response'].values
                
                # Paired
                pvalue = scipy.stats.ttest_rel(data0, data1)[1]
                
                # Store
                task_posthoc_l.append((task, w0, w1, pvalue))
        
    # Concat posthoc
    # Currently this reveals a barely nonsig anova for detection, and
    # a highly sig anova for discrimination. For discrimination, all
    # whiskers highly sig. For detection, C2 slightly differs from the others.
    # The detection results are so nearing signif that I'm not sure whether
    # to rely on them.
    task_aov_pvalue = pandas.DataFrame.from_records(
        task_aov_l, columns=['task', 'aov_pvalue']).set_index(
        'task')['aov_pvalue']
    task_posthoc_df = pandas.DataFrame.from_records(
        task_posthoc_l, columns=['task', 'w0', 'w1', 'pvalue']).set_index(
        ['task', 'w0', 'w1'])['pvalue'].sort_index()
    
    
    ## Iterate over task
    for task in agg_mean.index:
        
        ## Slice by task
        topl_mean = agg_mean.loc[task]
        topl_err = agg_err.loc[task]
        
        
        ## Get ax
        ax = axa[
            task_l.index(task),
        ]             
        ax.set_title('{}'.format(task), pad=15)
        
        
        ## Plot
        my.plot.grouped_bar_plot(
            topl_mean, 
            yerrlo=(topl_mean - topl_err),
            yerrhi=(topl_mean + topl_err),
            ax=ax,
            index2label=index2label,
            index2plot_kwargs=index2plot_kwargs,
            group_index2group_label=group_index2group_label,
            group_name_fig_ypos=-.1,
            xtls_kwargs={'size': 12},
            )


        ## Stats
        stats_yval = np.log(1.28)
        ax.plot([0, 2], [stats_yval] * 2, 'k-', lw=.75)
        task_pval = task_aov_pvalue.loc[task]
        task_sig_str = my.stats.pvalue_to_significance_string(task_pval)
        ax.text(1, stats_yval, task_sig_str, ha='center', va='bottom', size=12)
    
    
    ## Pretty
    yticklabels = np.array([1, 1.1, 1.2, 1.3])
    yticks = np.log(yticklabels)
    ylim = np.log([1, 1.3])
    
    for ax in axa.flatten():
        my.plot.despine(ax)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(ylim)
    
    
    ## Pretty
    axa[0].set_ylabel('firing rate gain\n(fold / contact)')
    f.text(.55, .05, 'contacting whisker', ha='center', va='center')


    ## Save
    f.savefig('BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK.svg')
    f.savefig('BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK.png', dpi=300)


    ## Stats
    stats_filename = 'STATS__BAR_PLOT_CONTACT_COEFS_EACH_WHISKER_BY_TASK'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons ({} det., {} disc.)\n'.format(
            len(data),
            len(data.loc['detection']),
            len(data.loc['discrimination']),
            ))
        fi.write('excluding 200CR\n')
        fi.write('error bars: SEM\n')
        fi.write('anova by task:\n{}\n'.format(task_aov_pvalue.to_string()))
        fi.write('posthoc paired t-test by task:\n{}\n'.format(
            task_posthoc_df.to_string()))
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM:
    ## Get data
    # Slice contact_binarized
    data = coef_wscale_df.xs('contact_binarized', level='metric').copy()

    # Keep only discrimination for now
    #~ data = data.loc['discrimination'].copy()
    
    # Results pretty similar if pooled
    data = data.droplevel('task')


    ## Slice coef_metric
    # Using the "scaled" metrics puts the results in a per-contact basis
    data = data[['scaled_coef_single', 'coef_single_p']]
    
    
    ## Aggregate over whiskers
    # Unstack whisker
    data = data.unstack('label')
    
    # Drop C0, though since we mean over whiskers, this might be unnecessary
    data = data.drop('C0', axis=1, level='label')
    data.columns = data.columns.remove_unused_levels()
    
    # Drop neurons missing whiskers
    drop_mask = data.isnull().any(1)
    if drop_mask.sum() > 0:
        print("warning: dropping {} neurons\n".format(drop_mask.sum()))
        data = data.loc[~drop_mask]
        data.index = data.index.remove_unused_levels()

    # Mean response over whiskers
    # Could consider maxing over whiskers here, but that only slight
    # raises responses and produces some annoying outliers
    meaned_responses = data['scaled_coef_single'].mean(axis=1)
    
    # Adjust p-value over whiskers
    # Use Bonferroni to control FWER
    adjusted_pvalues_l = []
    for row in data['coef_single_p'].values:
        adjusted_pvalues_l.append(
            statsmodels.stats.multitest.multipletests(
            row, method='bonferroni')[1])
    adjusted_pvalues = pandas.DataFrame(
        adjusted_pvalues_l,
        index=data['coef_single_p'].index,
        columns=data['coef_single_p'].columns,
        )
    
    # Min p-value over whiskers
    adjusted_pvalues = adjusted_pvalues.min(1)
    
    # Concat response and pvalue
    data = pandas.concat([
        meaned_responses,
        adjusted_pvalues,
        ], keys=['scaled_coef_single', 'coef_single_p'], axis=1, 
        names=['coef_metric'])


    ## Scaling
    # Convert gain to delta Hz
    data['scaled_coef_single_hz'] = np.exp(
        data['scaled_coef_single']).mul(
        FR_overall.loc[data.index]).sub(
        FR_overall.loc[data.index])        
    
    # Assess sign and significance
    data['positive'] = data['scaled_coef_single'] > 0
    data['signif'] = data['coef_single_p'] < .05
    data['pos_sig'] = data['positive'] & data['signif']
    data['neg_sig'] = ~data['positive'] & data['signif']


    ## Join on metadata
    data = data.join(
        big_waveform_info_df[['stratum', 'NS', 'layer', 'Z_corrected']], 
        on=['session', 'neuron'])
    
    
    ## Aggregate over neurons within stratum * NS
    # coef_metric to aggregate
    coef_metric_l = [
        'scaled_coef_single', 'scaled_coef_single_hz', 'pos_sig', 'neg_sig',
        ]

    # Aggregate
    agg_res = extras.aggregate_coefs_over_neurons(data, coef_metric_l)


    ## Bar plot the effect size
    for effect_sz_col in ['scaled_coef_single']:
        # Figure handle
        f, ax = my.plot.figure_1x1_small()

        # Grouped bar plot
        my.plot.grouped_bar_plot(
            df=agg_res['mean'][effect_sz_col],
            index2plot_kwargs=extras.index2plot_kwargs__NS2color,
            yerrlo=agg_res['lo'][effect_sz_col],
            yerrhi=agg_res['hi'][effect_sz_col],
            ax=ax,
            group_name_fig_ypos=.175,
            elinewidth=1.5,
            )

        # Pretty
        my.plot.despine(ax)
        ax.set_xticks([])
        
        # Set the ylim in either firing rate gain or spikes
        if effect_sz_col == 'scaled_coef_single':
            # Deal with log-scale on yaxis
            coef_ticklabels = np.array([1, 1.2, 1.4])
            coef_ticks = np.log(coef_ticklabels)
            ax.set_ylim(np.log((1, 1.5)))
            ax.set_yticks(coef_ticks)
            ax.set_yticklabels(coef_ticklabels)
        
            ax.set_ylabel('firing rate gain\n(fold change / contact)')
            
        elif effect_sz_col == 'scaled_coef_single_hz':
            # Not a log scale
            coef_ticklabels = np.array([0, 3, 6])
            coef_ticks = coef_ticklabels
            ax.set_ylim([0, 6])
            ax.set_yticks(coef_ticks)
            ax.set_yticklabels(coef_ticklabels)
            
            ax.set_ylabel('evoked firing rate\n(Hz / contact)')
        
        else:
            ax.set_ylabel(effect_sz_col)

        # Save
        f.savefig(
            'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_'
            'BY_NS_AND_STRATUM_{}.svg'.format(effect_sz_col)
            )
        f.savefig(
            'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_'
            'BY_NS_AND_STRATUM_{}.png'.format(effect_sz_col), dpi=300,
            )
    

    ## Pie chart the fraction significant
    # Extract pos_sig and neg_sig
    mfracsig = agg_res['mean'][['pos_sig', 'neg_sig']]
    
    # Pie chart
    f, ax = my.plot.figure_1x1_small()
    extras.horizontal_bar_pie_chart_signif(mfracsig, ax=ax)

    # Save
    f.savefig(
        'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
        '_fracsig.svg')
    f.savefig(
        'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
        '_fracsig.png', dpi=300)


    ## Stats on fracsig
    # Only do stats on this effect_sz_col
    effect_sz_col = 'scaled_coef_single'
    
    stats_filename = (
        'STATS__PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_'
        'MEAN_BY_NS_AND_STRATUM'
        )
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons total\n'.format(len(data)))
        fi.write('pooling across tasks because similar\n')
        fi.write('dropped C0; meaned response over whiskers; '
            'bonferroni-adjusted p-values and min over whiskers\n')
        fi.write('error bars: 95% CIs from bootstrapping\n')
        
        for (stratum, NS) in agg_res['mean'].index:
            n_neurons = agg_res['agg'].size().loc[(stratum, NS)]
            frac_pos_sig = agg_res['mean'].loc[(stratum, NS), 'pos_sig']
            frac_neg_sig = agg_res['mean'].loc[(stratum, NS), 'neg_sig']
            effect_mean = agg_res['mean'].loc[(stratum, NS), effect_sz_col]
            
            # Currently this is from bootstrapping
            # Make sure it changes in sync with the above
            effect_errlo = agg_res['lo'].loc[(stratum, NS), effect_sz_col]
            effect_errhi = agg_res['hi'].loc[(stratum, NS), effect_sz_col]
            
            # Dump
            fi.write('{} {}\n'.format(stratum, 'NS' if NS else 'RS'))
            fi.write('effect_sz_col: {}\n'.format(effect_sz_col))
            fi.write('n = {} neurons\n'.format(n_neurons))
            fi.write(
                'effect of contacts: mean {:.3f}, CI {:.3f} - {:.3f}\n'.format(
                effect_mean,
                effect_errlo,
                effect_errhi,
                ))
            fi.write(
                'effect of contacts: exp(mean) {:.3f}, exp(CI) {:.3f} - {:.3f}\n'.format(
                np.exp(effect_mean),
                np.exp(effect_errlo),
                np.exp(effect_errhi),
                ))                
            fi.write('frac pos sig: {} / {} = {:.4f}\n'.format(
                int(n_neurons * frac_pos_sig),
                n_neurons,
                frac_pos_sig,
                ))
            
            fi.write('\n')
    
    # Print
    with open(stats_filename) as fi:
        print(''.join(fi.readlines()))
    

    ## Plot versus depth
    for effect_sz_col in ['scaled_coef_single', 'scaled_coef_single_hz']:
        # Make figure handle
        f, ax = my.plot.figure_1x1_small()
        
        # These will be yaxis
        if effect_sz_col == 'scaled_coef_single':
            ylim = np.log((.5, 3.5))
            coef_ticklabels = np.array([.5, 1, 2])
            coef_ticks = np.log(coef_ticklabels)
            
        else:
            ylim = (-5, 25)
            coef_ticklabels = np.array([0, 10, 20])
            coef_ticks = coef_ticklabels

        # Plot
        my.plot.smooth_and_plot_versus_depth(
            data, effect_sz_col, ax=ax, layer_boundaries_ylim=ylim)
        
        # Set y-axis
        ax.set_yticks(coef_ticks)
        ax.set_yticklabels(coef_ticklabels)
        ax.set_ylim(ylim)
        
        
        if effect_sz_col == 'scaled_coef_single':
            ax.set_ylabel('firing rate gain\n(fold / contact)'.format(chr(176)))
        else:
            ax.set_ylabel('{} firing rate\n(Hz / contact)'.format(DELTA, chr(176)))

        # Plot unity line
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], 'k-', lw=.8)
        ax.set_xlim(xlim)


        ## Save
        f.savefig(
            'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
            '_vdepth_{}.svg'.format(effect_sz_col))
        f.savefig(
            'PLOT_DISCRIMINATION_CONTACTS_FRACSIG_AND_MEAN_BY_NS_AND_STRATUM'
            '_vdepth_{}.png'.format(effect_sz_col), dpi=300)


if BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM:
    # Bar plot of contact response to each whisker, by task * stratum * recloc
    # Combine over NS since similar
    # Will show that detection has decent topography, but discrimination
    # has strong C1 bias

    ## Get data
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df, big_waveform_info_df,
        include_task=('detection', 'discrimination',),
        )


    ## Group 
    whisker_l = ['C1', 'C2', 'C3']
    grouping_keys = ['task', 'recording_location', 'stratum']
    agg = data.groupby(grouping_keys)[whisker_l]
    
    # Mean and sem
    agg_mean = agg.mean()
    agg_err = agg.sem()
    agg_mean.columns.name = 'whisker'
    agg_err.columns.name = 'whisker'
    
    # Stack whisker
    agg_mean = agg_mean.stack()
    agg_err = agg_err.stack()


    ## Plot
    # Rows: stratum. Cols: task.
    # Axes: recording location * whisker in contact
    #f, ax = my.plot.figure_1x1_standard()
    stratum_l = ['superficial', 'deep']
    task_l = ['detection', 'discrimination']
    f, axa = plt.subplots(
        len(stratum_l),
        len(task_l),
        figsize=(6.5, 4.75),
        sharex=True, sharey=True,
        )
    f.subplots_adjust(left=.125, right=.95, bottom=.15, top=.94, hspace=.8, wspace=.2)
    
    def index2plot_kwargs(idx):
        if idx['whisker'] == 'C1':
            fc = 'b'
        elif idx['whisker'] == 'C2':
            fc = 'g'
        elif idx['whisker'] == 'C3':
            fc = 'r'
        else:
            fc = 'white'
        
        if idx['recording_location'] == idx['whisker']:
            alpha = 1
        else:
            alpha = .25

        ec = 'k'
        return {'fc': fc, 'ec': ec, 'alpha': alpha}
    
    def group_index2group_label(recloc):
        return {'C1': 'C1\ncolumn', 'C2': 'C2\ncolumn', 'C3': 'C3\ncolumn', 
            'off': 'off-\ntarget'}[recloc]
    
    
    ## Iterate over task and stratum (axes)
    for task in task_l:
        for stratum in stratum_l:
            # Get ax
            ax = axa[
                stratum_l.index(stratum),
                task_l.index(task),
            ]
            
            # Title ax
            ax.set_title('{} ({} layers)'.format(task, stratum))
            
            # Slice data
            topl = agg_mean.loc[task].xs(stratum, level='stratum')
            topl_err = agg_err.loc[task].xs(stratum, level='stratum')
    
            # Plot handles
            if ax in axa[-1]:
                this_group_index2group_label = group_index2group_label
                group_name_fig_ypos = .09
            else:
                this_group_index2group_label = group_index2group_label
                group_name_fig_ypos = .585
            
            # Plot
            my.plot.grouped_bar_plot(
                topl,
                index2plot_kwargs=index2plot_kwargs,
                yerrlo=(topl - topl_err),
                yerrhi=(topl + topl_err),
                group_index2group_label=this_group_index2group_label,
                group_name_kwargs={'size': 12},
                ax=ax,
                group_name_fig_ypos=group_name_fig_ypos,
                )
    
    
    ## Pretty
    # Legend
    f.text(.97, .92, 'C1 contact', ha='center', va='center', size=12, color='b')
    f.text(.97, .88, 'C2 contact', ha='center', va='center', size=12, color='g')
    f.text(.97, .84, 'C3 contact', ha='center', va='center', size=12, color='r')

    # Pretty each ax
    for ax in axa.flatten():
        # Despine
        my.plot.despine(ax)
        
        # Line at zero
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], 'k-', lw=.75)
        ax.set_xlim(xlim)
        
        # Limits and ticks
        yticklabels = np.array([1, 1.5, 2])
        ylim = np.array([.9, 2])
        ax.set_yticks(np.log(yticklabels))
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(np.log(ylim))
        
        # Labels
        if ax in axa[:, 0]:
            ax.set_ylabel('firing rate gain')
        
        if ax in axa[-1]:
            ax.set_xlabel('recording location', labelpad=30)
        
    
    ## Save
    f.savefig(
        'BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM.svg')
    f.savefig(
        'BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__BAR_PLOT_DISCRIMINATION_CONTACT_COEFS_EACH_WHISKER_BY_TASK_RECLOC_AND_STRATUM'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('including all neurons with contact coefficients\n')
        fi.write('\nN=...\n')
        fi.write(agg.size().unstack('task').to_string() + '\n\n')
        fi.write(agg.size().unstack('task').sum().to_string() + '\n\n')
        fi.write('error bars: SEM\n')
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))


if PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY:
    ## Get data
    # Pool across tasks because largely similar
    data = extract_contact_responses_to_each_whisker(
        coef_wscale_df, big_waveform_info_df,
        include_task=('discrimination', 'detection',),
        )

    
    ## Summarize
    data['bestresp'] = data[['C1', 'C2', 'C3']].max(1)
    data['selectivity'] = (
        data[['C1', 'C2', 'C3']].max(1) - data[['C1', 'C2', 'C3']].min(1))


    ## Summarize
    gobj = data.groupby(['NS', 'stratum'])
    rec_l = []
    rec_keys_l = []
    for group_key, sub_data in gobj:
        m_bestresp = sub_data['bestresp'].mean()
        m_selectivity = sub_data['selectivity'].mean()
        ci_bestresp = my.bootstrap.simple_bootstrap(
            sub_data['bestresp'].values)[2]
        ci_selectivity = my.bootstrap.simple_bootstrap( 
            sub_data['selectivity'].values)[2]
        
        rec_l.append({  
            'm_bestresp': m_bestresp,
            'm_selectivity': m_selectivity,
            'ci_lo_bestresp': ci_bestresp[0],
            'ci_hi_bestresp': ci_bestresp[1],
            'ci_lo_selectivity': ci_selectivity[0],
            'ci_hi_selectivity': ci_selectivity[1],
            'NS': group_key[0],
            'stratum': group_key[1]
            })
    parameterized = pandas.DataFrame.from_records(rec_l).set_index(
        ['stratum', 'NS']).sort_index()
    
    # Reorder
    parameterized = parameterized.reindex(['superficial', 'deep'], level=0)
    
    
    ## Plot
    f, ax = my.plot.figure_1x1_small()
    my.plot.grouped_bar_plot(
        parameterized['m_bestresp'],
        yerrlo=parameterized['ci_lo_bestresp'],
        yerrhi=parameterized['ci_hi_bestresp'],
        index2label=extras.index2label__stratum_NS,
        group_index2group_label=extras.group_index2group_label__stratum_NS,
        index2plot_kwargs=extras.index2plot_kwargs__stratum_NS,        
        ax=ax)

    ax.set_title('best whisker')
    ax.set_ylabel('firing rate gain')
    my.plot.despine(ax)
    yticklabels = (1, 2)
    ax.set_yticks(np.log(yticklabels))
    ax.set_yticklabels(yticklabels)

    f.savefig('PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_strength.svg')
    f.savefig('PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_strength.png', dpi=300)

    f, ax = my.plot.figure_1x1_small()
    my.plot.grouped_bar_plot(
        parameterized['m_selectivity'],
        yerrlo=parameterized['ci_lo_selectivity'],
        yerrhi=parameterized['ci_hi_selectivity'],
        index2label=extras.index2label__stratum_NS,
        group_index2group_label=extras.group_index2group_label__stratum_NS,
        index2plot_kwargs=extras.index2plot_kwargs__stratum_NS,        
        ax=ax)

    ax.set_title('selectivity')
    ax.set_ylabel('best / worst whisker')
    my.plot.despine(ax)
    yticklabels = (1, 2)
    ax.set_yticks(np.log(yticklabels))
    ax.set_yticklabels(yticklabels)

    f.savefig('PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_selectivity.svg')
    f.savefig('PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY_selectivity.png', dpi=300)


    ## Stats
    n_by_task = data.reset_index().groupby('task').size()
    n_by_cell_type = data.reset_index().groupby(
        ['task', 'stratum', 'NS']).size().unstack('task').T
    
    stats_filename = 'STATS__PLOT_CONTACT_COEF_STRENGTH_AND_SELECTIVITY'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('# neurons by task:\n' + n_by_task.to_string() + '\n\n')
        fi.write('# neurons by task, stratum, NS:\n' + n_by_cell_type.to_string() + '\n\n')
        fi.write('error bars: 95% bootstrapped CIs\n')
        fi.write('including only neurons for which we have contacts by each whisker\n')
        fi.write('pooling over tasks because similar\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
    print(''.join(lines))

    
plt.show()
