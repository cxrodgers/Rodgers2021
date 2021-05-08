## This compares across models
"""
8G	
    PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_contact_ll_per_whisk_wrt_null.svg	
    STATS__PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_contact_ll_per_whisk_wrt_null
    Bar plot of GOF for various models of contact responses, using DLL per whisk

S8A	
    PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_contact_score_wrt_null.svg
    STATS__PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_contact_score_wrt_null
    Bar plot of GOF for various models of contact responses, using pR2

7B	BAR_PLOT_ADDITIVE_MODEL_COMPARISON_ll	
    STATS__BAR_PLOT_ADDITIVE_MODEL_COMPARISON_ll
    Bar plot of GOF for one feature family at a time, using DLL per whisk

S7B	BAR_PLOT_ADDITIVE_MODEL_COMPARISON_score
    STATS__BAR_PLOT_ADDITIVE_MODEL_COMPARISON_score
    Bar plot of GOF for one feature family at a time, using pR2
    
5C	BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON	
    STATS__BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON	
    Bar plot of GOF for excluding one feature family at a time
"""
import json
import os
import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import my
import my.plot
import my.bootstrap


## Helper function
def model_compare_horizontal_bar(
    metric_scores, metric_err, models_to_compare_df, ax, pvalue_ser=None,
    signif_xpos=None):
    """Horizontal bar plot likelihoods of models to compare"""
    
    ## Drop models that don't need to be plotted and sort in order of position
    models_to_compare_df = models_to_compare_df.dropna().sort_values('position')
    
    # Sort
    metric_scores = metric_scores.loc[
        models_to_compare_df['pretty_name'].values]
    metric_err = metric_err.loc[
        models_to_compare_df['pretty_name'].values]

    # Choose topl err
    topl_err = np.array(
        [metric_err['mpl_lo'].values, metric_err['mpl_hi'].values])
    
    
    ## Horizontal bar
    yticks = models_to_compare_df['position'].values
    bar_container = ax.barh(
        yticks,
        width=metric_scores,
        xerr=topl_err,
        height=.8,
        facecolor='lightgray', edgecolor='k',
        error_kw={'lw': 1},
    )
    
    # Unclip the error bars
    bar_container.errorbar[2][0].set_clip_on(False)
    
    
    ## Model name labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(metric_scores.index.values, size=12)
    
    # Invert the y-axis
    ax.set_ylim((np.max(yticks) + .5, np.min(yticks) - .5))

    
    ## Asterisks
    if pvalue_ser is not None:
        for ytick, model in zip(yticks, metric_scores.index):
            try:
                pvalue = pvalue_ser.loc[model]
            except (IndexError, KeyError):
                continue
            
            if pvalue < .001:
                ax.text(signif_xpos, ytick, '***', ha='left', va='center', size=12)
            elif pvalue < .01:
                ax.text(signif_xpos, ytick, '**', ha='left', va='center', size=12)
            elif pvalue < .05:
                ax.text(signif_xpos, ytick, '*', ha='left', va='center', size=12)
            else:
                ax.text(signif_xpos, ytick, 'n.s.', ha='left', va='center', size=12)
    

## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Set up plotting
my.plot.poster_defaults()
my.plot.font_embed()

DELTA = chr(916)


## Do each model in turn
model_names = [
    # Full (too big to fit, but useful for extracting the features)
    'full',

    # Null model
    'null',

    # NULL_PLUS models -- This is how to identify potentially useful factors
    'whisking',
    'contact_binarized',
    'task',
    'fat_task',
    
    # The minimal model
    'minimal',
    
    # Minimal with whisk permutation
    'minimal+permute_whisks_with_contact',

    # Minimal with random_regressor
    #~ 'minimal+random_regressor',

    # MINIMAL_MINUS models
    # Whether the minimal model contains anything unnecessary
    'minimal-whisking',
    'minimal-contacts',
    'minimal-task',
    
    # CONTACTS_PLUS models
    # This identifies any additional features about contacts that matter at all
    'contact_binarized+contact_interaction',
    'contact_binarized+contact_angle',
    'contact_binarized+kappa_min',
    'contact_binarized+kappa_max',
    'contact_binarized+kappa_std',
    'contact_binarized+velocity2_tip',
    'contact_binarized+n_within_trial',
    'contact_binarized+contact_duration',
    'contact_binarized+contact_stimulus',
    'contact_binarized+xw_latency_on',
    'contact_binarized+phase',
    'contact_binarized+xw_angle',
    'contact_binarized+touching',
    
    # CONTACTS_MINUS
    # Currently this is just to test whether whisker identity matters
    'contact_count_by_time',
    
    # WHISKING
    # To compare the coding for position of each whisker
    'start_tip_angle+amplitude_by_whisker',
    'start_tip_angle+global_amplitude',
]


## For grouped_bar_plot
def index2label__stratum_NS(ser):
    return '' #ser['stratum']

def group_index2group_label__stratum_NS(stratum):
    return stratum

def index2plot_kwargs__stratum_NS(ser):
    if ser['NS']:
        ec = 'b'
    else:
        ec = 'r'
    fc = 'none'
    
    return {'ec': ec, 'fc': fc}


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params)

    
## Paths
# Where to get patterns from
glm_results_dir = os.path.join(params['glm_dir'], 'results')


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load the baseline firing rates
FR_overall = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'FR_overall')
    )
    

## Load each model
coef_wscale_df_l = []
fitting_results_df_l = []
keys_l = []

for model_name in model_names:
    ## Load results of this model
    # Where it is
    model_results_dir = os.path.join(glm_results_dir, model_name)
    
    # Load
    coef_wscale_df = pandas.read_pickle(os.path.join(
        model_results_dir, 'coef_wscale_df'))
    fitting_results_df = pandas.read_pickle(os.path.join(
        model_results_dir, 'fitting_results_df'))


    ## Store
    coef_wscale_df_l.append(coef_wscale_df)
    fitting_results_df_l.append(fitting_results_df)
    keys_l.append(model_name)

# Concat
big_coef_wscale_df = pandas.concat(coef_wscale_df_l, keys=keys_l, names=['model'])
big_fitting_results_df = pandas.concat(fitting_results_df_l, keys=keys_l, names=['model'])


## Include only those results left in big_waveform_info_df 
# (e.g., after dropping 1 and 6b)
big_coef_wscale_df = my.misc.slice_df_by_some_levels(
    big_coef_wscale_df, big_waveform_info_df.index)
big_fitting_results_df = my.misc.slice_df_by_some_levels(
    big_fitting_results_df, big_waveform_info_df.index)

# Count the neurons remaining
fit_neurons = big_coef_wscale_df.index.to_frame()[
    ['model', 'task', 'session', 'neuron']].drop_duplicates().reset_index(drop=True)
n_fit_neurons_by_model = fit_neurons.groupby('model').size()
print("# neurons fit in these models:\n{}\n".format(n_fit_neurons_by_model))


## Normalize likelihood to the null, and for the amount of data
big_fitting_results_df['ll_per_whisk'] = ((
    big_fitting_results_df['likelihood'] - 
    big_fitting_results_df['null_likelihood']) / 
    big_fitting_results_df['len_ytest'])

big_fitting_results_df['contact_ll_per_whisk'] = ((
    big_fitting_results_df['contact_likelihood'] - 
    big_fitting_results_df['contact_null_likelihood']) / 
    big_fitting_results_df['len_ytest_contacts'])

# Convert nats to bits
big_fitting_results_df['ll_per_whisk'] = (
    big_fitting_results_df['ll_per_whisk'] / np.log(2))
  
big_fitting_results_df['contact_ll_per_whisk'] = (
    big_fitting_results_df['contact_ll_per_whisk'] / np.log(2))

    
## Plot flags
PLOT_CONTACTS_PLUS_COMPARISON_PRETTY = True
BAR_PLOT_ADDITIVE_MODEL_COMPARISON = True
BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON = True


if PLOT_CONTACTS_PLUS_COMPARISON_PRETTY:
    ## Compare models that added various parameters of contact
    # The metrics to extract from fitting_results
    # Focus on scores during contact, not overall
    metric_l = [
        'contact_score',
        'contact_ll_per_whisk',
        ]
    
    # The scores to actually plot
    # _wrt_null is generated by subtracting off the contact_binarized null model
    use_metric_l = [
        'contact_score_wrt_null',
        'contact_ll_per_whisk_wrt_null',
    ]


    ## Get data in the same way as the other Fig 6 panels, to drop the same neurons
    # Slice contact_binarized
    coef_wscale_df = big_coef_wscale_df.loc['minimal']
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


    ## Slice out actual_best
    # Because we might need different regularization for each model
    sliced = big_fitting_results_df.xs('actual_best', level='analysis').copy()
    
    # Drop now-meaningless permute and reg_lambda levels
    sliced = sliced.droplevel(
        ['n_reg_lambda', 'n_permute'])
    assert not sliced.index.duplicated().any()

    # Pool across task because similar
    sliced = sliced.droplevel('task')
    

    ## Slice out metrics
    sliced = sliced.loc[:, metric_l].copy()
    sliced.columns.name = 'glm_metric'
    
    
    ## Mean metrics over folds
    scores = sliced.mean(
        level=[lev for lev in sliced.index.names if lev != 'n_fold'])
    
    # Unstack model so replicates on columns
    scores = scores.unstack('model').T.reorder_levels(
        ['model', 'glm_metric']).sort_index()
    
    
    ## Now slice the neurons to include only those in `data`
    scores = scores.loc[:, data.index].copy()
    
    # Error check
    assert not scores.isnull().any().any()
    assert (scores.index.to_frame().reset_index(drop=True).groupby(
        'model').size() == len(metric_l)).all()

    
    ## Choose the models to compare
    models_to_compare_df = pandas.DataFrame.from_records([
        # Baseline
        (0, 'contact_binarized', 'baseline (contacts with whisker identity)',),
        
        # Various removals
        #~ (0, 'null', 'without contacts',), # This is way too negative
        (1, 'contact_count_by_time', 'without whisker identity',),
        
        # Various additions
        (3, 'contact_binarized+xw_angle', 'with cross-whisker angle',),
        (4, 'contact_binarized+xw_latency_on', 'with cross-whisker latency',),
        (5, 'contact_binarized+contact_angle', 'with contact angle',),
        (6, 'contact_binarized+phase', 'with contact phase',),
        (7, 'contact_binarized+n_within_trial', 'with contact history',),
        (8, 'contact_binarized+contact_interaction', 'with cross-whisker interaction',),
        (9, 'contact_binarized+kappa_std', 'with contact-induced bending',),
        #~ (10, 'contact_binarized+touching', 'with touching',), # This one helps!
        (11, 'contact_binarized+contact_stimulus', 'with stimulus identity',),
        ], columns=['position', 'model_name', 'pretty_name'])

    
    # Slice
    scores = scores.loc[models_to_compare_df['model_name'].values]

    # Rename with pretty names
    pretty_name_d = models_to_compare_df[
        ['model_name', 'pretty_name']].set_index('model_name')[
        'pretty_name'].to_dict()
    scores = scores.rename(index=pretty_name_d)
    
    
    ## Baseline by the null model
    null_scores = scores.xs(pretty_name_d['contact_binarized'], level='model')
    scores_minus_null = scores.sub(null_scores, level='glm_metric')
    
    # Concat original scores with _wrt_null scores
    renaming_d = dict([(metric, metric + '_wrt_null') 
        for metric in scores_minus_null.index.levels[1]])
    scores_minus_null = scores_minus_null.rename(index=renaming_d)
    scores = pandas.concat(
        [scores, scores_minus_null], 
        verify_integrity=True).sort_index()

    
    ## Iterate over use_metric_l and plot each
    for use_metric in use_metric_l:
        ## Extract use_metric
        this_scores = scores.xs(use_metric, level='glm_metric')
        
        # Order according to model_names
        this_scores = this_scores.loc[
            models_to_compare_df['pretty_name'].values]
        
        # Get the plotting position of each
        yticks = models_to_compare_df['position'].values
        yticklabels = this_scores.index.values
        
        # Error bars: confidence intervals
        this_scores_err = my.bootstrap.bootstrap_CIs_on_dataframe(this_scores)
        
        
        ## Plot
        f, ax = plt.subplots(figsize=(7, 3.5))
        f.subplots_adjust(left=.525, bottom=.25, top=.95, right=.975)
        
        ax.barh(
            yticks,
            width=this_scores.mean(1).values,
            xerr=[this_scores_err['mpl_lo'].values, this_scores_err['mpl_hi'].values],
            facecolor='w', edgecolor='k',
            error_kw={'lw': 1},
        )
        
        # Labels
        ax.set_yticks(yticks)
        ax.set_yticklabels(this_scores.index.values)

        # Invert the y-axis
        ax.set_ylim((np.max(yticks) + .75, np.min(yticks) - .5))

        # Plot line at zero
        ax.plot([0, 0], ax.get_ylim(), 'k-', lw=.8)
        
        # Set x-label and x-ticks
        if use_metric == 'contact_ll_per_whisk_wrt_null':
            #~ ax.set_xlabel('{}(log2 likelihood) during contact\n(bits / whisk)'.format(DELTA))
            ax.set_xlabel('{}goodness-of-fit during contact\n(bits / whisk)'.format(DELTA))
            ax.set_xlim((-.033, .033))
            ax.set_xticks((-.03, 0, .03))

        elif use_metric == 'll_per_whisk_wrt_null':
            ax.set_xlabel('{}(log2 likelihood)\nbits / whisk'.format(DELTA))
            ax.set_xlim((-.022, .022))
            ax.set_xticks((-.02, 0, .02))            
        
        elif use_metric in ['score_wrt_null', 'contact_score_wrt_null']:
            ax.set_xlabel('pR2 compared to baseline')
        
        else:
            ax.set_xlabel(use_metric)
        
        # Despine
        my.plot.despine(ax)
        
        
        ## Save
        f.savefig('PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_{}.svg'.format(
            use_metric))
        f.savefig('PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_{}.png'.format(
            use_metric), dpi=300)


        ## Stats
        stats_filename = ('STATS__'
            'PLOT_CONTACTS_PLUS_COMPARISON_PRETTY_{}'.format(use_metric))
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(scores.shape[1]))
            fi.write('dropping neurons missing contact responses\n')
            fi.write('pooled across tasks because similar\n')
            fi.write('scores are compared to contact_binarized model\n')
            fi.write('using actual_best to compare across models\n')
            fi.write('error bars: 95% CIs from bootstrapping\n')
        
        with open(stats_filename) as fi:
            lines = fi.readlines()
        print(''.join(lines))

if BAR_PLOT_ADDITIVE_MODEL_COMPARISON:
    ## Params
    # How to metricate the performance
    for metricate in ['score', 'll']:
    
        if metricate == 'score':
            metric_l = ['score', 'contact_score']
        elif metricate == 'll':
            metric_l = ['ll_per_whisk', 'contact_ll_per_whisk']
        else:
            1/0
        
        
        # Just use this analysis
        use_analysis = 'actual_best'
        
        
        ## Choose the models to compare
        models_to_compare_df = pandas.DataFrame.from_records([
            (np.nan, 'null', 'null',),
            (0, 'task', 'task only',),
            (1, 'whisking', 'whisking only',),
            (2, 'contact_binarized', 'contacts only',),
            (4, 'minimal', 'task + whisking + contacts',),
            #~ (5, 'full', 'full',),
            ], columns=['position', 'model_name', 'pretty_name'])

        
        ## Extract scores
        # Slice by analysis and metric
        sliced = big_fitting_results_df.loc[:, metric_l].xs(
            use_analysis, level='analysis').droplevel(
            ['n_reg_lambda', 'n_permute'])
        assert not sliced.index.duplicated().any()
        
        # Mean over folds
        scores = sliced.mean(
            level=[lev for lev in sliced.index.names if lev != 'n_fold'])

        # Pool because similar
        scores = scores.droplevel('task')
        
        # Unstack model so replicates on index
        scores = scores.unstack('model').swaplevel(axis=1).sort_index(axis=1)
        
        # Transpose so replicates on columns
        scores = scores.T
        
        # Slice
        scores = scores.loc[models_to_compare_df['model_name'].values]

        # Rename with pretty names
        pretty_name_d = models_to_compare_df[
            ['model_name', 'pretty_name']].set_index('model_name')[
            'pretty_name'].to_dict()
        scores = scores.rename(index=pretty_name_d)

        # Baseline by the null model
        scores = scores.sub(scores.loc['null'], level=1).sort_index()
        scores = scores.drop('null')
        scores.index = scores.index.remove_unused_levels()
        

        ## Aggregate
        agg_scores = scores.mean(1)
        agg_err = my.bootstrap.bootstrap_CIs_on_dataframe(scores)


        ## Stats
        # Within each metric, compare every (baselined) model vs zero using wilcoxon
        stats_l = []
        stats_keys_l = []
        for metric in metric_l:
            for model_to_compare in scores.index.levels[0]:
                #~ # Skip the combination models
                #~ if model_to_compare in ['full', 'task + whisking + contacts']:
                    #~ continue
                
                # Compare versus zero
                comp_data = scores.loc[model_to_compare].loc[metric]
                test_res = scipy.stats.wilcoxon(
                    comp_data, alternative='two-sided')
                
                # Write
                stats_l.append(test_res.pvalue)
                stats_keys_l.append((metric, model_to_compare))
        
        # Concat
        pvalue_ser = pandas.Series(
            stats_l, index=pandas.MultiIndex.from_tuples(
            stats_keys_l, names=['metric', 'model']))


        ## Plot
        f, axa = plt.subplots(1, len(metric_l), figsize=(5.5, 1.75), sharey=True)
        f.subplots_adjust(left=.38, bottom=.275, top=.875, right=.95, wspace=.6)


        ## Iterate over metric (axis)
        for metric in metric_l:
            ## Get ax
            ax = axa[metric_l.index(metric)]

            # Title ax
            if metric in ['ll_per_whisk', 'score']:
                ax.set_title('all whisks')
            elif metric in ['contact_ll_per_whisk', 'contact_score']:
                ax.set_title('whisks with contact')

            
            ## Slice data
            metric_scores = agg_scores.xs(metric, level=1)
            metric_err = agg_err.xs(metric, level=1)
            metric_pvalue = pvalue_ser.loc[metric]
            

            ## Plot
            # Where significance stuff goes
            if metric == 'll_per_whisk':
                signif_xpos = .06
            elif metric == 'contact_ll_per_whisk':
                signif_xpos = .3
            elif metric == 'score':
                signif_xpos = .08
            elif metric == 'contact_score':
                signif_xpos = .2

            # Bar plot
            model_compare_horizontal_bar(
                metric_scores, metric_err,
                models_to_compare_df, ax, 
                pvalue_ser=metric_pvalue, 
                signif_xpos=signif_xpos)

            # Pretty
            my.plot.despine(ax)


            ## X-axis by metric
            if metric == 'll_per_whisk':
                ax.set_xlim((0, .06))
                ax.set_xticks((0, .03, .06))
            
            elif metric == 'contact_ll_per_whisk':
                ax.set_xlim((0, .3))
                ax.set_xticks((0, .15, .3))

            elif metric == 'score':
                ax.set_xlim((0, .08))
                ax.set_xticks((0, .04, .08))

            elif metric == 'contact_score':
                ax.set_xlim((0, .2))
                ax.set_xticks((0, .1, .2))
            
            else:
                ax.set_xlabel(metric)


        ## Shared x-label
        #~ f.text(.7, .00, '{}(log2 likelihood) bits / whisk\ncompared with null model'.format(chr(916)),
            #~ ha='center', va='center')
        
        if metricate == 'll':
            f.text(.7, .00, '{}goodness-of-fit (bits / whisk)\ncompared with null model'.format(chr(916)),
                ha='center', va='center')
        
        elif metricate == 'score':
            f.text(.7, .00, '{}goodness-of-fit (pseudo R2)\ncompared with null model'.format(chr(916)),
                ha='center', va='center')

        
        ## Save
        f.savefig(
            'BAR_PLOT_ADDITIVE_MODEL_COMPARISON_{}.svg'.format(metricate))
        f.savefig(
            'BAR_PLOT_ADDITIVE_MODEL_COMPARISON_{}.png'.format(metricate), dpi=300)
        

        ## Stats
        stats_filename = 'STATS__BAR_PLOT_ADDITIVE_MODEL_COMPARISON_{}'.format(metricate)
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(scores.shape[1]))
            fi.write('both tasks only\n')
            fi.write(
                'quantified as LL/whisk of model, minus '
                'LL/whisk of null (+drift-population)\n')
            fi.write('error bars: 95% confidence intervals from bootstrapping\n')
        
            # Compare each with a paired mann-whitney
            fi.write('comparing each versus zero using two-sided Wilcoxon:\n')
            fi.write(pvalue_ser.to_string() + '\n')
        
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))
    

if BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON:
    ## Params
    # How to metricate the performance
    for metricate in ['score', 'll']:
        
        if metricate == 'score':
            metric_l = ['score', 'contact_score']
        elif metricate == 'll':
            metric_l = ['ll_per_whisk', 'contact_ll_per_whisk']
        else:
            1/0
        
        # Just use this analysis
        use_analysis = 'actual_best'
        
        
        ## Choose the models to compare
        models_to_compare_df = pandas.DataFrame.from_records([
            (np.nan, 'null', 'null',),
            (0, 'minimal', 'task + whisking + contacts',),
            (2, 'minimal-task', 'without task',),
            (3, 'minimal-whisking', 'without whisking',),
            (4, 'minimal-contacts', 'without contacts',),
            ], columns=['position', 'model_name', 'pretty_name'])

        
        ## Extract scores
        # Slice by analysis and metric
        sliced = big_fitting_results_df.loc[:, metric_l].xs(
            use_analysis, level='analysis').droplevel(
            ['n_reg_lambda', 'n_permute'])
        assert not sliced.index.duplicated().any()
        
        # Mean over folds
        scores = sliced.mean(
            level=[lev for lev in sliced.index.names if lev != 'n_fold'])

        # Pool because similar
        scores = scores.droplevel('task')
        
        # Unstack model so replicates on index
        scores = scores.unstack('model').swaplevel(axis=1).sort_index(axis=1)
        
        # Transpose so replicates on columns
        scores = scores.T
        
        # Slice
        scores = scores.loc[models_to_compare_df['model_name'].values]

        # Rename with pretty names
        pretty_name_d = models_to_compare_df[
            ['model_name', 'pretty_name']].set_index('model_name')[
            'pretty_name'].to_dict()
        scores = scores.rename(index=pretty_name_d)

        # Baseline by the null model TODO: baseline by minimal
        scores = scores.sub(scores.loc['task + whisking + contacts'], level=1).sort_index()
        #~ scores = scores.drop('null')
        scores.index = scores.index.remove_unused_levels()


        ## Aggregate
        agg_scores = scores.mean(1)
        agg_err = my.bootstrap.bootstrap_CIs_on_dataframe(scores)


        ## Stats
        # Within each metric, compare every model vs TWC using wilcoxon
        stats_l = []
        stats_keys_l = []
        for metric in metric_l:
            for model_to_compare in scores.index.levels[0]:
                # Skip self-comparison
                if model_to_compare == 'task + whisking + contacts':
                    continue
                
                # Compare
                null_data = scores.loc['task + whisking + contacts'].loc[metric]
                comp_data = scores.loc[model_to_compare].loc[metric]
                test_res = scipy.stats.wilcoxon(
                    null_data, comp_data, alternative='two-sided')
                
                # Write
                stats_l.append(test_res.pvalue)
                stats_keys_l.append((metric, model_to_compare))
        
        # Concat
        pvalue_ser = pandas.Series(
            stats_l, index=pandas.MultiIndex.from_tuples(
            stats_keys_l, names=['metric', 'model']))


        ## Plot
        f, axa = plt.subplots(1, len(metric_l), figsize=(5.5, 1.75), sharey=True)
        f.subplots_adjust(left=.38, bottom=.275, top=.875, right=.95, wspace=.6)


        ## Iterate over metric (axis)
        for metric in metric_l:
            ## Get ax
            ax = axa[metric_l.index(metric)]

            # Title ax
            if metric in ['ll_per_whisk', 'score']:
                ax.set_title('all whisks')
            elif metric in ['contact_ll_per_whisk', 'contact_score']:
                ax.set_title('whisks with contact')

            
            ## Slice data
            metric_scores = agg_scores.xs(metric, level=1)
            metric_err = agg_err.xs(metric, level=1)
            metric_pvalue = pvalue_ser.loc[metric]
            

            ## Plot
            # Where significance stuff goes
            if metric == 'll_per_whisk':
                signif_xpos = 0.001
                zero_xpos = .0025
            elif metric == 'contact_ll_per_whisk':
                signif_xpos = 0.003
                zero_xpos = .0075
            elif metric == 'score':
                signif_xpos = 0.001
                zero_xpos = .0025
            elif metric == 'contact_score':
                signif_xpos = 0.003
                zero_xpos = .0075

            # Bar plot
            model_compare_horizontal_bar(
                metric_scores, metric_err,
                models_to_compare_df, ax, 
                pvalue_ser=metric_pvalue, 
                signif_xpos=signif_xpos)

            # Pretty
            my.plot.despine(ax, which=('left', 'top'))
            

            ## X-axis by metric
            if metric == 'll_per_whisk':
                ax.set_xlim((-.03, 0))
                ax.set_xticks((-.03, 0))
            
            elif metric == 'contact_ll_per_whisk':
                ax.set_xlim((-.1, 0))
                ax.set_xticks((-.1, -.05, 0))

            elif metric == 'score':
                ax.set_xlim((-.03, 0))
                ax.set_xticks((-.03, 0))

            elif metric == 'contact_score':
                ax.set_xlim((-.1, 0))
                ax.set_xticks((-.1, -.05, 0)) 

            else:
                ax.set_xlabel(metric)

            # Text a zero at the baseline model
            ax.text(zero_xpos, 0, '0', size=12, ha='left', va='center')


        ## Shared x-label
        if metricate == 'll':
            f.text(.7, .00, '{}goodness-of-fit (bits / whisk)\ncompared with "task + whisking + contacts"'.format(chr(916)),
                ha='center', va='center')
        
        elif metricate == 'score':
            f.text(.7, .00, '{}goodness-of-fit (pseudo R2)\ncompared with "task + whisking + contacts"'.format(chr(916)),
                ha='center', va='center')


        ## Save
        f.savefig(
            'BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON_{}.svg'.format(metricate))
        f.savefig(
            'BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON_{}.png'.format(metricate), dpi=300)
            
        
        ## Stats
        stats_filename = 'STATS__BAR_PLOT_SUBTRACTIVE_MODEL_COMPARISON_{}'.format(metricate)
        with open(stats_filename, 'w') as fi:
            fi.write(stats_filename + '\n')
            fi.write('n = {} neurons\n'.format(scores.shape[1]))
            fi.write('discrimination only\n')
            fi.write('error bars: 95% confidence intervals from bootstrapping\n')
        
            # Compare each with a paired mann-whitney
            fi.write('comparing each versus null using two-sided Mann-Whitney:\n')
            fi.write(pvalue_ser.to_string() + '\n')
        
        with open(stats_filename, 'r') as fi:
            lines = fi.readlines()
        print(''.join(lines))


plt.show()    