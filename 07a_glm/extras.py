import json
import os
import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import my
import my.plot
import my.bootstrap
import extras


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM
import scipy.stats


## Ordering
stratum_NS_ordering_idx = pandas.MultiIndex.from_tuples([
    ('superficial', False),
    ('superficial', True),
    ('deep', False),
    ('deep', True),
    ], names=['stratum', 'NS'])
    
    
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
    
def index2plot_kwargs__NS2color(ser):
    """Color bars solid blue or red by NS"""
    if ser['NS']:
        ec = 'b'
    else:
        ec = 'r'

    return {'ec': ec, 'fc': ec}


## Other functions
def aggregate_coefs_over_neurons(data, colnames):
    ## Aggregate over neurons
    agg = data.groupby(['stratum', 'NS'])[colnames]
    agg_mean = agg.mean()
    
    # Test whether colnames is singleton or list
    if data[colnames].ndim == 1:
        typ = 'singleton'
    else:
        typ = 'list'
    
    
    ## CIs
    if typ == 'singleton':
        CI_l = []
        CI_keys_l = []
        
        # Iterate over groups
        for group_key, sub_data in agg:
            CI = my.bootstrap.simple_bootstrap(sub_data)[2]
            CI_l.append(CI)
            CI_keys_l.append(list(group_key))
        
        # DataFrame them
        agg_err = pandas.DataFrame(CI_l, columns=['lo', 'hi'], 
            index=pandas.MultiIndex.from_tuples(
            CI_keys_l, names=['stratum', 'NS']))    
    
        agg_errlo = agg_err['lo']
        agg_errhi = agg_err['hi']
    
    elif typ == 'list':
        CI_l = []
        CI_keys_l = []
        
        ## Confint by bootstrap
        # Iterate over groups
        for group_key, sub_data in agg:
            
            # Iterate over coef_metric
            for coef_metric in colnames:
                CI = my.bootstrap.simple_bootstrap(sub_data[coef_metric])[2]
                CI_l.append(CI)
                CI_keys_l.append(list(group_key) + [coef_metric])
        
        # DataFrame them
        agg_err = pandas.DataFrame(CI_l, columns=['lo', 'hi'], 
            index=pandas.MultiIndex.from_tuples(
            CI_keys_l, names=['stratum', 'NS', 'coef_metric']))    
    
        
        ## Binomial confint for the signif
        # This ends up being generally pretty similar to bootstrapping
        # So not using currently
        CI_l = []
        CI_keys_l = []
        for group_key, sub_data in agg:
            for coef_metric in ['pos_sig', 'neg_sig']:
                CI = my.stats.binom_confint(data=sub_data[coef_metric].values)
                CI_l.append(CI)
                CI_keys_l.append(list(group_key) + [coef_metric])
        agg_err_binom = pandas.DataFrame(CI_l, columns=['lo', 'hi'], 
            index=pandas.MultiIndex.from_tuples(
            CI_keys_l, names=['stratum', 'NS', 'coef_metric']))  

    
        ## Extract lo and hi
        agg_errlo = agg_err['lo'].unstack('coef_metric')
        agg_errhi = agg_err['hi'].unstack('coef_metric')
    
    
    ## Order
    agg_mean = agg_mean.reindex(stratum_NS_ordering_idx)
    agg_errlo = agg_errlo.reindex(stratum_NS_ordering_idx)
    agg_errhi = agg_errhi.reindex(stratum_NS_ordering_idx)

    
    ## Return
    return {
        'agg': agg,
        'mean': agg_mean,
        'lo': agg_errlo,
        'hi': agg_errhi,
        }


def horizontal_bar_pie_chart_signif(mfracsig, ax=None):
    """Pie chart signif 
    
    """
    
    ## Order
    mfracsig = mfracsig.loc[stratum_NS_ordering_idx]
    bar_pos = [0, 1, 3, 4]
    tick_pos = [.5, 3.5]
    tick_labels = ['superficial', 'deep'] # double check matches order
    colors = [  
        'blue' if NS else 'red' 
        for NS in mfracsig.index.get_level_values('NS')]
    
    
    ## Calculate the borders of the bars
    pos_neg_border = mfracsig['pos_sig']
    neg_width = mfracsig['neg_sig']
    neg_ns_border = pos_neg_border + neg_width
    ns_width = 1 - neg_ns_border
    
    
    ## Plot the pos_sig
    bar_container = ax.bar(
        x=bar_pos,
        bottom=0,
        height=pos_neg_border.values,
        ec='k', fc='lightgray', alpha=.7, lw=0,
        )
    
    for color, patch in zip(colors, bar_container.patches):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    # Plot positive again to get the edge alpha correct
    bar_container = ax.bar(
        x=bar_pos,
        bottom=0,
        height=pos_neg_border.values,
        ec='k', fc='none', lw=.8,
        )    

    for color, patch in zip(colors, bar_container.patches):
        patch.set_edgecolor(color)
    
    
    ## Plot the neg_sig
    bar_container = ax.bar(
        x=bar_pos,
        bottom=pos_neg_border.values,
        height=neg_width.values,
        ec='k', fc='lightgray', alpha=.3, lw=0,
        )
    
    for color, patch in zip(colors, bar_container.patches):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    # Plot negative again to get the edge alpha correct
    bar_container = ax.bar(
        x=bar_pos,
        bottom=pos_neg_border.values,
        height=neg_width.values,
        ec='k', fc='none', lw=.8,
        )    

    for color, patch in zip(colors, bar_container.patches):
        patch.set_edgecolor(color)
    
    
    ## Plot nonsig
    bar_container = ax.bar(
        x=bar_pos,
        bottom=neg_ns_border.values,
        height=ns_width.values,
        ec='k', fc='w', lw=.8,
        clip_on=False,
        )    

    for color, patch in zip(colors, bar_container.patches):
        patch.set_edgecolor(color)
    
    
    ## Pretty
    my.plot.despine(ax)
    ax.set_xticks(tick_pos)
    ax.xaxis.set_tick_params(length=0, pad=8)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim((-.5, np.max(bar_pos) + 0.5))
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_yticklabels(('0.0', '', '0.5', '', '1.0'))
    
    ax.set_ylabel('fraction of neurons')

    return ax


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
    
    
