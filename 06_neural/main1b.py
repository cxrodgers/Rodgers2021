## Plots from main1a
"""
5F	
    CYCLE_PLOT_SPIKES_BY_STRATUM_fold
    STATS__CYCLE_PLOT_SPIKES_BY_STRATUM	
    PSTH versus time in cycle, for whisks with or without contact, by cell type

S5B	
    CYCLE_PLOT_SPIKES_BY_STRATUM_hz
    STATS__CYCLE_PLOT_SPIKES_BY_STRATUM	
    PSTH versus time in cycle, for whisks with or without contact, by cell type, in Hz

"""

import matplotlib.pyplot as plt
import pandas
import numpy as np
import my.plot
import json
import os


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Set up plotting
my.plot.manuscript_defaults()
my.plot.font_embed()
this_WHISKER2COLOR = {'C0': 'gray', 'C1': 'b', 'C2': 'g', 'C3': 'r'}


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data from main1a to plot
mean_by_neuron_and_shift = pandas.read_pickle('mean_by_neuron_and_shift')
n_by_neuron_and_shift = pandas.read_pickle('n_by_neuron_and_shift')
mean_FR_per_bin = pandas.read_pickle('mean_FR_per_bin')

# Insert task level
mean_by_neuron_and_shift = my.misc.insert_level(
    mean_by_neuron_and_shift, name='task', 
    func=lambda idx: session_df.loc[idx['session'].values, 'task'].values)
n_by_neuron_and_shift = my.misc.insert_level(
    n_by_neuron_and_shift, name='task', 
    func=lambda idx: session_df.loc[idx['session'].values, 'task'].values)

# Or just droplevel to combine both tasks
# Results are qualitatively similar but scaled up for detection ... 
# Quantify this
mean_by_neuron_and_shift = mean_by_neuron_and_shift.droplevel('task')
n_by_neuron_and_shift = n_by_neuron_and_shift.droplevel('task')


## Plots
CYCLE_PLOT_SPIKES_BY_STRATUM = True
CYCLE_PLOT_SPIKES_BY_STRATUM_NOCELLTYPE = False # This is not in the paper

## Iterate over units
for units in ['fold', 'Hz']:
    
    ## Normalize according to units
    if units == 'fold':
        # Slice out baselin just for these neurons
        this_mean_FR = mean_FR_per_bin.loc[
            pandas.MultiIndex.from_frame(
            mean_by_neuron_and_shift.index.to_frame()[
            ['session', 'neuron']].drop_duplicates())]
        
        # Normalize each neuron to its baseline
        norm_mbnas = mean_by_neuron_and_shift.divide(this_mean_FR)
        assert not norm_mbnas.isnull().any()
    
    elif units == 'Hz':
        # Divide by binwidth
        norm_mbnas = mean_by_neuron_and_shift / .005

    else:
        1/0
    
    
    ## Now mean and sem over neurons
    # Group over everything except trial * cycle
    gobj2 = norm_mbnas.groupby([
        lev for lev in mean_by_neuron_and_shift.index.names 
        if lev not in ['session', 'neuron']
        ])

    # Aggregate
    mean_by_neuron_and_cycle_type = gobj2.mean().dropna().unstack(
        ['shift', 'typ'])
    sem_by_neuron_and_cycle_type = gobj2.sem().dropna().unstack(
        ['shift', 'typ'])
    n_by_neuron_and_cycle_type = gobj2.size().unstack(
        ['shift', 'typ'])

    # There should be the same number of neurons for each shift * typ
    # Note that this wouldn't be true for very long shift or very rare typ
    n_neurons_by_layer_and_NS = n_by_neuron_and_cycle_type.iloc[:, 0]
    assert (n_by_neuron_and_cycle_type.sub(
        n_neurons_by_layer_and_NS, axis=0) == 0).all().all()

    # Stack in preparation for grouping
    mean_by_neuron_and_cycle_type = mean_by_neuron_and_cycle_type.stack('typ')
    sem_by_neuron_and_cycle_type = sem_by_neuron_and_cycle_type.stack('typ')


    ## Now mean and sem over neurons, without cell type
    # Group over everything except trial * cycle
    gobj3 = norm_mbnas.groupby([
        lev for lev in mean_by_neuron_and_shift.index.names 
        if lev not in ['session', 'neuron', 'stratum', 'NS']
        ])

    # Aggregate
    mean_by_neuron_and_cycle_type3 = gobj3.mean().dropna().unstack('shift')
    sem_by_neuron_and_cycle_type3 = gobj3.sem().dropna().unstack('shift')
    n_by_neuron_and_cycle_type3 = gobj3.size().unstack('shift')

    # There should be the same number of neurons for each shift * typ
    # Note that this wouldn't be true for very long shift or very rare typ
    n_neurons_by_layer_and_NS3 = n_by_neuron_and_cycle_type3.iloc[:, 0]
    assert (n_by_neuron_and_cycle_type3.sub(
        n_neurons_by_layer_and_NS3, axis=0) == 0).all().all()


    ## Plots
    if CYCLE_PLOT_SPIKES_BY_STRATUM:
        ## PSTHs: strata in columns, cycle typ in rows
        # Neural subpops (rows)
        stratum_l = ['superficial', 'deep']
        NS_l = [False, True]

        # Cycle subpops (cols)
        cycle_typ_l = ['contact', 'none']
            

        ## Create handles
        #~ f, axa = my.plot.figure_1x2_small(sharex=True, sharey=True)
        f, axa = plt.subplots(2, 1, figsize=(2.2, 3.5), sharex=True, sharey=True)
        f.subplots_adjust(left=.25, bottom=.15, hspace=.4, top=.925, right=.95)
        

        ## Group by stratum, NS, and typ
        grouping_keys = ['stratum', 'NS', 'typ']
        gobj = mean_by_neuron_and_cycle_type.groupby(grouping_keys)

        # Iterate over groups
        for grouped_keys, sub_resp in gobj:
            # Zip up the keys
            grouped_d = dict(zip(grouping_keys, grouped_keys))
            
            
            ## Get ax
            try:
                ax = axa[
                    #~ stratum_l.index(grouped_d['stratum']),
                    #NS_l.index(grouped_d['NS']),
                    cycle_typ_l.index(grouped_d['typ']),
                    ]
            except ValueError:
                continue
            
            
            ## Plot
            # Color by NS
            color = 'b' if grouped_d['NS'] else 'r'
            
            # Linestyle by stratum
            linestyle = '-' if grouped_d['stratum'] == 'deep' else '--'
            
            # Slice data
            assert sub_resp.shape[0] == 1
            topl = sub_resp.iloc[0].sort_index()
            
            # Slice error
            topl_err = sem_by_neuron_and_cycle_type.loc[grouped_keys].sort_index()
            assert (topl.index == topl_err.index).all()
            
            # Plot
            ax.plot(
                topl.index.values * 5, 
                topl.values, 
                color=color, linestyle=linestyle)
            ax.fill_between(
                x=(topl.index.values * 5),
                y1 =((topl.values - topl_err.values)),
                y2 =((topl.values + topl_err.values)),
                color=color, alpha=.3, lw=0)

            
            ## Pretty
            # Title
            if grouped_d['typ'] == 'contact':
                ax.set_title('with contact')
            elif grouped_d['typ'] == 'none':
                ax.set_title('without contact')


        ## Pretty
        for ax in axa.flatten():
            my.plot.despine(ax)
            ax.set_xlim((-30, 30))
            
            if units == 'Hz':
                ax.set_ylim((0, 45))
                ax.set_yticks((0, 20, 40))
            
            elif units == 'fold':
                ax.set_ylim((0, 6))
                ax.set_yticks((0, 3, 6))
                ax.plot([-30, 30], [1, 1], 'k-', lw=.8)

        
        # Label x-axis
        axa[1].set_xlabel('time from cycle peak (ms)')

        # Label y-axis
        f.text(.075, .53, 'firing rate gain', rotation=90, ha='center', va='center')


        ## Save
        f.savefig('CYCLE_PLOT_SPIKES_BY_STRATUM_{}.svg'.format(units))
        f.savefig('CYCLE_PLOT_SPIKES_BY_STRATUM_{}.png'.format(units), dpi=300)


    if CYCLE_PLOT_SPIKES_BY_STRATUM_NOCELLTYPE:
        ## Cycle subpops (cols)
        cycle_typ_l = ['contact', 'none']


        ## Create handles
        f, axa = my.plot.figure_1x2_small(sharex=True, sharey=True)
        f.subplots_adjust(top=.825, bottom=.25)


        ## Group by stratum, NS, and typ
        # Iterate over cycle_typ_l
        for cycle_typ in cycle_typ_l:
            ## Get ax
            try:
                ax = axa[
                    cycle_typ_l.index(cycle_typ)
                    ]
            except ValueError:
                continue
            
            
            ## Plot
            color = 'k'
            linestyle = '-'
            
            # Slice data
            sub_resp = mean_by_neuron_and_cycle_type3.loc[cycle_typ].sort_index()
            #~ assert sub_resp.shape[0] == 1
            topl = sub_resp.sort_index()
            
            # Slice error
            topl_err = sem_by_neuron_and_cycle_type3.loc[cycle_typ].sort_index()
            assert (topl.index == topl_err.index).all()
            
            # Plot
            ax.plot(
                topl.index.values * 5, 
                topl.values, 
                color=color, linestyle=linestyle)
            ax.fill_between(
                x=(topl.index.values * 5),
                y1 =((topl.values - topl_err.values)),
                y2 =((topl.values + topl_err.values)),
                color=color, alpha=.3, lw=0)

            
            ## Pretty
            # Title
            if cycle_typ == 'contact':
                ax.set_title('with contact')
            elif cycle_typ == 'none':
                ax.set_title('without contact')


        ## Pretty
        for ax in axa.flatten():
            my.plot.despine(ax)
            ax.set_xlim((-30, 30))
            
            if units == 'Hz':
                ax.set_ylim((0, 45))
                ax.set_yticks((0, 20, 40))
            
            elif units == 'fold':
                ax.set_ylim((0, 3))
                ax.set_yticks((0, 1, 2, 3))
                ax.plot([-30, 30], [1, 1], 'k-', lw=.8)

        
        # Label x-axis
        f.text(.575, .045, 'time from cycle peak (ms)', ha='center', va='center')

        # Label y-axis
        if units == 'Hz':
            f.text(.05, .52, 'firing rate (Hz)', ha='center', va='center', rotation=90)
        elif units == 'fold':
            axa[0].set_ylabel('normalized firing rate')


        ## Save
        f.savefig('CYCLE_PLOT_SPIKES_BY_STRATUM_NOCELLTYPE_{}.svg'.format(units))
        f.savefig('CYCLE_PLOT_SPIKES_BY_STRATUM_NOCELLTYPE_{}.png'.format(units), dpi=300)

## Stats
stats_filename = 'STATS__CYCLE_PLOT_SPIKES_BY_STRATUM'
with open(stats_filename, 'w') as fi:
    fi.write(stats_filename + '\n')
    fi.write('number of neurons:\n')
    fi.write(n_neurons_by_layer_and_NS.to_string() + '\n')
    fi.write('combining across tasks\n')
    fi.write('error bars: SEM\n')

with open(stats_filename) as fi:
    lines = fi.readlines()
print(''.join(lines))


plt.show()
