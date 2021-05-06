## Firing rate analyses
"""
5D
    PLOT_MEAN_FIRING_RATE_BY_DEPTH
    STATS__PLOT_MEAN_FIRING_RATE_BY_DEPTH
    Depth plot of mean firing rate by cell type
"""
# This dumps FR_overall (including ITIs) in neural_dir

import json
import os
import pandas
import numpy as np
import kkpandas
import MCwatch
import whiskvid
import runner.models
import tqdm
import matplotlib.pyplot as plt
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load waveform info stuff
big_waveform_info_df = my.dataload.load_bwid(params, drop_1_and_6b=True)

    
## Iterate over sessions
rec_l = []
rec_keys_l = []

for session_name in tqdm.tqdm(big_waveform_info_df.index.levels[0]):
    ## Get session objects
    gs = runner.models.GrandSession.objects.filter(name=session_name).first()
    vs = whiskvid.django_db.VideoSession.from_name(session_name)


    ## Load spikes
    spikes = pandas.read_pickle(
        os.path.join(vs.session_path, 'spikes'))
    included_clusters = np.sort(spikes['cluster'].unique())

    # Only iterate over those left in bwid
    analyze_clusters = big_waveform_info_df.loc[session_name].index.values
    
    # Estimate duration of this session
    duration_total = spikes['time'].max() - spikes['time'].min()

    
    ## Fold on rwin
    for cluster in analyze_clusters:
        # Extract cluster times
        this_cluster = spikes.loc[spikes['cluster'] == cluster, 'time']
    
        # Store records
        rec_l.append({
            'n_spikes_total': len(this_cluster),
            'duration_total': duration_total,
            })
        rec_keys_l.append((session_name, cluster))


## Concat the firing rates
resdf = pandas.DataFrame.from_records(rec_l, 
    index=pandas.MultiIndex.from_tuples(rec_keys_l, 
    names=['session', 'neuron']))

# Join on metadata
resdf = resdf.join(big_waveform_info_df[
    ['stratum', 'layer', 'NS', 'Z_corrected']], 
    on=['session', 'neuron'])

# Parameterize
resdf['FR_total'] = resdf['n_spikes_total'] / resdf['duration_total']


## Plot
PLOT_MEAN_FIRING_RATE_BY_DEPTH = True
if PLOT_MEAN_FIRING_RATE_BY_DEPTH:
    ## Parameters
    # Seems more or less consistent across tasks (though 5A is more sharply
    # peaked during detection) so pool.
    
    
    ## Plot
    f, ax = plt.subplots(1, 1, figsize=(2.6, 2.4))
    f.subplots_adjust(bottom=.28, left=.3, right=.95, top=.95)
    resdf['log_FR_total'] = np.log10(resdf['FR_total'])
    my.plot.smooth_and_plot_versus_depth(
        resdf, 'log_FR_total', ax=ax,
        layer_boundaries_ylim=(-1, 2))
    
    # Y-axis
    ax.set_ylim((-1, 2))
    ax.set_yticks((-1, 0, 1, 2))
    ax.set_yticklabels(('0.1', '1', '10', '100'))
    ax.set_ylabel('firing rate (Hz)', labelpad=-3)
    
    # Legend
    f.text(.9, .4, 'inhib.', color='b', ha='center', va='center')
    f.text(.9, .32, 'excit.', color='r', ha='center', va='center')
    
    
    ## Save
    f.savefig('PLOT_MEAN_FIRING_RATE_BY_DEPTH.svg')
    f.savefig('PLOT_MEAN_FIRING_RATE_BY_DEPTH.png', dpi=300)
    
    
    ## Stats
    stats_filename = 'STATS__PLOT_MEAN_FIRING_RATE_BY_DEPTH'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} neurons ({} excit, {} inhib)\n'.format(
            len(resdf),
            len(resdf[resdf['NS'] == False]),
            len(resdf[resdf['NS'] == True]),
            ))
        fi.write('L1 and L6b excluded\n')
        fi.write('Both tasks included\n')

    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))

plt.show()


## Dump the overall firing rates
FR_overall = resdf['FR_total'].rename('firing_rate')
FR_overall.to_pickle(os.path.join(params['neural_dir'], 'FR_overall'))