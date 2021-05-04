## Waveform plot
"""
5B
    PLOT_WAVEFORM_WIDTH_HIST
    N/A
    Histogram of waveform widths

5B, inset
    PLOT_WAVEFORMS
    N/A
    Inset of waveforms
"""

import json
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)

## Load waveform info stuff
# The metadata
big_waveform_info_df = my.dataload.load_bwid(params)
big_waveform_info_df['layer'] = big_waveform_info_df['layer'].astype(str)

# The actual waveforms
big_waveform_df = pandas.read_pickle(
    os.path.join(params['unit_db_dir'], 'big_waveform_df'))
big_waveform_df = big_waveform_df.loc[big_waveform_info_df.index]


## Normalize and align each waveform
t_waveform = (np.arange(82) - 82 / 2) / 30.0

# normalize
normed_big_waveform_df = big_waveform_df.copy()
normed_big_waveform_df = normed_big_waveform_df.divide(
    normed_big_waveform_df.std(1), axis=0)

# Apply the shift
shifted_l = []
for idx in normed_big_waveform_df.index:
    # Get the appropriate shift to align peak to time zero
    shift = int(big_waveform_info_df.loc[idx, 'peak_idx'] - len(t_waveform) // 2)
    
    # Apply the shift
    shifted = normed_big_waveform_df.loc[idx].shift(-shift)
    
    # Store
    shifted_l.append(shifted)
shifted_df = pandas.concat(shifted_l, axis=1, verify_integrity=True, 
    sort=True).T


## Plot
PLOT_WAVEFORMS = True
PLOT_WAVEFORM_WIDTH_HIST = True


if PLOT_WAVEFORMS:
    f, ax = plt.subplots(figsize=(1, 0.8))
    #~ f.subplots_adjust(bottom=.2, left=.15, right=.95, wspace=.4)
    f.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Shift and plot each
    for NS, sub_bwid in big_waveform_info_df.groupby('NS'):
        # Data for this NS
        sub_shifted = shifted_df.loc[sub_bwid.index]
        
        # Subslice
        sub_shifted = sub_shifted.iloc[::3]

        # Plot
        color = 'b' if NS else 'r'
        ax.plot(t_waveform, sub_shifted.values.T, color=color, alpha=.1)

    # The cutoff is 8 samples (266.67us) and above are RS
    # So 7 samples (233.33us) is the widest a NS can be
    # Split the difference and say the "cutoff" is 250us (if we had that
    # much temporal resolution)
    #~ ax.plot([.250, .250], [-6, 4], 'k-', lw=.5)
    #~ ax.plot((-.5, 1), [0, 0], 'k-', lw=.5)
    ax.set_xlim((-.500, 1))
    ax.set_ylim((-7, 3))
    ax.set_yticks([])
    ax.set_xticks([])
    #~ ax.set_ylabel('normalized waveform')
    #~ ax.set_xlabel('time from peak (ms)')
    my.plot.despine(ax, which=('left', 'right', 'top', 'bottom'))

    f.savefig('PLOT_WAVEFORMS.svg')
    f.savefig('PLOT_WAVEFORMS.png', dpi=300)

if PLOT_WAVEFORM_WIDTH_HIST:
    # Histo
    f, ax = my.plot.figure_1x1_small()
    for NS in [True, False]:
        # Data for this NS
        sub_ntwd = big_waveform_info_df.loc[big_waveform_info_df['NS'] == NS]
        sub_widths = big_waveform_info_df.loc[sub_ntwd.index, 'width']

        #~ # Hist
        #~ counts, edges = np.histogram(sub_widths.values,
            #~ bins=np.arange(30).astype(np.int))
        
        # Bar plot
        color = 'b' if NS else 'r'
        #~ ax.bar(edges[:-1] / 30., counts, 
            #~ align='center', width=1/30., ec=color, fc='none')
        ax.hist(
            sub_widths.values/30., bins=np.arange(30)/30., 
            histtype='step', color=color)
    
    # Pretty
    ax.set_xlim((0, 0.7))
    ax.set_xticks((0, .3, .6))
    ax.set_ylabel('number of neurons')
    ax.set_xlabel('waveform half-width (ms)')
    #~ f.text(.95, .9, 'NS', color='b', ha='right')
    #~ f.text(.95, .825, 'RS', color='r', ha='right')
    my.plot.despine(ax)
    f.savefig('PLOT_WAVEFORM_WIDTH_HIST.svg')
    f.savefig('PLOT_WAVEFORM_WIDTH_HIST.png', dpi=300)

plt.show()

