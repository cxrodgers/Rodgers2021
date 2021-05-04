## Cross-correlations of contact times
"""
2D	PLOT_XCORR_MEANED_OVER_WHISKER	
    STATS__PLOT_XCORR_MEANED_OVER_WHISKER	
    Correlograms of contact times within and across whiskers
"""
import json
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import whiskvid
import my
import kkpandas
import my.plot
import matplotlib


## Fonts
my.plot.manuscript_defaults()
my.plot.font_embed()
DK = chr(916) + chr(954)


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(
    params, 'big_tm')
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')
   
# Load the contact results
# These have to be manually sliced to remove opto trials
big_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_ccs_df'))


## Use big_tm to slice big_ccs_df
# Get included_trials
included_trials = big_tm.index

# Use those cycles to slice big_ccs_df
big_ccs_df = big_ccs_df.set_index(
    ['trial', 'whisker', 'cycle'], append=True).reorder_levels(
    ['session', 'trial', 'cycle', 'whisker', 'cluster']).sort_index()
big_ccs_df = my.misc.slice_df_by_some_levels(
    big_ccs_df, C2_whisk_cycles.index)

# Return big_ccs_df to its original index
big_ccs_df = big_ccs_df.reset_index().set_index(['session', 'cluster'])


## Stack onsets
session_l = sorted(np.unique(big_ccs_df.index.get_level_values('session')))
whisker_l = ['C1', 'C2', 'C3']
count_l = []
count_keys_l = []
for session in session_l:
    # Get contacts for this session
    session_ccs = big_ccs_df.loc[session]
    
    # Iterate over whiskers
    for nw0 in range(len(whisker_l)):
        for nw1 in range(len(whisker_l)):
            # Get whiskers
            w0 = whisker_l[nw0]
            w1 = whisker_l[nw1]
            auto = w0 == w1
            
            # Get contact 
            w0_frames = session_ccs.loc[
                session_ccs['whisker'] == w0, 'frame_start'].copy()
            w1_frames = session_ccs.loc[
                session_ccs['whisker'] == w1, 'frame_start'].copy()
            
            # Skip if nothing
            if len(w0_frames) == 0:
                continue
            if len(w1_frames) == 0:
                continue

            # Jitter to break ties
            w0_frames += 0.1 * np.random.standard_normal(size=w0_frames.size)
            w1_frames += 0.1 * np.random.standard_normal(size=w1_frames.size)
            
            # Correlate
            count, bins = kkpandas.utility.correlogram(
                w0_frames / 200., 
                w1_frames / 200., 
                bin_width=.005, limit=0.15, auto=auto,
                )

            # Normalize: This was calculating by folding on the spikes in 
            # the shorter signal, so divide by the length of the shorter signal
            # to make it into an expectation of the number of spikes per bin.
            rate = count / float(np.min([len(w0_frames), len(w1_frames)]))
            
            # Store
            count_l.append(rate)
            count_keys_l.append(
                (session, 'auto' if auto else 'cross', w0, w1))


## Bincenters
bincenters = (bins[:-1] + bins[1:]) / 2.0


## Concat the correlograms
corrdf = pandas.DataFrame(
    count_l, 
    index=pandas.MultiIndex.from_tuples(
    count_keys_l, names=['session', 'typ', 'w0', 'w1']),
    columns=bincenters,
    )

# Add task level
corrdf = my.misc.insert_mouse_and_task_levels(corrdf, mouse2task)

# Drop detection because so few contacts
# Results for detection are actually basically the same, but there's some
# missing N because of the small number of contacts, so it's annoying
corrdf = corrdf.loc['discrimination']


## Aggregate
# Mean across sessions
mouse_mcorrdf = corrdf.mean(
    level=[lev for lev in corrdf.index.names if lev != 'session'])

# Slice out auto
mouse_auto_mcorrdf = mouse_mcorrdf.xs('auto', level='typ').droplevel('w1')

# Slice out cross, and drop non-adjacent whiskers
mouse_cross_mcorrdf = mouse_mcorrdf.xs('cross', level='typ')
idxdf = mouse_cross_mcorrdf.index.to_frame()
drop_mask = (
    (mouse_cross_mcorrdf.index.get_level_values('w0') == 'C1') &
    (mouse_cross_mcorrdf.index.get_level_values('w1') == 'C3')
    )
mouse_cross_mcorrdf = mouse_cross_mcorrdf.loc[~drop_mask]
drop_mask = (
    (mouse_cross_mcorrdf.index.get_level_values('w0') == 'C3') &
    (mouse_cross_mcorrdf.index.get_level_values('w1') == 'C1')
    )
mouse_cross_mcorrdf = mouse_cross_mcorrdf.loc[~drop_mask]

# Mean across mice, leaving whisker
whisker_auto_mcorrdf = mouse_auto_mcorrdf.mean(level='w0')
whisker_cross_mcorrdf = mouse_cross_mcorrdf.mean(level=['w0', 'w1'])

# Grand mean first across whisker, leaving mice; then mean and sem over mice
# Doing all pairs of whiskers here to make it symmetric
to_agg_auto_mcorrdf = mouse_auto_mcorrdf.mean(level='mouse')
to_agg_cross_mcorrdf = mouse_cross_mcorrdf.mean(level='mouse')
agg_auto_mcorrdf = to_agg_auto_mcorrdf.mean()
agg_cross_mcorrdf = to_agg_cross_mcorrdf.mean()
aggerr_auto_mcorrdf = to_agg_auto_mcorrdf.sem()
aggerr_cross_mcorrdf = to_agg_cross_mcorrdf.sem()


## Plots
PLOT_XCORR_MEANED_OVER_WHISKER = True

if PLOT_XCORR_MEANED_OVER_WHISKER:
    ## Plot auto and cross meaned over whiskers
    f, ax = plt.subplots(figsize=(2.5, 2.25))
    f.subplots_adjust(bottom=.25, left=.27, right=.97, top=.9)
    
    # Plot auto
    ax.plot(agg_auto_mcorrdf, color='k', lw=1, clip_on=False)
    ax.fill_between(
        x=agg_auto_mcorrdf.index.values,
        y1=(agg_auto_mcorrdf - aggerr_auto_mcorrdf),
        y2=(agg_auto_mcorrdf + aggerr_auto_mcorrdf),
        color='k', alpha=.2, lw=0,
        )
    
    # Plot cross
    ax.plot(agg_cross_mcorrdf, color='k', lw=1, ls='--',clip_on=False)
    ax.fill_between(
        x=agg_cross_mcorrdf.index.values,
        y1=(agg_cross_mcorrdf - aggerr_cross_mcorrdf),
        y2=(agg_cross_mcorrdf + aggerr_cross_mcorrdf),
        color='k', alpha=.2, lw=0,
        )
    
    # Pretty
    my.plot.despine(ax)
    ax.set_xlabel('{}t (s)'.format(chr(916)))
    ax.set_ylabel('Pr(contact)')
    ax.set_xlim((-.15, .15))
    ax.set_xticks((-.1, 0, .1))
    ax.set_ylim((0, .2))
    ax.set_yticks((0, .1, .2))
    
    # Legend
    ax.plot([-.022, .0], [.22, .22], 'k--', lw=1, clip_on=False)
    ax.plot([-.022, .0], [.195, .195], 'k-', lw=1, clip_on=False)
    ax.text(.02, .22, 'across whisker', size=12, ha='left', va='center')
    ax.text(.02, .195, 'within whisker', size=12, ha='left', va='center')
    
    # Save
    f.savefig('PLOT_XCORR_MEANED_OVER_WHISKER.svg')
    f.savefig('PLOT_XCORR_MEANED_OVER_WHISKER.png', dpi=300)
    
    # Stats
    with open('STATS__PLOT_XCORR_MEANED_OVER_WHISKER', 'w') as fi:
        # Check same for all
        assert to_agg_auto_mcorrdf.index.equals(to_agg_cross_mcorrdf.index)
        
        fi.write('STATS__PLOT_XCORR_MEANED_OVER_WHISKER\n')
        fi.write('n = {} mice (all disc.)\n'.format(len(to_agg_auto_mcorrdf)))
        fi.write('error bars: sem\n')
    
    with open('STATS__PLOT_XCORR_MEANED_OVER_WHISKER', 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))
    

plt.show()
