## No whiskers plot
# TODO: replace the use of get_perf_metrics with actual trial matrix
# 

import json
import os
import numpy as np
import pandas
import MCwatch.behavior
import runner.models
import matplotlib.pyplot as plt
import my.plot
import scipy.stats


my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)

    
## Get data
session_table = MCwatch.behavior.db.get_django_session_table()
session_table.index.name = 'session'
pdf = MCwatch.behavior.db.get_perf_metrics()
trims = MCwatch.behavior.db.get_whisker_trims_table()

# Index by session
pdf = pdf.set_index('session').sort_index()

# Fix this
pdf = pdf.drop(['20150610152812.KM38', '20160520155855.KM63'])
assert not pdf.index.duplicated().any()


## The single-whisker trim mice
mouse_names = [
    '228CR',
    '229CR',
    '231CR',
    '245CR',
    '255CR',
    '267CR',
    'KF132',
    'KF134',
    'KM101',
    'KM131',
    ]

## Iterate over mouse
N_PRE = 5
ptb_keys_l = []
ptb_l = []
for mouse in mouse_names:
    ## Slice by mouse
    mouse_bdf = session_table[session_table['mouse'] == mouse]
    mouse_trims = trims[trims['Mouse'] == mouse]

    # Exclude LickTrain, for which there is no perf
    mouse_bdf = mouse_bdf[
        ~mouse_bdf['scheduler'].isin(
        ['ForcedAlternationLickTrain', 'LickTrain'])]

    # Join on perf
    mouse_bdf = mouse_bdf.join(pdf)

    # Sort by date
    mouse_bdf = mouse_bdf.sort_values('date_time_start')

    ## Find the day of None
    iidx = np.where(mouse_trims['Which Spared'] == 'None')[0]
    assert len(iidx) == 1
    assert iidx[0] == len(mouse_trims) - 1
    dt_none = mouse_trims['dt'].iloc[-1]


    ## Split the perf into pre- and post- trim
    pre_trim_bdf = mouse_bdf[
        mouse_bdf['date_time_start'] < dt_none].iloc[-N_PRE:].copy()
    pre_trim_bdf['n_day'] = np.arange(-N_PRE, 0)
    post_trim_bdf = mouse_bdf[
        mouse_bdf['date_time_start'] >= dt_none].copy()
    post_trim_bdf['n_day'] = np.arange(0, len(post_trim_bdf))
    
    # Concat
    peri_trim_bdf = pandas.concat([pre_trim_bdf, post_trim_bdf])
    

    ## Store
    ptb_l.append(peri_trim_bdf)
    ptb_keys_l.append(mouse)


## Concat
big_peri = pandas.concat(
    ptb_l, keys=ptb_keys_l, names=['mouse']).set_index(
    'n_day', append=True).reset_index('session')


## Extract perf
peri_perf = big_peri['perf_unforced'].unstack('mouse')

# Bin into pre and post
binned_pp = pandas.concat([
    peri_perf.loc[[-3, -2, -1]].mean(),
    peri_perf.loc[[0]].mean(),
    ], axis=1, keys=['pre', 'post'],
    ).T


## Stats
ttr = scipy.stats.ttest_rel(
    binned_pp.loc['pre'].values, binned_pp.loc['post'].values,)

stats_filename = 'STATS__WHISKER_TRIM'
with open(stats_filename, 'w') as fi:
    fi.write('n = {} mice\n'.format(binned_pp.shape[1]))
    fi.write('perf before trim: mean {:.4f}, sem {:.4f}\n'.format(
        binned_pp.loc['pre'].mean(),
        binned_pp.loc['pre'].sem(),
        ))
    fi.write('perf after trim: mean {:.4f}, sem {:.4f}\n'.format(
        binned_pp.loc['post'].mean(),
        binned_pp.loc['post'].sem(),
        ))        
    fi.write('two-sample t-test: p = {:g}\n'.format(ttr.pvalue))

with open(stats_filename) as fi:
    lines = fi.readlines()
print(''.join(lines))
    
    
## Plot
f, ax = plt.subplots(figsize=(2.25, 2.5))
f.subplots_adjust(left=.3, right=.95, top=.9, bottom=.225)

# Plot individual mice
ax.plot(
    binned_pp.values,
    color='gray', marker=None, alpha=.5, lw=1,
    )

# Plot the mean
ax.plot(
    binned_pp.mean(1).values,
    marker='o', color='k', lw=3, mfc='k', ms=6)

# sig
ax.plot([0, 1], [.98, .98], 'k-', lw=.8)
if ttr.pvalue < .001:
    ax.text(.5, .98, '***', ha='center', va='bottom')
else:
    1/0


## Pretty
ax.set_xlim((-0.25, 1.25))
ax.set_xticks((0, 1))
ax.set_xticklabels(('pre', 'post'))
ax.set_ylim((.4, 1))
ax.set_ylabel(('performance'))
ax.set_xlabel('whisker trim')
ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.8)
my.plot.despine(ax)


## Save
f.savefig('WHISKER_TRIM.svg')
f.savefig('WHISKER_TRIM.png', dpi=300)

plt.show()