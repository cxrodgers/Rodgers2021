## No whiskers plot

import json
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import my.plot
import scipy.stats


my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Get data
whisker_trim_dir = os.path.join(params['pipeline_input_dir'], 'whisker_trim')
big_peri = pandas.read_pickle(os.path.join(whisker_trim_dir, 'big_peri'))
wt_bigtm = pandas.read_pickle(os.path.join(whisker_trim_dir, 'wt_bigtm'))


## Calculate perf
# Include only random, non-opto trials
wt_bigtm = wt_bigtm[
    wt_bigtm.isrnd &
    wt_bigtm['opto'].isin([0, 2]) &
    wt_bigtm['outcome'].isin(['hit', 'error'])
    ]

# Calculate perf
session_perf = wt_bigtm['outcome'].groupby(
    'session').value_counts().unstack('outcome').fillna(0).astype(np.int)
session_perf['perf'] = session_perf['hit'].divide(
    session_perf['hit'] + session_perf['error'])

# Join perf
big_peri = big_peri.join(session_perf['perf'], on='session')


## Extract perf
peri_perf = big_peri['perf'].unstack('mouse')

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