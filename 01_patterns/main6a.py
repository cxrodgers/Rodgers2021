## Quantify performance of whisker tracking
# relies on trainset_dir
"""
S2D
    WHISKER_TRACKING_ERROR_RATE_BY_DATASET
    N/A
    Error rate of whisker tracking algorithm    
"""

import json
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import my
import my.plot

my.plot.font_embed()
my.plot.manuscript_defaults()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Paths to the last tracking run
last_curation_dir = params['trainset_dir']


## Load curated joints
joints_df = pandas.read_pickle(
    os.path.join(last_curation_dir, 'joints_df_with_folds'))
fold_ids = joints_df[['session', 'frame', 'fold']].drop_duplicates().set_index(
    ['session', 'frame']).sort_index()
joints_df = joints_df.set_index(['session', 'frame', 'whisker', 'joint'])    

    
## Load results of main7a in the last curation run
optimized_predictions = pandas.read_pickle(
    os.path.join(last_curation_dir, 'optimized_predictions'))
optimized_wwr = pandas.read_pickle(
    os.path.join(last_curation_dir, 'optimized_wwr'))
optimized_fwr = pandas.read_pickle(
    os.path.join(last_curation_dir, 'optimized_fwr'))


## Rename dataset
optimized_wwr['dataset'] = optimized_wwr['dataset'].replace({
    '18-10-16': 1,
    '19-06-27': 2,
    '19-07-05': 3,
    '19-12-17': 4,
    '20-04-05': 5,
    }).astype(np.int)
optimized_fwr['dataset'] = optimized_fwr['dataset'].replace({
    '18-10-16': 1,
    '19-06-27': 2,
    '19-07-05': 3,
    '19-12-17': 4,
    '20-04-05': 5,
    }).astype(np.int)
    

## Aggregate fwr
# Mean within dataset * mouse
optimized_fwr['contains_error'] = optimized_fwr['contains_error'].astype(np.int)
optimized_fwr['contains_missing'] = optimized_fwr['contains_missing'].astype(np.int)
fwr_by_mouse_and_dataset = optimized_fwr.groupby(['dataset', 'mouse']).mean()

# Extract metrics
metrics_topl = ['contains_error', 'contains_missing', 'mean_errdist']
fwr_by_mouse_and_dataset = fwr_by_mouse_and_dataset.loc[:, metrics_topl].copy()


## Aggregate wwr
# For simplicity, pool poor and poor tip
optimized_wwr['typ'] = optimized_wwr['typ'].replace({'poor_tip': 'poor'})

# Count wwr typ
wwr_typ_by_mouse_and_dataset = optimized_wwr.groupby(
    ['dataset', 'mouse'])['typ'].value_counts().unstack('typ').fillna(0)

# Normalize
wwr_typ_by_mouse_and_dataset = wwr_typ_by_mouse_and_dataset.divide(
    wwr_typ_by_mouse_and_dataset.sum(1), axis=0)

# Extract metrics
# Ignore nearly correct for this purpose
metrics_topl = ['extraneous', 'incorrect', 'missing', 'poor']
wwr_by_mouse_and_dataset = wwr_typ_by_mouse_and_dataset.loc[:, metrics_topl].copy()


## Concatenate
er_by_mouse_and_dataset = pandas.concat(
    [fwr_by_mouse_and_dataset, wwr_by_mouse_and_dataset], axis=1)


typ2color = {
    'extraneous': 'r', 'incorrect': 'g', 'missing': 'gray', 
    'nearly correct': 'orange', 'poor': 'pink'}
def index2plot_kwargs(idx):
    color = typ2color[idx[1]]
    return {'fc': color, 'ec': 'k'}

# grouped bar plot
topl = er_by_mouse_and_dataset.stack().unstack('mouse').reindex(
    ['incorrect', 'extraneous', 'poor', 'missing'], level=1)

# convert to percent
topl *= 100

f, ax = plt.subplots(figsize=(5, 2.5))
f.subplots_adjust(left=.15, bottom=.22, right=.98)

my.plot.despine(ax)
my.plot.grouped_bar_plot(
    topl,
    index2plot_kwargs=index2plot_kwargs,
    plot_error_bars_instead_of_points=True,
    ax=ax,
    group_name_fig_ypos=.15,
    )

f.text(.2, .9, 'incorrectly classified', color='g', ha='left', va='center', size=12)
f.text(.2, .82, 'false positive', color='r', ha='left', va='center', size=12)
f.text(.2, .74, 'poorly traced', color='pink', ha='left', va='center', size=12)
f.text(.2, .66, 'false negative', color='gray', ha='left', va='center', size=12)

ax.set_xticks([])
ax.set_ylim((0, 4))
ax.set_yticks((0, 2, 4))
ax.set_ylabel('error rate\n(% of whiskers)')
ax.set_xlabel('dataset', labelpad=20)

# Plot zeros
is_zero = topl.mean(1) == 0
for (dataset, typ) in is_zero.index[is_zero.values]:
    xtick = (
        (dataset - 1) * 5 + 
        ['incorrect', 'extraneous', 'poor', 'missing'].index(typ)
        )
    color = typ2color[typ]
    ax.text(xtick, .2, '0', ha='center', va='center', size=12, color=color)

f.savefig('WHISKER_TRACKING_ERROR_RATE_BY_DATASET.svg')
f.savefig('WHISKER_TRACKING_ERROR_RATE_BY_DATASET.png', dpi=300)


## Dump stats
stats_filename = 'STATS__WHISKER_TRACKING_ERROR_RATE_BY_DATASET'
meaned_er_over_mice_pct = topl.loc[1].mean(1)
with open(stats_filename, 'w') as fi:
    fi.write(stats_filename + '\n')
    fi.write('Whisker tracking error rate on dataset 1, meaned over mice (%):\n')
    fi.write(meaned_er_over_mice_pct.to_string() + '\n')
    fi.write('\nTotal of above (all types of errors): {:.3f}%\n'.format(
        meaned_er_over_mice_pct.sum()))
    fi.write('Grand accuracy rate: {:.3f}%'.format(
        100 - meaned_er_over_mice_pct.sum()))
plt.show()
