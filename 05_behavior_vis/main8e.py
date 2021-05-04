## Compare performance at various levels of trims
# These mice are used:
# Index(['219CR', '221CR', '229CR', '231CR', 'KF119', 'KF132', 'KF134', 'KM101', 'KM131'], dtype='object', name='mouse')
# Only KM91 is missing, and that is because we have no data before first trim

"""
S1C
    PLOT_PERFORMANCE_BY_N_ROWS
    N/A
    Performance versus number of rows of whiskers
"""
import json
import os
import datetime
import pytz
import tqdm
import numpy as np
import pandas
import matplotlib.pyplot as plt
import MCwatch.behavior
import runner.models
import whisk_video.models
import my

## Plotting params
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    
    
## Load metadata about sessions
orig_session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


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


## Load trims
trims = MCwatch.behavior.db.get_whisker_trims_table()


## Choose behavioral sessions
# Mice to analyze
mouse_l = list(task2mouse.loc['discrimination'])

# approx 2017-1-1 is when the servo spacing got finalized
dt_start = datetime.datetime(
    year=2017, month=1, day=1).astimezone(pytz.timezone('America/New_York'))

# apply filter: mouse name, date, and 3 positions
session_table = session_table[
    session_table['mouse'].isin(mouse_l) &
    (session_table['date_time_start'] > dt_start) &
    (session_table['scheduler'] == 'Auto') &
    session_table['stimulus_set'].isin(
        ['trial_types_2shapes_CCL_3srvpos', 'trial_types_CCL_3srvpos']) &
    (session_table.index != '20170406092802.KM101') & # screwed up servo pos for some reason
    (session_table['mouse'] != 'KM91') # no data from before first trim
    ].copy()

#~ # Filter
#~ gs_qs = runner.models.GrandSession.objects.filter(
    #~ session__python_param_stimulus_set__in=[
        #~ 'trial_types_2shapes_CCL_3srvpos', 'trial_types_CCL_3srvpos'],
    #~ session__date_time_start__gte=dt_start,
    #~ session__mouse__name__in=list(task2mouse.loc['discrimination']),
    #~ )
#~ session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))

session_name_l = sorted(session_table.index)


## Iterate over sessions
rec_l = []
tm_l = []
tm_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Get mouse name
    mouse_name = session_name.split('.')[1]
    
    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = MCwatch.behavior.db.get_trial_matrix(session_name)

    # Identify servo spacing
    servo_pos = np.sort(trial_matrix['servo_pos'].unique())
    servo_pos_is_correct = (servo_pos == np.array([1670, 1760, 1850])).all()
    assert servo_pos_is_correct
    

    ## Identify whether this was an opto session
    gs = runner.models.GrandSession.objects.filter(session__name=session_name).first()
    has_opto_session = False
    is_sham = False
    if gs is not None:
        try:
            opto_session = gs.optosession
            has_opto_session = True
        except runner.models.OptoSession.DoesNotExist:
            pass

        if has_opto_session:
            is_sham = opto_session.sham
            assert is_sham in [True, False] 

    # Set opto column properly
    if (not has_opto_session) or is_sham:
        trial_matrix['opto'] = 0    
    

    ## Store
    tm_l.append(trial_matrix)
    tm_keys_l.append((mouse_name, session_name))


## Concat    
tmdf = pandas.concat(tm_l, keys=tm_keys_l, names=['mouse', 'session']).sort_index()


## Add trim to session_table
# Slice trims by included mice and sort by time
trims = trims.loc[
    trims['Mouse'].isin(session_table['mouse'].unique())].sort_values('dt')

# Apply each trim in order
session_table['trim'] = 'all'
for trim_idx in trims.index:
    # Get mouse for this trim
    mouse = trims.loc[trim_idx, 'Mouse']
    trim_dt = trims.loc[trim_idx, 'dt']
    trim_label = trims.loc[trim_idx, 'Which Spared']
    
    # Mask out rows of session_df 1) for this mouse 2) >= dt
    mask = (
        (session_table['mouse'] == mouse) &
        (session_table['date_time_start'] >= trim_dt)
        )
    
    # Apply trim label to those rows
    session_table.loc[mask, 'trim'] = trim_label


## Count rows
session_table['n_rows'] = session_table['trim'].map({
    'C*; b': 1,
    'C*; g': 1,
    'all': 5,
    'C*; D*; b; g': 2,
    'B*; C*; D*; a; b; g': 3,
    'C*; b; g': 1,
    'None': 0,
    'B*; C*; D*; b; g': 3,
    'B*; C*; D*; b; g; d': 3,
    'C*; D*; g; d': 2,
    })
session_table['n_rows'] = session_table['n_rows'].astype(np.int)    

# Exclude 0 rows for now
session_table = session_table.loc[session_table['n_rows'] > 0].copy()


## Make sure n_rows is decreasing
session_table = session_table.sort_values('date_time_start')
n_rows_l = []
n_rows_keys_l = []
drop_session_l = []
for mouse, mouse_table in session_table.groupby('mouse'):
    n_rows = mouse_table['n_rows']
    assert (n_rows.diff().dropna() <= 0).all()
    n_rows_l.append(n_rows.values)
    n_rows_keys_l.append(mouse)
    
    # Keep only the ~10 most recent sessions with 5 rows
    assert n_rows.iloc[0] == 5
    all_mask = np.where(n_rows == 5)[0]
    if len(all_mask) >= 10:
        drop_sessions = n_rows.index.values[all_mask[:-10]]
        drop_session_l.append(drop_sessions)

# Apply the drop
all_drop = np.concatenate(drop_session_l)
session_table = session_table.drop(all_drop)

# Apply that mask to tmdf
tmdf = my.misc.slice_df_by_some_levels(tmdf, session_table.index)


## More processing of trial matrix
# Include only easy
tmdf = tmdf.loc[tmdf['stepper_pos'].isin([50, 150])].copy()

# Include only random non-spoiled
tmdf = tmdf.loc[tmdf['outcome'].isin(['hit', 'error']) & tmdf.isrnd].copy()

# Drop opto
# Definitely include opto == 0 (sham and no laser), 
# but what about 2 (laser off during actual opto test)?
tmdf = tmdf.loc[tmdf['opto'].isin([0, 2])].copy()
tmdf.index = tmdf.index.remove_unused_levels()


## Mask out sessions
# This doesn't change much
# Count trials of each type per session
n_trials_per_session = tmdf.groupby(
    ['mouse', 'session', 'rewside', 'servo_pos']).size().unstack(
    ['rewside', 'servo_pos']).fillna(0).astype(np.int)

# Require >60 trials total and >10 of each type
include_mask = (
    (n_trials_per_session.sum(1) >= 60) & 
    (n_trials_per_session > 10).all(1)
    )

# Apply include_mask
tmdf = my.misc.slice_df_by_some_levels(
    tmdf, include_mask.index[include_mask.values])
tmdf.index = tmdf.index.remove_unused_levels()

# Apply include_mask to session_table
session_table = session_table.loc[
    tmdf.index.get_level_values('session').unique()]


## Count sessions per mouse * n_rows and ensure enough of each type
# 229CR has no 3-row sessions
# The others sometimes have as few as 1 or 2 sessions per n_rows
n_session_by_mouse_and_nrows = session_table.groupby(
    ['mouse', 'n_rows']).size().unstack()


## Group and aggregate perf
perf_by_session = tmdf.groupby(
    ['rewside', 'servo_pos', 'mouse', 'session'])[
    'outcome'].value_counts().unstack('outcome')
perf_by_session['perf'] = perf_by_session['hit'] / (
    perf_by_session['hit'] + perf_by_session['error'])


## Join n_rows on perf
perf_by_session = perf_by_session.join(session_table['n_rows'])
perf_by_session = perf_by_session.set_index(
    'n_rows', append=True).reorder_levels(
    ['mouse', 'n_rows', 'rewside', 'servo_pos', 'session']).sort_index()


## Aggregate 
# Over sessions within mouse
agg_perf = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev != 'session'])

# Over mice
mean_perf = agg_perf.mean(
    level=[lev for lev in agg_perf.index.names 
    if lev != 'mouse']).unstack('n_rows')
err_perf = agg_perf.sem(
    level=[lev for lev in agg_perf.index.names 
    if lev != 'mouse']).unstack('n_rows')


## Plot
PLOT_PERFORMANCE_BY_N_ROWS = True

if PLOT_PERFORMANCE_BY_N_ROWS:
    
    rewside_l = ['left', 'right']
    row2color = {1: 'b', 2: 'g', 3: 'r', 5: 'k'}
    #~ f, axa = plt.subplots(1, len(rewside_l), figsize=(6.5, 2.75))
    #~ f.subplots_adjust(wspace=.4, bottom=.225, right=.975, left=.125)
    f, axa = my.plot.figure_1x2_small(sharey=True)
    f.subplots_adjust(top=.85, bottom=.275)
    for n_rows in mean_perf.columns:
        if n_rows == 0:
            continue
        
        color = row2color[n_rows]
        linestyle = '-'
        
        for rewside in rewside_l:
            ax = axa[rewside_l.index(rewside)]
            ax.set_title({'left': 'concave', 'right': 'convex'}[rewside])

            topl = mean_perf.loc[rewside, n_rows]
            topl_err = err_perf.loc[rewside, n_rows]
            
            ax.plot(topl, linestyle=linestyle, color=color)

            ax.fill_between(
                x=topl.index, 
                y1=(topl - topl_err),
                y2=(topl + topl_err),
                lw=0, alpha=.2, color=color)

    for ax in axa:
        my.plot.despine(ax)
        ax.set_xlim((1870, 1650))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_ylim((0.5, 1))
        ax.set_xticks((1670, 1760, 1850))
        ax.set_xticklabels(('far', 'med.', 'close'))
        #~ ax.set_xlabel('stimulus position')
        #~ ax.set_ylabel('performance')
        ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.8)

    axa[0].set_ylabel('performance')
    f.text(.55, .035, 'stimulus position', ha='center', va='center')
    
    # Legend
    f.text(.55, .85, '5 rows', size=12, color=row2color[5], ha='center', va='center')
    f.text(.55, .775, '3 rows', size=12, color=row2color[3], ha='center', va='center')
    f.text(.55, .7, '2 rows', size=12, color=row2color[2], ha='center', va='center')
    f.text(.55, .625, '1 row', size=12, color=row2color[1], ha='center', va='center')


    ## Save
    f.savefig('PLOT_PERFORMANCE_BY_N_ROWS.svg')
    f.savefig('PLOT_PERFORMANCE_BY_N_ROWS.png', dpi=300)
    

    stats_filename = 'STATS__PLOT_PERFORMANCE_BY_N_ROWS'
    stats_data = agg_perf.unstack('mouse')
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice;\n{}\n'.format(stats_data.shape[1], stats_data.columns))
        fi.write('error bars: SEM over mice\n')
    
    with open(stats_filename) as fi:
        lines = fi.readlines()
        print('\n'.join(lines))


plt.show()