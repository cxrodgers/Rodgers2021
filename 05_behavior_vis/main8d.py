## Plot performance for flatter shapes

"""
S1A, right
    PLOT_PERFORMANCE_BY_DIFFICULTY
    STATS__PLOT_PERFORMANCE_BY_DIFFICULTY
    Performance on flatter shapes
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


## Behavioral datasets
# In this date range
dt_start = datetime.datetime(
    year=2017, month=1, day=4).astimezone(pytz.timezone('America/New_York'))
dt_stop = datetime.datetime(
    year=2018, month=7, day=1).astimezone(pytz.timezone('America/New_York'))

# Filter
gs_qs = runner.models.GrandSession.objects.filter(
    session__python_param_stimulus_set='trial_types_2shapes_CCL_3srvpos',
    session__date_time_start__gte=dt_start,
    session__date_time_start__lte=dt_stop,
    ).exclude(session__mouse__name__in=[
    'KF95', 
    'KF89', 
    'KF104',
    ]).exclude(tags__name='munged').exclude(name__in=[
    '170615_KM100',
    ])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Load trims
trims = MCwatch.behavior.db.get_whisker_trims_table()


## Iterate over sessions
rec_l = []
tm_l = []
tm_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Get session data
    gs = runner.models.GrandSession.objects.filter(name=session_name).first()
    
    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = MCwatch.behavior.db.get_trial_matrix(gs.session.name)

    # Identify servo spacing
    servo_pos = np.sort(trial_matrix['servo_pos'].unique())
    assert (servo_pos == np.array([1670, 1760, 1850])).all()
    
    # Identify task
    assert (
        gs.session.python_param_stimulus_set 
        == 'trial_types_2shapes_CCL_3srvpos')


    ## Identify whether has video session
    try:
        vs = gs.videosession
        has_video_session = True
    except whisk_video.models.VideoSession.DoesNotExist:
        has_video_session = False

    
    ## Identify whether has opto session
    try:
        opto_session = gs.optosession
        has_opto_session = True
    except runner.models.OptoSession.DoesNotExist:
        has_opto_session = False
    
    if has_opto_session:
        is_sham = opto_session.sham
    else:
        is_sham = False

    
    ## Set opto column properly
    if (not has_opto_session) or is_sham:
        trial_matrix['opto'] = False    
    
    
    ## Store
    rec_l.append({
        'session': session_name,
        'dt_start': gs.session.date_time_start,
        'opto': has_opto_session,
        'video': has_video_session,
        'sham': is_sham,
        'mouse': gs.session.mouse.name,
    })
    tm_l.append(trial_matrix)
    tm_keys_l.append((gs.session.mouse.name, session_name))


## Concat    
session_df = pandas.DataFrame.from_records(rec_l).set_index('session')
tmdf = pandas.concat(tm_l, keys=tm_keys_l, names=['mouse', 'session']).sort_index()


## Add trim to session_df
# Slice trims by included mice and sort by time
trims = trims.loc[
    trims['Mouse'].isin(session_df['mouse'].unique())].sort_values('dt')

# Apply each trim in order
session_df['trim'] = 'all'
for trim_idx in trims.index:
    # Get mouse for this trim
    mouse = trims.loc[trim_idx, 'Mouse']
    trim_dt = trims.loc[trim_idx, 'dt']
    trim_label = trims.loc[trim_idx, 'Which Spared']
    
    # Mask out rows of session_df 1) for this mouse 2) >= dt
    mask = (
        (session_df['mouse'] == mouse) &
        (session_df['dt_start'] >= trim_dt)
        )
    
    # Apply trim label to those rows
    session_df.loc[mask, 'trim'] = trim_label

# Drop those that correspond to trims under 1row
# The ones that remain should all be 1row
session_df = session_df.loc[
    ~session_df['trim'].isin(['C1-2', 'None', 'C2', 'b; C1-2'])].copy()
assert session_df['trim'].isin(['C*; b', 'C*; g']).all()

# Apply that mask to tmdf
tmdf = my.misc.slice_df_by_some_levels(tmdf, session_df.index)


## More processing of trial matrix
# Label hard
tmdf['difficulty'] = 'hard'
tmdf.loc[tmdf['stepper_pos'].isin([50, 150]), 'difficulty'] = 'easy'

# Include only random non-spoiled
tmdf = tmdf.loc[tmdf['outcome'].isin(['hit', 'error']) & tmdf.isrnd]

# Drop opto
# Let's include only those with no laser, rather than laser-off trials
# But sham opto are still included
tmdf = tmdf.loc[tmdf['opto'].isin([0])].copy()
tmdf.index = tmdf.index.remove_unused_levels()


## Mask out sessions
# Count trials of each type per session
n_trials_per_session = tmdf.groupby(
    ['mouse', 'session', 'rewside', 'servo_pos']).size().unstack(
    ['rewside', 'servo_pos']).fillna(0).astype(np.int)

tmdf.index = tmdf.index.remove_unused_levels()


## Mask out mice
n_sessions_per_mouse = tmdf.index.to_frame().reset_index(drop=True)[
    ['mouse', 'session']].drop_duplicates().groupby(
    'mouse').size().sort_values() 


## Calculate perf for each mouse * session * rewside * stepper_pos * servo_pos
perf_by_session = tmdf.groupby(
    ['rewside', 'difficulty', 'servo_pos', 'mouse', 'session'])[
    'outcome'].value_counts().unstack('outcome')
perf_by_session['perf'] = perf_by_session['hit'] / (
    perf_by_session['hit'] + perf_by_session['error'])


## Aggregate over sessions within mouse
# Including rewside and difficulty
agg_perf_by_rewside_and_difficulty = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev != 'session'])

# Including only difficulty and meaning over rewside
agg_perf_by_difficulty = perf_by_session['perf'].mean(
    level=[lev for lev in perf_by_session.index.names 
    if lev not in ['rewside', 'session']])


## Aggregate over mice
# By rewside and difficulty
agg_by_rewside_and_difficulty_mean = agg_perf_by_rewside_and_difficulty.mean(
    level=[lev for lev in agg_perf_by_rewside_and_difficulty.index.names 
    if lev != 'mouse'])
agg_by_rewside_and_difficulty_err = agg_perf_by_rewside_and_difficulty.sem(
    level=[lev for lev in agg_perf_by_rewside_and_difficulty.index.names 
    if lev != 'mouse'])

# By difficulty only
agg_by_difficulty_mean = agg_perf_by_difficulty.mean(
    level=[lev for lev in agg_perf_by_difficulty.index.names 
    if lev != 'mouse'])
agg_by_difficulty_err = agg_perf_by_difficulty.sem(
    level=[lev for lev in agg_perf_by_difficulty.index.names 
    if lev != 'mouse'])


## Plot
PLOT_PERFORMANCE_BY_DIFFICULTY = True

if PLOT_PERFORMANCE_BY_DIFFICULTY:
    f, ax = my.plot.figure_1x1_standard()
    for difficulty in ['easy', 'hard']:
        color = 'magenta' if difficulty == 'hard' else 'k'
        linestyle = '-'

        topl = agg_by_difficulty_mean.loc[difficulty]
        topl_err = agg_by_difficulty_err.loc[difficulty]
        
        ax.plot(topl, color=color, linestyle=linestyle)

        ax.fill_between(
            x=topl.index, 
            y1=(topl - topl_err),
            y2=(topl + topl_err),
            color=color, lw=0, alpha=.2)

    my.plot.despine(ax)
    ax.set_ylim((0, 1))
    ax.set_xlim((1870, 1650))
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_xticks((1670, 1760, 1850))
    ax.set_xticklabels(('far', 'med.', 'close'))
    ax.set_xlabel('stimulus position')
    ax.set_ylabel('performance')
    ax.plot(ax.get_xlim(), [.5, .5], 'k--', lw=.8)


    ## Save
    f.savefig('PLOT_PERFORMANCE_BY_DIFFICULTY.svg')
    f.savefig('PLOT_PERFORMANCE_BY_DIFFICULTY.png', dpi=300)
    
    
    ## Stats
    n_mice = len(agg_perf_by_difficulty.index.get_level_values('mouse').unique())
    stats_filename = 'STATS__PLOT_PERFORMANCE_BY_DIFFICULTY'
    with open(stats_filename, 'w') as fi:
        fi.write(stats_filename + '\n')
        fi.write('n = {} mice\n'.format(n_mice))
        fi.write(', '.join(agg_perf_by_difficulty.index.get_level_values('mouse').unique().values) + '\n')
        fi.write('mean within session, then within mice\n')
        fi.write('error bars: sem\n'.format(n_mice))
    
    with open(stats_filename, 'r') as fi:
        lines = fi.readlines()
    print(''.join(lines))



plt.show()