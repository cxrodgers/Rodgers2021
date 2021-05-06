## This directory dumps contacts and whisking data for decoding
# This file generates session_df, a DataFrame of metadata about sessions
# TODO: Replace this file with hardcoded session_df

import json
import os
import numpy as np
import whiskvid
import matplotlib.pyplot as plt
import pandas
import MCwatch.behavior
import runner.models
import tqdm


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Create pipeline and patterns dir
if not os.path.exists(params['pipeline_output_dir']):
    os.mkdir(params['pipeline_output_dir'])
if not os.path.exists(params['patterns_dir']):
    os.mkdir(params['patterns_dir'])
    
    
## Iterate over sessions
sliced_angle_by_whisker_df_l = []
sliced_phases_df_l = []
subtm_l = []
keys_l = []

rec_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Get session data
    gs = runner.models.GrandSession.objects.filter(name=session_name).first()
    vs = whiskvid.django_db.VideoSession.from_name(session_name)

    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = MCwatch.behavior.db.get_trial_matrix(vs.bsession_name)

    # Identify servo spacing
    servo_pos = np.sort(trial_matrix['servo_pos'].unique())
    if (servo_pos == np.array([1670, 1760, 1850])).all():
        servo_spacing = 'normal'
    else:
        servo_spacing = 'abnormal'
        1/0
    
    # Identify task
    if gs.session.python_param_stimulus_set == 'trial_types_2shapes_CCL_3srvpos':
        task = 'discrimination'
        is_2shapes = True
    elif gs.session.python_param_stimulus_set == 'trial_types_CCL_3srvpos':
        task = 'discrimination'
        is_2shapes = False
    elif gs.session.python_param_stimulus_set == 'trial_types_detect_FR75_3srvpos':
        task = 'detection'
        is_2shapes = False
    else:
        1/0


    ## Identify whether opto
    try:
        opto_session = vs._django_object.grand_session.optosession
        has_opto_session = True
    except runner.models.OptoSession.DoesNotExist:
        has_opto_session = False
    
    if has_opto_session:
        is_sham = opto_session.sham
        target = opto_session.target
        power = opto_session.start_power
    else:
        is_sham = False
        target = None
        power = None

    
    ## Store
    rec_l.append({
        'session': session_name,
        'opto': has_opto_session,
        'sham': is_sham,
        'task': task,
        'twoshapes': is_2shapes,
        'mouse': vs._django_object.grand_session.session.mouse.name,
    })
    
session_df = pandas.DataFrame.from_records(rec_l).set_index('session')
session_df.to_pickle(os.path.join(params['pipeline_dir'], 'session_df'))
