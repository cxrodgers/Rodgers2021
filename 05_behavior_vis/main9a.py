## Behavioral stats
# Number, sex, genotypes etc of mice
# Number of training sessions
# Note: the anatomy image is from 10170-1, a female Emx-GFP, not included here

import json
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import MCwatch.behavior
import runner.models
import my

my.plot.manuscript_defaults()
my.plot.font_embed()


## Trims
trims_table = MCwatch.behavior.db.get_whisker_trims_table()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df = pandas.read_pickle(
    os.path.join(params['pipeline_dir'], 'session_df'))
task2mouse = session_df.groupby('task')['mouse'].unique()
mouse2task = session_df[
    ['task', 'mouse']].drop_duplicates().set_index('mouse')['task']

mouse_name_l = sorted(session_df['mouse'].unique())

# Append the flatter mouse
mouse_name_l.append('KM100')

# Append the whisker trim mice from main8a
whisker_trim_mice = [
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
for mouse_name in whisker_trim_mice:
    # They should all already be included
    assert mouse_name in mouse_name_l

# Append the single whisker mice (KM100 already included, but KM102 is not)
single_whisker_mice = ['KM100', 'KM102']
for mouse_name in single_whisker_mice:
    if mouse_name not in mouse_name_l:
        mouse_name_l.append(mouse_name)

# Append the gradual trim mice
gradual_trim_mice = ['219CR', '221CR', '229CR', '231CR', 'KF119', 'KF132', 'KF134', 'KM101', 'KM131']
for mouse_name in gradual_trim_mice:
    # They should all already be included
    assert mouse_name in mouse_name_l

# Append the lesion mice (none included above)
lesion_mice = ['KM129', 'KF133', 'KM136', 'KM147', 'KM148', '242CR', '230CR', '243CR']
for mouse_name in lesion_mice:
    if mouse_name not in mouse_name_l:
        mouse_name_l.append(mouse_name)

# Sort
mouse_name_l = sorted(mouse_name_l)


## Which mice to skip for session number quantification
# The second group of mice listed here are missing data about licktrain
skip_SNQ = lesion_mice + ['KM91', 'KM101', 'KM100', 'KM102']


## Fields to extract from session object
field2name = {
    'python_param_scheduler_name': 'scheduler',
    'python_param_stimulus_set': 'stim_set',
    'board': 'board',
    'box__name': 'box',
    'date_time_start': 'dt_start',
    'grand_session__name': 'gs',
    }

## Find each mouse in the db
rec_l = []
training_session_df_l = []
training_session_df_keys_l = []
for mouse_name in mouse_name_l:
    ## Get metadata about mice
    # Find mouse 
    mouse_qs = runner.models.Mouse.objects.filter(name=mouse_name)
    assert len(mouse_qs) == 1
    mouse_db = mouse_qs.first()
    
    # Extract vitals
    genotype = mouse_db.genotype
    dob = mouse_db.dob
    sex = dict(mouse_db.SEX_CHOICES)[mouse_db.sex]
    
    
    ## Only quantify session number for some mice
    if mouse_name not in skip_SNQ:
        ## Get sessions for mice
        # Sessions
        session_qs = runner.models.Session.objects.filter(
            mouse__name=mouse_name).order_by('date_time_start')
        
        # Count by Scheduler
        training_session_df = pandas.DataFrame.from_records( 
            list(session_qs.values_list(*field2name.keys())),
            columns=field2name.values()).sort_values('dt_start')
        
        # Check that it was already sorted
        assert (training_session_df.index == np.arange(len(training_session_df))).all()
        

        ## Split into epochs
        # Get the last licktrain session
        last_licktrain_session = (
            np.where(training_session_df['scheduler'] == 'ForcedAlternationLickTrain')
            [0][-1])
        
        # First Auto
        first_auto_session = (
            np.where(training_session_df['scheduler'] == 'Auto')
            [0][0])
        
        # Last session before transfer
        # TODO: handle case where they got switched back to original box later
        last_session_before_transfer = (
            np.where(training_session_df['box'] != 'CR0')
            [0][-1])
        
        
        ## Assign stage
        # Default to blank
        training_session_df['stage'] = ''

        # Assign licktrain
        if last_licktrain_session is not None:
            assert (
                training_session_df['scheduler'].iloc[:last_licktrain_session + 1] == 
                'ForcedAlternationLickTrain').all()
            training_session_df.loc[:last_licktrain_session + 1, 'stage'] = 'licktrain'

        # Assign initFA
        if last_licktrain_session is not None:
            assert (
                training_session_df['scheduler'].iloc[last_licktrain_session + 1:first_auto_session] == 
                'ForcedAlternation').all()
            training_session_df.loc[last_licktrain_session + 1:first_auto_session, 'stage'] = 'initFA'    
        else:
            assert (
                training_session_df['scheduler'].iloc[:first_auto_session] == 
                'ForcedAlternation').all()
            training_session_df.loc[:first_auto_session, 'stage'] = 'initFA'            
        
        # Assign training
        training_session_df.loc[first_auto_session:last_session_before_transfer + 1, 'stage'] = 'training'        

        # Assign rig0
        assert (
            training_session_df['box'].iloc[last_session_before_transfer + 1:] == 
            'CR0').all()
        training_session_df.loc[last_session_before_transfer + 1:, 'stage'] = 'rig0'    
        
        
        ## Error check
        assert not (training_session_df['stage'] == '').any()
        
        
        ## Store training session
        training_session_df_l.append(training_session_df)
        training_session_df_keys_l.append(mouse_name)
    
    
    ## Store
    start_date = training_session_df['dt_start'].iloc[0].date()
    start_age = (start_date - dob).days
    rec_l.append({
        'mouse': mouse_name,
        'genotype': genotype,
        'sex': sex,
        'dob': dob,
        'start_date': start_date,
        'start_age': start_age,
        })
    

## Concat over mice
resdf = pandas.DataFrame.from_records(rec_l).set_index('mouse')
big_sdf = pandas.concat(training_session_df_l, keys=training_session_df_keys_l, names=['mouse'])


## Join on task
resdf = resdf.join(mouse2task)

# Manually adjust some
# This one was used for flatter shapes but not for video analysis
resdf.loc['KM100', 'task'] = 'discrim_novideo'

# This one was used for single-whisker trim but not for video analysis
resdf.loc['KM102', 'task'] = 'single_whisker'

# Lesion mice
resdf.loc[lesion_mice, 'task'] = 'lesion'

# Extract new_mouse2task
new_mouse2task = resdf['task'].copy()

# Reorder the levels
resdf = resdf.set_index('task', append=True).swaplevel().sort_index()

# Count mice by task and sex
counted_mice = resdf.groupby(['task', 'sex']).size().unstack(
    'sex').fillna(0).astype(np.int)
counted_mice.loc['total'] = counted_mice.sum()


## Count training sessions
n_stages = big_sdf.groupby(
    ['mouse', 'stage']).size().unstack('stage').fillna(0).astype(np.int)

# Join on new_mouse2task
n_stages = n_stages.join(new_mouse2task)

# Reorder levels
n_stages = n_stages.set_index('task', append=True).swaplevel().sort_index()
n_stages = n_stages.loc[:, ['licktrain', 'initFA', 'training', 'rig0']]


## Dump stats
with open('STATS__MOUSE_HUSBANDRY', 'w') as fi:
    fi.write('Included mice\n')
    fi.write('------------\n')
    fi.write(resdf.to_string() + '\n')
    fi.write('\n\n')
    
    fi.write('Counted by task and type\n')
    fi.write('------------------------\n')
    fi.write(counted_mice.to_string() + '\n')
    fi.write('\n\n')
    
    fi.write('Mean age at start\n')
    fi.write('-----------------\n')
    fi.write(resdf['start_age'].mean(level='task').to_string())
    fi.write('\nboth tasks\n')
    fi.write(str(resdf['start_age'].mean()))
    fi.write('\n\n\n')
    
    fi.write('Training sessions required\n')
    fi.write('--------------------------\n')
    fi.write(n_stages.to_string())
    fi.write('\n\n\n')
    
    fi.write('Mean training sessions required (where data available)\n')
    fi.write('------------------------------------------------------\n')
    fi.write(n_stages.mean(level='task').to_string())
    fi.write('\nboth tasks\n')
    fi.write(n_stages.mean().to_frame().T.to_string())
    fi.write('\n\n\n')
    
with open('STATS__MOUSE_HUSBANDRY', 'r') as fi:
    lines = fi.readlines()
print(''.join(lines))
