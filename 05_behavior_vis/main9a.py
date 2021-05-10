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
import my
import my.plot


## Plotting params
my.plot.manuscript_defaults()
my.plot.font_embed()


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Get data
training_time_dir = os.path.join(params['pipeline_input_dir'], 'training_time')
resdf = pandas.read_pickle(os.path.join(training_time_dir, 'resdf'))
big_sdf = pandas.read_pickle(os.path.join(training_time_dir, 'big_sdf'))


## Process resdf
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
