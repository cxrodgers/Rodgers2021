## Run all scripts

"""
# First create a conda environment
conda create --name Rodgers2021 --file requirements.txt
conda activate Rodgers2021
pip install -r pip_requirements.txt

# Then set this config var to ensure agg is always used
conda env config vars set MPLCONFIGDIR=/home/jack/dev/Rodgers2021/mpl_config_dir

# Then reactivate
conda deactivate
conda activate Rodgers2021
"""

import subprocess
import os
import glob

this_filename = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_filename)
print("The name of this file is {}".format(this_filename))
print("Starting in {}".format(this_file_dir))

start_at = '06_neural', 'main0b2.py'
skip_start_flag = False

## Get dirnames to run
dirnames = [
    '01_patterns',
    '03_logreg',
    '04_logreg_vis',
    '05_behavior_vis',
    '06_neural',
    '07a_glm',
    '08_neural_vis',
    '09_neural_decoding',
    ]


try:
    for dirname in dirnames:
        if skip_start_flag:
            if dirname != start_at[0]:
                continue
        
        # CWD
        print("\tswitching to {}".format(dirname))
        os.chdir(dirname)

        # Get filenames to run
        pyfilenames = sorted(glob.glob('*.py'))
        pyfilenames = [fn for fn in pyfilenames if fn != 'extras.py']
        print("\tpyfilenames:\n{}".format('\n'.join(
            ['\t- ' + line for line in pyfilenames])))

        for pyfilename in pyfilenames:
            if skip_start_flag:
                if pyfilename != start_at[1]:
                    continue            
                else:
                    skip_start_flag = False
                    
            # Run
            print("\t\trunning {} in {}".format(pyfilename, os.getcwd()))
            proc_output = subprocess.check_output([
                'python', 
                '-W', 'ignore:Matplotlib is currently using agg:UserWarning::', 
                pyfilename,
                ])
            
            # Decode
            proc_output = proc_output.decode('utf-8')
            
            # Disp output
            print("\t\toutput:")
            print('\n'.join([
                '\t\t- {}'.format(line) for line in proc_output.split('\n')
                ]))

        # CWD
        print("\tswitching to {}".format(this_file_dir))
        os.chdir(this_file_dir)


except subprocess.CalledProcessError:
    print("Encountered CalledProcessError, aborting")

finally:
    print("finally")
    print("return to {}".format(this_file_dir))
    os.chdir(this_file_dir)

    os.environ['MPLCONFIGDIR'] = ''    