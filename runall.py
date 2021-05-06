## Run all scripts

"""This must be run in the conda environment.
conda create --name Rodgers2021 --file requirements.txt
conda activate Rodgers2021
pip install -r pip_requirements.txt
"""

import subprocess
import os
import glob

this_file_dir = os.path.split(__file__)[0]
print("The name of this file is {}".format(__file__))
print("Starting in {}".format(this_file_dir))


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
        # CWD
        print("\tswitching to {}".format(dirname))
        os.chdir(dirname)

        # Get filenames to run
        pyfilenames = sorted(glob.glob('*.py'))
        print("\tpyfilenames:\n{}".format('\n'.join(
            ['\t- ' + line for line in pyfilenames])))

        for pyfilename in pyfilenames:
            # Run
            print("\t\trunning {} in {}".format(pyfilename, os.getcwd()))
            proc_output = subprocess.check_output([
                'python', pyfilename,
                ])
            
            # Decode
            proc_output = proc_output.decode('utf-8')
            
            # Disp output
            print("\t\toutput:")
            print('\n'.join([
                '\t\t- {}'.format(line) for line in proc_output.split('\n')
                ]))
            1/0

except subprocess.CalledProcessError:
    print("Encountered CalledProcessError, aborting")

finally:
    print("finally")
    print("return to {}".format(this_file_dir))
    os.chdir(this_file_dir)
