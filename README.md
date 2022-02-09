# Rodgers2021

This repository provides the code necessary to recapitulate the analyses
and figures presented in the article entitled "Sensorimotor strategies and
neuronal representations for shape discrimination". 
https://doi.org/10.1016/j.neuron.2021.05.019

A somewhat out-of-date preprint version of the article is available at bioRxiv
https://www.biorxiv.org/content/10.1101/2020.06.16.126631v1.

# Downloading the dataset

The data will be available at the following DOI: 10.5281/zenodo.4743837.
https://zenodo.org/record/4743837#.YgQ7xYzMIuU

The dataset on Zenodo consists of an archive file called `ShapeDataset.zip`.
Download that file and unzip it. Make note of the location. The rest of
this document assumes that the unzipped data resides at `~/data/ShapeDataset/`.

For a full description of the contents of this data archive, see the file
`DATA_README.md` in this repository.

# Installation

## Set up the `conda` environment

While the data is downloading, you can start setting up an environment
for analyzing it. Linux is the only OS on which this code has been tested.

If you have any problems, please create a Github Issue on this repository.
Please note that I cannot promise to provide support, though I can try to
answer questions as they come up. In principle, this code should run in a
properly configured conda environment on any operating system; but in practice,
I don't know anything about Windows or OSX, so if you have problems on
those operating systems, it is very unlikely that I will be able to help.

Most of the data is stored as pickled `pandas.DataFrame` files. __It is
very unlikely you will be able to open these files unless you use exactly
the same version of `pandas` as I did to create them.__ The following steps
show you how to set up an environment with the correct version of all
requirements.

First, clone this repository into your home directory. (You can easily
modify these commands if you wish to clone it somewhere else.)

```
cd ~
git clone --recursive https://github.com/cxrodgers/Rodgers2021
```

The flag "--recursive" here trigger git to also fetch the required
submodules, which are just some of my other github repositories that this
repository depends on. Those will go into the subdirectory called `submodules`.

Set up a conda environment called `Rodgers2021`. This will install all
the necessary packages (listed in `requirements.txt` and
`pip_requirements.txt`) into that environment.

```
cd ~/Rodgers2021
conda create --name Rodgers2021 --file requirements.txt
conda activate Rodgers2021
pip install -r pip_requirements.txt
```

You must also set some environment variables. First, set `$MPLCONFIGDIR`
so that the "agg" background is always used. (Otherwise, these files hang
until the plot is closed.)

```
conda deactivate
conda activate Rodgers2021
conda env config vars set MPLCONFIGDIR=$HOME/Rodgers2021/mpl_config_dir
```

Second, set the `$PYTHONPATH` to include the path to the submodules.
```
conda deactivate
conda activate Rodgers2021
conda env config vars set PYTHONPATH=$HOME/Rodgers2021/submodules
```

Reactivate the environment once again to trigger these changes.
```
conda deactivate
conda activate Rodgers2021
```

## Set up the `parameters` file

All of the scripts use a JSON file called `~/Rodgers2021/parameters` to find the
data. Open up that file with a text editor.

Near the bottom of the file will be a bunch of lines that contain the
string "ShapeDataset". These paths must be fixed to go to the location
of the downloaded data.

```
    "pipeline_input_dir": "/home/jack/data/ShapeDataset",
    "root_session_dir": "/home/jack/data/ShapeDataset/sessions",
    "unit_db_dir": "/home/jack/data/ShapeDataset/neural_metadata",
    "trainset_dir": "/home/jack/data/ShapeDataset/trainset",
    "glm_fits_dir": "/home/jack/data/ShapeDataset/glm_fits",
    "glm_example_dir": "/home/jack/data/ShapeDataset/glm_example",
    "example_frames_dir": "/home/jack/data/ShapeDataset/example_frames"
```

In this case I have them pointing to `/home/jack/data/ShapeDataset`. Change
that to the actual location of the unzipped data on your computer.

Running this code creates many large intermediate data files. Choose a
location for those intermediate files. Let's say it's
`~/data/ShapeDataset_output`. These lines at the top of the file must
be set to that path.

```
    "pipeline_output_dir": "/home/jack/data/ShapeDataset_output",
    "patterns_dir": "/home/jack/data/ShapeDataset_output/patterns",
    "logreg_dir": "/home/jack/data/ShapeDataset_output/logreg",
    "neural_dir": "/home/jack/data/ShapeDataset_output/neural",
    "glm_dir": "/home/jack/data/ShapeDataset_output/glm",
    "scaling_dir": "/home/jack/data/ShapeDataset_output/scaling",
```

Unless your username is also "jack", the default paths will not be correct
on your computer, so you must make these changes.

## Run the code

Once you have completed the steps above, you can run the analyses. All
of the analyses exist in numbered directories:

```
01_patterns/
03_logreg/
04_logreg_vis/
05_behavior_vis/
06_neural/
07a_glm/
08_neural_vis/
09_neural_decoding/
```

Within each of these numbered directory is a set of Python scripts.
```
01_patterns/main1b.py
01_patterns/main1c.py
01_patterns/main2a.py
...
```

Each script must be run in alphabetical order, because the later ones
might depend on the earlier ones.

If you're feeling lucky, you can use the script `runall.py` to run all
of the scripts in sequence. If anything goes wrong (most likely a missing
import or an incorrect path to the data), it will crash in place.

```
conda activate Rodgers2021
cd ~/Rodgers2021
python runall.py
```

The results will be stored as figures in the numbered subdirectories.

[This CSV file](figure_panels.csv) lists the subdirectory and script name that generates each
panel in the manuscript, as well as the image filename of the figure.
