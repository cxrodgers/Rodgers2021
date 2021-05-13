# Introduction

This README explains how to use the dataset of behavioral and neural results
collected during the shape discrimination and detection tasks.

You will likely want to start by downloading the following items:
1. The dataset, hosted on Zenodo at the following DOI: XXX
2. The code used to process the data in a forthcoming paper. This code is
   hosted as a repository on Github at
   https://github.com/cxrodgers/Rodgers2021

This document explains each component file of the dataset, how to open it,
and how to interpret it. For documentation on the code used to process the
data, see the README for the github repository.

## Structure of the dataset

The dataset is available as a single archive, a zip file. Within this archive
is a single directory called `ShapeDataset/`.

This directory contains the following directories and files. Each is
described briefly here, and in detail below. Directory names end with a `/`.
The most important files are indicated with __bold text__.

| File | Description |
| --- | --- |
| `example_frames/` | Example images from the behavioral video |
| `flatter_shapes/` | Behavioral data on the more difficult shapes |
| `glm_example/` | Example GLM fits from the neural encoding analysis |
| `glm_fits/` | Coefficients and fitting results from each GLM |
| `flatter_shapes/` | Behavioral data on the more difficult shapes |
| `gradual_trims/` | Behavioral data on gradually trimming off rows |
| `lesion/` | Behavioral data after cortical lesion |
| `neural_metadata/` | __Metadata about each recorded neuron__ |
| `sessions/` | __A large amount of various behavioral and neural data about each session__ |
| `single_whisker/` | Behavioral data on trimming to a single whisker |
| `training_time/` | Metadata about mice and their training times |
| `trainset/` | Quantifications of whisker tracking accuracy |
| `whisker_trim/` | Behavioral data on trimming all whiskers |
| `neural_session_df` | __Metadata about the sessions with neural data__ |
| `session_df` | __Metadata about every session__ |

The most important files are `session_df`, `neural_session_df`,
`neural_metadata/`, and `sessions/`. The other files here are mainly various
control or follow-up experiments and analyses.

The vast majority of the data is contained within the `sessions/`
directory, in sub-directories named by each session. Each session's name is
formatted like `161215_KM91`, in which the first six digits indicate the date
of the session (2016-12-15), and the characters after the `_` indicate
the mouse's name (KM91).

The following files are contained with `sessions/SESSION_NAME/`, where
the token `SESSION_NAME` indicates the name of each session. The files
`spikes` and `neural_trial_timings` only exist for sessions with neural data,
i.e., those sessions that are contained in the file `neural_session_df`.
The most important files are shown in __bold text__.

| File | Description |
| --- | --- |
| `all_edges.npy` | Location of the shape's edge on every frame |
| `behavioral_logfile` | Plain text log of all behavioral events |
| `colorized_contacts_summary` | __Metadata about every whisker contact__ |
| `colorized_whisker_ends` | __Location of every whisker on every frame__ |
| `cwe_with_kappa` | Like colorized_whisker_ends, but also including whisker bending |
| `edge_summary` | Location of the final position of the shapes at each position |
| `frontier_all_stim` | Location of the closest possible position of the shapes |
| `joints` | Location of each of 8 equally spaced "joints" along each whisker on each frame |
| `neural_trial_timings` | __Metadata about each trial for synchronizing neural and behavioral data__ |
| `spikes` | __The times of each spike from each recorded neuron__ |
| `trial_matrix` | __Behavioral metadata about each trial__ |

## File formats

Each file described below is one of the following formats. The table provides
a brief description and a suggested Python function for reading each format.

| Format | Description | Read with |
| --- | --- | --- |
| pickled `pandas.DataFrame` | A pickled `DataFrame` from the `pandas` module | `pandas.read_pickle` |
| plain text | Plain text | `open` or similar |
| HDF5 | A table stored in HDF5 format | `pandas.read_table` |
| pickle | A pickled Python object | `pickle.load` |
| npy | A numpy array | `numpy.load` |
| image | An image in PNG format | `imageio.imread` |

Note that the npy files were stored with Python 2. To load them in Python 3,
use something like this:

* `numpy.load(filename, allow_pickle=True, encoding='latin1')`

## Timebases

TODO: Document the _neural, behavioral, and video timebases_.

## Optogenetics

TODO: Document the use of optogenetics.

## Raw versus intermediate data

This dataset provides raw data where possible, feasible, and useful. The
files described here correspond to those "raw data". The code provided in
the repository may be used to generate many "intermediate data files", and
ultimately the exact figures presented in the manuscript.

These intermediate data files are likely to be more useful than the raw
data files. For instance, the intermediate files remove unnecessary trials
from the raw data, and synchronize behavioral, video, and neural data.

The manuscript relies in two particularly time-intensive analyses: whisker
tracking from raw video, and fitting of GLMs to identify neural tuning
properties. Running these analyses requires days or weeks of computation,
even with multiple GPUs and hundreds or thousands of CPU cores. Therefore,
for those two analyses, the "raw data" provided here is actually the
result of those intensive computations.

# Detailed description of each file

This section describes the format and content of each individual file
in the dataset.

## `example_frames/SESSION_NAME_FRAME_NUMBER.png`

### Format

image

### Short description

This directory contains three images, extracted from the raw video,
which are used as example images in the manuscript.

* `180119_KM131_137393.png`
* `180221_KF132_242546.png`
* `180221_KF132_490102.png`

## `flatter_shapes/tmdf`

### Format

pickled `pandas.DataFrame`

### Short description

This contains behavioral metadata about sessions using the flatter,
more difficult shapes.

### Detailed description

This is similar to `trial_matrix`, but only for the sessions with flatter
shapes, and concatenated over all such sessions. An additional column
codes the difficulty of the shape.

Some of these sessions are also included in `sessions/`, and some correspond
to sessions that are used only for this analysis and nothing else.

#### Example

```
                          start_time  ...  difficulty
mouse session      trial              ...            
KF119 180108_KF119 45        432.257  ...        easy
                   46        440.124  ...        easy
                   47        448.442  ...        easy
                   48        463.942  ...        hard
...                              ...  ...         ...
KM91  170126_KM91  366      3983.109  ...        easy
                   367      3991.105  ...        easy
                   368      3999.106  ...        easy
                   369      4016.106  ...        hard

[3545 rows x 12 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `mouse` : the mouse's name
* `session` : the session name
* `trial` : the trial number

#### Columns
* `difficulty` : one of ``{'easy', 'hard'}``, where `'easy'` indicates the
  typical shape difficulty, and `'hard'` indicates the more difficult,
  flatter shapes. Note that this information may also be obtained from the
  `stepper_pos` column.

All other columns are documented under `sessions/SESSION_NAME/trial_matrix`.

## `glm_example/SESSION_NAME-NEURON`

### Format

pickled `dict`

### Short description

The results of the GLM fitting procedure for two example neurons.

### Detailed description

These detailed GLM fitting results are used only for illustrating an
example fit. The other GLM analyses rely on the summarized data contained
in `glm_fits/`.

### Items

This `dict` contains the following (key, value) pairs.

* `neuron` : the number of the single unit (neuron)
* `session` : the name of the session
* `fitting_results` : A `pandas.DataFrame` with the full results of the
  fitting procedure.
* `coef_df` : A `pandas.DataFrame` with the coefficients obtained.
* `input_mean` : A `pandas.Series` with the mean of each input feature.
* `input_scale` : A `pandas.Series` with the scale (standard deviation) of
  each input feature.

## `glm_fits/MODEL_NAME/coef_wscale_df`

### Format

pickled `pandas.DataFrame`

### Short description

The coefficients obtained by fitting the GLM named `MODEL_NAME` to all
of the neurons.

### Detailed description

Each model contains a different set of input features, but they are all fit
to the same neural firing rates.

The model named `'minimal'` is the one identified in the text as the "best",
by the process of model selection. All other models are used primarily as
controls or points of comparison for the `'minimal'` model.

### Example
```
coef_metric                                                                 coef_best  ...  coef_single_p
task           session      neuron metric                            label             ...               
detection      190716_228CR 3      contact_binarized                 C1      0.009168  ...   5.305718e-01
                                                                     C2      0.009901  ...   6.096449e-01
                                   drift                             drift   0.106485  ...   2.012746e-06
                                   log_cycle_duration                all     0.049059  ...   2.069254e-02
...                                                                               ...  ...            ...
discrimination 180122_KM131 926    whisking_indiv_set_point_smoothed C0     -0.089461  ...   4.238757e-06
                                                                     C1     -0.007743  ...   7.942368e-01
                                                                     C2      0.162901  ...   0.000000e+00
                                                                     C3      0.157536  ...   3.552714e-15

[38231 rows x 10 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `task` : one of `{'detection', 'discrimination'}`; the task performed
  by the mouse.
* `session` : the session name
* `neuron` : the id of the single unit (neuron).
* `metric` : the family of features, such as `contact_binarized`
* `label` : the specific member of that family of features, such as `C2`.

The levels `session` and `neuron` can be used to index `big_waveform_info_df`
to get the neural metadata, such as cell type and location.

#### Columns

* `coef_best` : The coefficient obtained for the best regularization parameter,
  averaged over folds.
* `coef_single` : The coefficient obtained for the default regularization
  parameter, averaged over folds.
* `scale` : The scale (standard deviation) of the feature. The GLM was fit
  to the standardized features, which are the raw features minus their mean
  and then divided by this scale.
* `mean` : The mean of the feature
* `scaled_coef_best` : The coefficient `coef_best` divided by `scale`. This
  "scaled coefficient" can be multiplied directly by the raw feature values,
  instead of by the standardized feature values.
* `scaled_coef_single` : The coefficient `coef_single` divided by `scale`.
* `perm_single_std` : The standard deviation of the coefficients obtained
  by fitting to permuted data, using the default regularization parameter.
* `perm_single_mean` : The mean of the coefficients obtained
  by fitting to permuted data, using the default regularization parameter.
* `coef_single_z` : The "z-score" of `coef_single` with respect to the
  permuted distribution, obtained by subtracting `perm_single_mean` and then
  dividing by `perm_single_std`.
* `coef_single_p` : The two-tailed p-value corresponding to `coef_single_z`,
  obtained by summing the area under the tails beyond this z-score in a
  standard normal distribution.

## `glm_fits/MODEL_NAME/fitting_results_df`

### Format

pickled `pandas.DataFrame`

### Short description

The goodness-of-fit of the GLM named `MODEL_NAME` for all of the neurons.

### Detailed description

This is similar to `glm_fits/MODEL_NAME/coef_wscale_df`, but instead of
the coefficients, this file contains information about the goodness-of-fit.

### Example

```
task           session      neuron analysis        n_fold n_reg_lambda n_permute                    ...             
detection      190716_228CR 3      actual          0      0            0                 17.913815  ...     0.000055
                                                          1            0                 17.914700  ...     0.000179
                                                          2            0                 17.917540  ...     0.000568
                                                          3            0                 17.926914  ...     0.001781
...                                                                                            ...  ...          ...
discrimination 180122_KM131 926    permuted_single 4      8            36               219.767000  ...     0.006842
                                                                       37               216.768437  ...     0.007874
                                                                       38               287.863576  ...     0.003900
                                                                       39               205.727405  ...     0.003914

[274725 rows x 18 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `task` : one of `{'detection', 'discrimination'}`; the task performed
  by the mouse.
* `session` : the session name
* `neuron` : the id of the single unit (neuron).
* `analysis` : one of `{'actual', 'actual_best', 'actual_single', 'permuted_single'}`.
  `actual` means a fit to the actual data, over all regularization parameters.
  `actual_best` is the actual fit, for the best regularization parameter.
  `actual_single` is the actual fit, for the default regularization parameter.
  `permuted_single` is the fit to the permuted data, using the default
  regularization parameter.
* `n_fold` : the index of the cross-validation fold.
* `n_reg_lambda` : the index of the regularization parameter.
* `n_permute` : the permutation number, or 0 for the actual data.

The levels `session` and `neuron` can be used to index `big_waveform_info_df`
to get the neural metadata, such as cell type and location.

#### Columns

* `contact_deviance` : the deviance on the test set during whisks with contact
* `contact_likelihood` : the likelihood of the test set under the fit
  model during whisks with contact
* `contact_null_likelihood` : the likelihood of the test set under the null
  model during whisks with contact
* `contact_oracle_likelihood` : the likelihood of the test set under the oracle
  model during whisks with contact
* `contact_score` : the pseudo R2 on the test set under the fit
  model during whisks with contact
* `converged` : whether the GLM fit converged
* `deviance` : the deviance on the test set
* `icpt` : the intercept of the fit model
* `len_ytest` : the number of whisks in the test set
* `len_ytest_contacts` : the number of whisks with contact in the test set
* `likelihood` : the likelihood of the test set under the fit model
* `niter` : the number of iterations required
* `null_likelihood` : the likelihood of the test set under the null model
* `oracle_likelihood` : the likelihood of the test set under the oracle model
* `permuted` : whether the features were permuted before fitting
* `reg_lambda` : the regularization parameter used
* `score` : the pseudo R2 on the test set under the fit model
* `train_score` : the pseudo R2 on the training set under the fit model

## `gradual_trims/session_table`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about the sessions during which the whiskers were gradually
trimmed off, row by row.

### Detailed description

#### Example

```
                                 stimulus_set scheduler  ...   trim n_rows
session                                                  ...              
20181120144917.219CR  trial_types_CCL_3srvpos      Auto  ...    all      5
20181121182338.219CR  trial_types_CCL_3srvpos      Auto  ...    all      5
20181126125224.219CR  trial_types_CCL_3srvpos      Auto  ...    all      5
20181127152520.219CR  trial_types_CCL_3srvpos      Auto  ...    all      5
...                                       ...       ...  ...    ...    ...
20180117194144.KM131  trial_types_CCL_3srvpos      Auto  ...  C*; b      1
20180118170958.KM131  trial_types_CCL_3srvpos      Auto  ...  C*; b      1
20180119170743.KM131  trial_types_CCL_3srvpos      Auto  ...  C*; b      1
20180122193908.KM131  trial_types_CCL_3srvpos      Auto  ...  C*; b      1

[639 rows x 7 columns]

```

#### Index

The name of the _behavioral session_. Note that this is a different format
than the name of the sessions in `sessions/`. The reason is that there is
no video available for these sessions.

#### Columns

* `stimulus_set` : the name of the stimulus set used
* `scheduler` : the name of the scheduler used to deliver the trials
* `mouse` : the mouse's name
* `board`, `box` : ignore
* `trim` : which whiskers were spared
* `n_rows` : the number of whisker rows remaining

## `gradual_trims/tmdf`

### Format

pickled `pandas.DataFrame`

### Short description

Behavioral metadata about the trials during gradual trim sessions.

### Detailed description

#### Example

```
                                  start_time  ...  stepper_pos
mouse session              trial              ...             
219CR 20181120144917.219CR 45        448.484  ...          150
                           46        463.483  ...           50
                           47        470.480  ...          150
                           48        477.225  ...           50
...                                      ...  ...          ...
KM131 20180122193908.KM131 461      4482.895  ...          150
                           462      4491.282  ...          150
                           463      4498.950  ...           50
                           465      4523.341  ...           50

[113398 rows x 11 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels.
* `mouse` : the mouse's name
* `session` : the name of the _behavioral session_, as in
  `gradual_trims/session_table`.
* `trial` : the trial number

#### Columns

The columns are defined in `sessions/SESSION_NAME/trial_matrix`.

## `lesion/big_peri`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about the sessions before and after barrel cortex lesion.

### Detailed description

#### Example

```
                                   session  perf_unforced             stimulus_set
mouse epoch    n_day                                                              
230CR baseline 0      20200120131650.230CR       0.779343  trial_types_CCL_2srvpos
               1      20200121113424.230CR       0.768595  trial_types_CCL_2srvpos
               2      20200127094724.230CR       0.757576  trial_types_CCL_2srvpos
      contra   3      20200128103954.230CR       0.567251  trial_types_CCL_2srvpos
...                                    ...            ...                      ...
KM148 contra   6      20180219133307.KM148       0.505263  trial_types_CCL_3srvpos
               7      20180220173452.KM148       0.639175  trial_types_CCL_3srvpos
               8      20180221161637.KM148       0.683761  trial_types_CCL_3srvpos
      nwt      9      20180222174537.KM148       0.666667  trial_types_CCL_3srvpos

[85 rows x 3 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels:

* `mouse` : the mouse's name
* `epoch` : one of `{'baseline', 'contra', 'ipsi'}`, indicating whether the
  session was pre-lesion, post-contra lesion, or post-ipsi lesion.
* `n_day` : the number of the session, starting three days before lesion.

#### Columns

* `session` : the name of the _behavioral session_
* `perf_unforced` : the performance
* `stimulus_set` : the stimulus set used

## `neural_metadata/big_waveform_df`

### Format

pickled `pandas.DataFrame`

### Short description

The average waveform of each recorded neuron.

### Detailed description

#### Example
```
                      0    1   ...         80         81
session      neuron            ...                      
180116_KM131 25      0.0  0.0  ...   7.510362   7.233901
             27      0.0  0.0  ...   1.051386   0.450699
             44      0.0  0.0  ...  12.955654  12.003397
             54      0.0  0.0  ...   3.220749   1.944383
...                  ...  ...  ...        ...        ...
200401_267CR 217     0.0  0.0  ...   2.827376   2.206264
             220     0.0  0.0  ...   6.564794   5.351141
             381     0.0  0.0  ...   9.014249   7.663073
             382     0.0  0.0  ...  13.713473  11.686242

[999 rows x 82 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels:

* `session` : the session name
* `neuron` : the index of the single unit (number)

#### Columns

Indexes each of 82 samples at 30 kHz in the waveform

## `neural_metadata/big_waveform_info_df`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about every recorded neuron.

### Detailed description

#### Example
```
                     width  ...  location_is_strict
session      neuron         ...                    
180116_KM131 25         13  ...               False
             27          8  ...               False
             44         14  ...               False
             54         12  ...               False
...                    ...  ...                 ...
200401_267CR 217        11  ...                True
             220        12  ...                True
             381        14  ...                True
             382        14  ...                True

[999 rows x 11 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels:

* `session` : the session name
* `neuron` : the index of the single unit (number)

#### Columns

* `'width'` : half-width of the waveform, in samples
* `'peak_idx'` : ignore
* `'big_ichannel'` : ignore
* `'NS'` : bool for "narrow-spiking", True if narrow and putative inhibitory
* `'Z_corrected'` : depth in microns within the cortex
* `'Srt'` : ignore
* `'layer'` : cortical layer
* `'stratum'` : one of ``{'superficial', 'deep'}``, taken from `layer`
* `'recording_location'` : cortical column in which the neuron was recorded
* `'crow_recording_location'` : closest C-row column
* '`location_is_strict'`: True if ``'recording_location' == 'crow_recording_location'``

## `sessions/SESSION_NAME/`

This directory contains 115 subdirectories, one for each session. The
token `SESSION_NAME` is used here to indicate the name of each subdirectory,
which is also the session name.

### Format

directory

### Short description

This subdirectory contains all the data relating to the session named
`SESSION_NAME`.

## `sessions/SESSION_NAME/all_edges.npy`

### Format
numpy array

### Short description

This is a numpy array of the location of the edge of the shape on each frame
in the behavioral video.

### Detailed description

* The array has shape `(N, 2)`, where `N` is the number of frames in the video.
* Each entry within the array is a location `(r, c)` of one pixel of the edge
  of the shape.

## `sessions/SESSION_NAME/behavioral_logfile`

### Format
plain text

### Short description

This is a plain text logfile of the behavioral events in that session.

### Detailed description

* This file contains a series of lines issued by the Arduino during the behavior.
* The first part of each line is the time (in milliseconds) since the beginning
  of the session. These are the _behavioral times_, as opposed to _video times_
  or _neural times_.
* For the interpretation of each line, see the ArduFSM repository:
  https://github.com/cxrodgers/ArduFSM

## `sessions/SESSION_NAME/colorized_contacts_summary`

### Format
pickled `pandas.DataFrame`

### Short description
This describes parameters of every whisker contact.

### Detailed description

#### Example

```
        whisker       tip_x  ...  angle_range  duration
cluster                      ...                       
1            C0  137.748608  ...     0.000000         1
2            C0  142.128101  ...     0.297169         2
3            C1  216.211622  ...     0.215119         2
4            C1  222.110354  ...     0.755843         4
...         ...         ...  ...          ...       ...
2050         C3  468.859681  ...     4.519621         3
2051         C3  499.155766  ...     2.149398         3
2052         C3  490.217290  ...     2.935094         4
2053         C3  483.829284  ...     1.286765         3

[2053 rows x 10 columns]
```

#### Index

The index is an arbitrary label given to each contact. It is named `cluster`
because a single contact event is clustered across nearby pixels in
contiguous frames.

#### Columns
* whisker : whisker name
* tip_x, tip_y : location of the tip
* fol_x, fol_y : location of the follicle
* angle : whisker angle
* frame_start : frame the contact started on (inclusive)
* frame_stop : frame the contact ended on (inclusive)
* duration : duration of the contact

## `sessions/SESSION_NAME/colorized_whisker_ends`

### Format

pickled `pandas.DataFrame`

### Short description

The location of each whisker on each frame.

### Detailed description

#### Example

```
          frame whisker  ...       fol_y      angle
0             0      C0  ...  172.431671 -51.083353
1             0      C1  ...  203.063225 -44.215501
2             0      C2  ...  217.092224 -35.377798
3             0      C3  ...  239.791621 -10.878188
...         ...     ...  ...         ...        ...
2082237  528599      C0  ...  171.665999 -52.850519
2082238  528599      C1  ...  207.020448 -43.687128
2082239  528599      C2  ...  225.207901 -30.049172
2082240  528599      C3  ...  248.406602  -3.678544

[2082241 rows x 7 columns]
```

#### Index

Arbitrary

#### Columns

* `frame` : The frame number within the video.
* `whisker` : Name of the whisker, one of `{'C0', 'C1', 'C2', 'C3'}`.
* `tip_x`, `tip_y` : Location of the tip of the whisker on that frame. Note
  that the origin of the frame is in the upper left, and y-coordinates increase
  going downward (image, not graph, convention).
* `fol_x`, `fol_y` : Location of the follicle (base) of the whisker on
  that frame. Note: this not the actual follicle, but the most proximal
  part of the whisker that can still be tracked.
* `angle` : The angle between the base and the tip, in degrees. Zero degrees
  is horizontal, and positive values mean protraction (forward; toward the
  top of the frame).


## `sessions/SESSION_NAME/cwe_with_kappa`

### Format

pickled `pandas.DataFrame`

### Short description

The location of each whisker on each frame, and its degree of bending (kappa).

### Detailed description

This is identical to the DataFrame called `colorized_whisker_ends`, except
that it also includes an extra column called `kappa`.

#### Example

```
          frame whisker  ...      angle     kappa
0             0      C0  ... -51.083353  0.000223
1             0      C1  ... -44.215501  0.000300
2             0      C2  ... -35.377798 -0.000768
3             0      C3  ... -10.878188 -0.003539
...         ...     ...  ...        ...       ...
2082237  528599      C0  ... -52.850519  0.000323
2082238  528599      C1  ... -43.687128  0.000253
2082239  528599      C2  ... -30.049172 -0.000335
2082240  528599      C3  ...  -3.678544 -0.003451

[2082241 rows x 8 columns]
```

#### Index

Arbitrary

#### Columns

* `kappa` : The bending of the whisker on that frame, in units of (1/mm).
  These values were obtained with Nathan Clack's `whisk` software. A horizontal
  line has kappa = 0. Positive values correspond to the whisker pushing into
  the shape. Negative values are the opposite, typically encountered during
  detachment from the shape.

All other columns (and their values) are identical to `colorized_whisker_ends`.

## `sessions/SESSION_NAME/edge_summary`

### Format

pickled `pandas.DataFrame`

### Short description

The edge of each shape at each of its final positions.

### Detailed description

This is similar to the data in `all_edges`, except that it only includes
the final position of the shape at each possible position, rather than its
position on every individual frame.

This is a 2-dimensional histogram over trials. Each value is a number
indicating on how many trials the edge was contained within that pixel.

You can mean over all index levels other than `row` to generate a heatmap
of possible shape positions.

#### Example

```
col                                  1.0    3.0    ...  637.0  639.0
rewside servo_pos stepper_pos row                  ...              
left    1670      150         1.0      0.0    0.0  ...    0.0    0.0
                              3.0      0.0    0.0  ...    0.0    0.0
                              5.0      0.0    0.0  ...    0.0    0.0
                              7.0      0.0    0.0  ...    0.0    0.0
...                                    ...    ...  ...    ...    ...
right   1850      50          543.0    0.0    0.0  ...    0.0    0.0
                              545.0    0.0    0.0  ...    0.0    0.0
                              547.0    0.0    0.0  ...    0.0    0.0
                              549.0    0.0    0.0  ...    0.0    0.0

[1650 rows x 320 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `rewside` : the rewarded side. One of the following values: {'left', 'right'}.
* `servo_pos` : the servo position (proximity to the face). One of the
  following values: {1670, 1760, 1850}.
* `stepper_pos` : the stepper position (which shape is presented). One of the
  following values: {50, 150, 0, 199}.
* `row` : the row number within the video frame.

#### Columns

* `col` : the column number within the video frame.

## `sessions/SESSION_NAME/frontier_all_stim`

### Format

pickled `pandas.DataFrame`

### Short description

The "frontier" on each frame with respect to the response window,
representing the closest possible position of any shape (the convex hull).
A sampling whisk is one that crosses this frontier, regardless of whether
contact was made.

### Detailed description

This is similar to the data in `all_edges`, except that it is the convex
hull of all possible edges at that time with respect to the response window,
rather than the edge position on every individual frame.

#### Example

```
                     row  col
locking_frame index          
-400          0      496   72
              1      496   74
              2      500   88
              3      500   90
...                  ...  ...
-10           435    368  480
              436    372  484
              437    376  488
              438    442  548

[12330 rows x 2 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `locking_frame` : the frame number with respect to response window opening.
  For efficiency and because the shape moves slowly, this is only calculated
  every 10 frames (50 ms).
* `index` : arbitrary; indexes each pixel in the edge.

#### Columns

* `row` : the row number of that pixel on the edge
* `col` : the column number of that pixel on the edge

## `sessions/SESSION_NAME/joints`

### Format

pickled `pandas.DataFrame`

### Short description

The location of each of 8 equally spaced "joints" along each whisker on
every frame.

### Detailed description

This is similar to `colorized_whisker_ends`, but it contains all 8 joints,
rather than just the follicle and tip (corresponding to the first and last
joints). It also contains the "confidence score" assigned by
DeeperCut/PoseTensorflow/PoseTF to each joint.

The bending (kappa) of each whisker was calculated by fitting a spline
through these joints and running the output through `whisk`.

#### Example

```
                         c              ...         p          
joint                    0           1  ...         6         7
frame  whisker                          ...                    
0      C0       447.833118  466.035449  ...  0.999930  0.999692
       C1       381.717058  410.374808  ...  0.999886  0.999901
       C2       371.997747  402.577738  ...  0.999865  0.999668
       C3       413.103072  438.023576  ...  0.999893  0.999849
...                    ...         ...  ...       ...       ...
528599 C0       455.288867  473.403332  ...  0.999916  0.999035
       C1       371.784443  403.080980  ...  0.999905  0.999942
       C2       353.080241  386.107643  ...  0.999925  0.999949
       C3       406.218208  431.193778  ...  0.999905  0.999854

[2082241 rows x 24 columns]
```

#### Index

`pandas.MultiIndex` with the following levels:
* `frame` : the frame number within the video.
* `whisker` : the name of the whisker, one of `{'C0', 'C1', 'C2', 'C3'}`

#### Columns

`pandas.MultiIndex` with the following levels:
* [unnamed] : one of `{'c', 'r', 'p'}`, corresponding to the column, row,
  and confidence of each joint.
* `joint` : indexes the joint number. An integer between 0 (the follicle) and 7
  (the tip), inclusively.

## `sessions/SESSION_NAME/neural_trial_timings`

### Format

pickled `pandas.DataFrame`

### Short description

This only exists for sessions with neural data. It contains metadata about
the timing of each trial within the _neural timebase_.

### Detailed description

This is similar to `trial_matrix` but it contains information for
synchronizing neural and behavioral data on each trial.

NaN values indicate that that trial did not occur during the neural recording.

Use the `rwin_time_nbase` column to get the response window time of each trial
in the _neural timebase_, the same time base in which the spike times are
reported in the file called `spikes`.

#### Example

```
       start_time_nbase  start_sample  ...  rwin_time_nbase  rwin_sample
trial                                  ...                              
0                   NaN           NaN  ...              NaN          NaN
1                   NaN           NaN  ...              NaN          NaN
2                   NaN           NaN  ...              NaN          NaN
3                   NaN           NaN  ...              NaN          NaN
...                 ...           ...  ...              ...          ...
298         2805.089700    77660691.0  ...      2808.112700   77751381.0
299         2821.080900    78140427.0  ...      2824.104900   78231147.0
300         2828.449300    78361479.0  ...      2831.471300   78452139.0
301         2837.219567    78624587.0  ...      2840.242567   78715277.0

[302 rows x 6 columns]
```

#### Index

The trial number. This aligns exactly with `trial_matrix`.

#### Columns

* `start_time_nbase` : The start time of the trial in the _neural timebase_.
* `start_sample` : The sample number within the neural recording binary file
  corresponding to `start_time_nbase`.
* `nbase_resid` : ignore
* `vframe_npred` : ignore
* `rwin_time_nbase` : The time at which the response window opened on this
  trial, in the _neural timebase_. __Use this column to synchronize neural
  and behavioral data.__
* `rwin_sample` : The sample number within the neural recording binary file
  corresponding to `rwin_time_nbase`.

## `sessions/SESSION_NAME/neural_trial_timings`

### Format

pickled `pandas.DataFrame`

### Short description

This only exists for sessions with neural data. It contains metadata about
the timing of each trial within the _neural timebase_.

### Detailed description

This is similar to `trial_matrix` but it contains information for
synchronizing neural and behavioral data on each trial.

NaN values indicate that that trial did not occur during the neural recording.

Use the `rwin_time_nbase` column to get the response window time of each trial
in the _neural timebase_, the same time base in which the spike times are
reported in the file called `spikes`.

#### Example

```
       start_time_nbase  start_sample  ...  rwin_time_nbase  rwin_sample
trial                                  ...                              
0                   NaN           NaN  ...              NaN          NaN
1                   NaN           NaN  ...              NaN          NaN
2                   NaN           NaN  ...              NaN          NaN
3                   NaN           NaN  ...              NaN          NaN
...                 ...           ...  ...              ...          ...
298         2805.089700    77660691.0  ...      2808.112700   77751381.0
299         2821.080900    78140427.0  ...      2824.104900   78231147.0
300         2828.449300    78361479.0  ...      2831.471300   78452139.0
301         2837.219567    78624587.0  ...      2840.242567   78715277.0

[302 rows x 6 columns]
```

#### Index

The trial number. This aligns exactly with `trial_matrix`.

#### Columns

* `start_time_nbase` : The start time of the trial in the _neural timebase_.
* `start_sample` : The sample number within the neural recording binary file
  corresponding to `start_time_nbase`.
* `nbase_resid` : ignore
* `vframe_npred` : ignore
* `rwin_time_nbase` : The time at which the response window opened on this
  trial, in the _neural timebase_. __Use this column to synchronize neural
  and behavioral data.__
* `rwin_sample` : The sample number within the neural recording binary file
  corresponding to `rwin_time_nbase`.

## `sessions/SESSION_NAME/trial_matrix`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about each behavioral trial.

### Detailed description

#### Example

```
       start_time  release_time  ...  shape_stop_frame  shape_stop_time
trial                            ...                                   
0          11.541        19.324  ...               NaN              NaN
1          19.326        34.185  ...               NaN              NaN
2          34.188        48.838  ...               NaN              NaN
3          48.841        63.540  ...               NaN              NaN
...           ...           ...  ...               ...              ...
298      2701.337      2717.320  ...               NaN              NaN
299      2717.323      2724.685  ...               NaN              NaN
300      2724.688      2733.453  ...               NaN              NaN
301      2733.456           NaN  ...               NaN              NaN

[302 rows x 21 columns]
```

#### Index

The trial number.

#### Columns

* `start_time` : The time the trial started, in the _behavioral timebase_.
  In most cases, you want to use `rwin_time` instead.
* `release_time` : ignore
* `duration` : The duration of the trial, in seconds
* `dirdel` : either 2 or 3, where 3 indicates a reward was delivered
  automatically at the beginning of the response window. All such trials were
  used only for behavioral shaping and are marked as "munged" and ignored.
* `isrnd` : either 2 or 3, where 3 indicates the stimulus was randomly chosen
  on that trial. All other trials were used only for behavioral shaping and
  are marked as "munged" and should be ignored.
* `opto` : either 2 or 3, where 2 indicates no optogenetic stimulus occurred.
  Note that even when this value is 3, an optogenetic stimulus would only have
  been delivered on the minority of sessions for which `opto` is True in
  the file called `session_df`.
* `outcome` : one of `{'hit', 'error', 'spoil', 'curr'}`. `'hit'` means that
  the trial was correct, `'error'` means incorect, `'spoil'` means that the
  mouse made no choice at all, and `'curr'` means the trial was in progress
  when the session ended.
* `choice` : one of `{'left', 'right', 'nogo'}`, where `left` means the mouse
  licked left, `right` means it licked right, and `nogo` means it didn't lick
  at all (in which case the outcome was spoiled).
* `rewside` : the rewarded side, one of `{'left', 'right'}`. The meaning of
  this depends on the task, which is one of `{'discrimination', 'detection'}`
  and is coded in `session_df`. For discrimination, `'left'` means concave and
  `'right'` means convex. For detection, `'left'` means no shape and `'right'`
  means either shape.
* `servo_pos` : the position of the linear servo (distance from the face);
  one of `{1670, 1760, 1850}`, corresponding to far, medium, and close trials.
* `stepper_pos` : the position of the stepper (which shape was presented);
  one of {50, 150, 100, 199}. 50 always means convex and 150 always means
  convex. For discrimination sessions, 100 and 199 mean the flatter shapes,
  which were only used on a minority of sessions. For all detection sessions,
  100 and 199 mean no shape.
* `choice_time` : the time of the choice lick, in seconds.
* `rwin_time` : the time of the response window opening, in seconds in the
  _behavioral timebase_.
* `rt` : the "reaction time", which is simply the difference between the
  `choice_time` and the `rwin_time` (not a true reaction time).
* `exact_start_frame` : the frame number in the video corresponding to the
  start time of the trial.
* `vbase_resid` : ignore
* `exact_start_frame_next` : the frame number in the video corresponding to the
  start time of the next trial.
* `munged` : boolean encoding whether the trial was "munged", meaning that it
  should be ignored. Most commonly, this is because the trial was not truly
  random (`isrnd` is 2) for some reason relating to behavioral shaping.
* `rwin_frame` : the frame number in the video corresponding to the opening
  of the response window. __Use this column to synchronize video and
  behavior.__
* `shape_stop_frame` : the frame number at which the shape reached its
  final position and stopped moving.
* `shape_stop_time` : the time at which the shape reached its final position
  and stopped moving, in the _behavioral timebase_.

## `single_whisker/big_perf`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about the sessions while trimming down to a single whisker.

### Detailed description

#### Example

```
outcome                                                  error  ...      ci_h
mouse n_whiskers session              rewside servo_pos         ...          
KM100 5          20170616152758.KM100 left    1670           9  ...  0.827856
                                              1760           5  ...  0.948911
                                              1850           6  ...  0.892711
                                      right   1670           7  ...  0.924650
...                                                        ...  ...       ...
KM102 0          20170614122002.KM102 left    1850           6  ...  0.823389
                                      right   1670          10  ...  0.813593
                                              1760          10  ...  0.756138
                                              1850           4  ...  0.945536

[132 rows x 6 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels:

* `'mouse'` : the mouse's name
* `'n_whiskers'` : the number of whiskers remaining
* `'session` : the name of the _behavioral session_
* `'rewside'` : one of `{'left', 'right'}`, indicating the rewarded side
* `'servo_pos'` : the position of the servo (distance from the face).
  See documentation in `sessions/SESSION_NAME/trial_matrix`.

#### Columns

* `'error'` : the number of errors
* `'hit'` : the number of correct trials
* `'n'` : the sum of `error` and `hit`
* ``'perf'`` : the proportion of 'hit' trials
* `'ci_l'` : the lower bound of the 95% confidence interval on performance
* `'ci_h'` : the upper bound of the 95% confidence interval on performance

## `training_time/big_sdf`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about the sessions during training.

### Detailed description

#### Example

```
                            scheduler  ...      stage
mouse                                  ...           
219CR 0    ForcedAlternationLickTrain  ...  licktrain
      1    ForcedAlternationLickTrain  ...  licktrain
      2    ForcedAlternationLickTrain  ...  licktrain
      3    ForcedAlternationLickTrain  ...  licktrain
...                               ...  ...        ...
KM131 142                        Auto  ...       rig0
      143                        Auto  ...       rig0
      144                        Auto  ...       rig0
      145                        Auto  ...       rig0

[2006 rows x 7 columns]
```
#### Index

A `pandas.MultiIndex` with the following levels:
* ``'mouse'`` : the name of the mouse
* [unnamed] : index of the training session

#### Columns

* `scheduler` : the name of the scheduler used to deliver the trials
* `stim_set` : the name of the stimulus set
* `board`, `box` : ignore
* `dt_start` : `pandas.Timestamp` of the session start time
* `gs` : name of the session with video data, if any
* `stage` : training stage

## `training_time/resdf`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about each mouse analyzed in the manuscript.

### Detailed description

#### Example

```
      sex         dob  start_date  start_age            task
mouse                                                       
219CR   F  2018-04-30  2018-09-17      140.0  discrimination
221CR   F  2018-05-20  2018-09-24      127.0  discrimination
228CR   F  2018-07-07  2019-01-03      180.0       detection
229CR   F  2018-09-14  2019-01-03      111.0  discrimination
...    ..         ...         ...        ...             ...
KM136   M  2017-03-22        None        NaN          lesion
KM147   M  2017-07-03        None        NaN          lesion
KM148   M  2017-07-03        None        NaN          lesion
KM91    M  2016-05-22        None        NaN  discrimination

[25 rows x 5 columns]
```
#### Index

The name of the mouse.

#### Columns

* `'sex'` : the mouse's sex
* `'dob'` : the mouse's date of birth
* `'start_date'` : the mouse's start date of training
* `'age'` : the mouse's age at the start of training
* `'task'` : 'discrimination' means used for shape discrimination. 'detection'
  means used for shape detection. 'single_whisker' means used only for
  single whisker trim. 'lesion' means used only for lesion. 'discrim_novideo'
  means used for discrimination, but no video available.

## `trainset/optimized_fwr`

### Format

pickled `pandas.DataFrame`

### Short description

Quantification of frame-wise error rate in whisker tracking.

### Detailed description

This is like `trainset/optimized_wwr`, but aggregated over all whiskers
in each frame.

#### Example

```
                     correct  nearly correct  ...  mouse   dataset
session      frame                            ...                 
161215_KM91  11958         4               0  ...   KM91  19-07-05
             37196         4               0  ...   KM91  19-07-05
             64870         4               0  ...   KM91  19-07-05
             76694         4               0  ...   KM91  19-07-05
...                      ...             ...  ...    ...       ...
200401_267CR 86356         3               0  ...  267CR  20-04-05
             221791        4               0  ...  267CR  20-04-05
             258824        4               0  ...  267CR  20-04-05
             294575        4               0  ...  267CR  20-04-05

[11651 rows x 16 columns]
```
#### Index

A `pandas.MultiIndex` with the following levels:
* `'session'` : the name of the session
* `'frame'` : the frame number in the video

#### Columns

* `'correct'` : the number of whiskers correctly traced (out of 4)
* `'nearly_correct'` : the number of whiskers "nearly correctly" traced,
  where this means that if two whiskers were swapped, they were so close
  together that it's not really possible to say which is which anyway.
* `'missing'` : the number of whiskers that were present, but not identified
* `'poor'` : the number of whiskers that were poorly traced
* `'poor_tip'` : the number of whiskers that were poorly traced at the tip
* `'incorrect'` : the number of whiskers mis-classified (e.g., C1 labeled C2)
* `'extraneous'` : the number of whiskers that are actually obscured or
  missing, but were incorrectly labeled
* `'contains_error'` : True if even one type of error on that frame
* `'contains_error_strict'` : True if even one type of error on that frame,
  including the "nearly_correct" type
* `'contains_missing'` : True if even one missing whisker
* `'max_errdist'` : Maximum error distance over all joints and whiskers
* `'max_errdist_tip'` : Maximum error distance over all whisker tips
* `'mouse'` : name of the mouse
* `'dataset'` : date the whisker tracking dataset was curated

## `trainset/optimized_wwr`

### Format

pickled `pandas.DataFrame`

### Short description

Quantification of whisker-wise error rate in whisker tracking.

### Detailed description

#### Example

```
                             assignment correct_assignment  ...  mouse   dataset
session      frame  whisker                                 ...                 
161215_KM91  11958  C0              0.0               True  ...   KM91  19-07-05
                    C1              1.0               True  ...   KM91  19-07-05
                    C2              2.0               True  ...   KM91  19-07-05
                    C3              3.0               True  ...   KM91  19-07-05
...                                 ...                ...  ...    ...       ...
200401_267CR 294575 C0              0.0               True  ...  267CR  20-04-05
                    C1              1.0               True  ...  267CR  20-04-05
                    C2              2.0               True  ...  267CR  20-04-05
                    C3              3.0               True  ...  267CR  20-04-05

[46288 rows x 9 columns]
```
#### Index

A `pandas.MultiIndex` with the following levels:
* `'session'` : the name of the session
* `'frame'` : the frame number in the video
* `'whisker'` : the name of the whisker

#### Columns

* `'assignment'` : which tracked object was labeled as this whisker
* `'correct_assignment'` : which tracked object should have been labeled
  as this whisker
* `'cost'` : the "cost" of this assignment, which is a weighted sum that
  trades off different kinds of errors.
* `'errdist'` : the distance between ground truth and labeled joints,
  aggregated over all joints
* `'errdist_tip'` : the distance between ground truth and labeled tip
* `'typ'` : the type of the assignment, one of
  `{'correct', 'missing', 'poor', 'incorrect', 'poor_tip', 'nearly correct', 'extraneous'}`
* `'whisker_comp'` : the "comparison whisker" used to calculate error distances
  Typically this is the ground truth labeled whisker.
* `'mouse'` : name of the mouse
* `'dataset'` : date the whisker tracking dataset was curated

## `whisker_trim/big_peri`

### Format

pickled `pandas.DataFrame`

### Short description

Metadata about the sessions before and after trimming off all whiskers.

### Detailed description

#### Example

```
                          session  ...   sham
mouse n_day                        ...       
228CR -5     20190717174230.228CR  ...  False
      -4     20190718191133.228CR  ...  False
      -3     20190719170005.228CR  ...  False
      -2     20190720104920.228CR  ...  False
...                           ...  ...    ...
KM131 -3     20180118170958.KM131  ...  False
      -2     20180119170743.KM131  ...  False
      -1     20180122193908.KM131  ...  False
       0     20180123163817.KM131  ...   True

[63 rows x 8 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels:

* `mouse` : the mouse's name
* `n_day` : the index of the day from whisker trim

#### Columns

* `session` : the name of the _behavioral session_
* `stimulus_set` : the stimulus set used
* `'mouse'` : redundant
* `'board', 'box'` : ignore
* `'opto', 'sham'` : ignore, refers to irrelevant optogenetic params


## `whisker_trim/wt_bigtm`

### Format

pickled `pandas.DataFrame`

### Short description

Behavioral metadata about the trials during whisker trim sessions.

### Detailed description

#### Example

```
                            start_time  ...  stepper_pos
session              trial              ...             
20190717174230.228CR 0          11.118  ...          199
                     1          28.553  ...          100
                     2          62.571  ...          150
                     3          68.921  ...          100
...                                ...  ...          ...
20180123163817.KM131 334      4505.108  ...          150
                     335      4556.601  ...          150
                     336      4587.556  ...          150
                     337      4639.611  ...          150

[17673 rows x 11 columns]
```

#### Index

A `pandas.MultiIndex` with the following levels.
* `session` : the name of the _behavioral session_, as in
  `whisker_trim/big_peri`.
* `trial` : the trial number

#### Columns

The columns are defined in `sessions/SESSION_NAME/trial_matrix`.

## `'session_df'`

### Format

pickled `pandas.DataFrame`

### Short description

Behavioral metadata about each session in the dataset.

### Detailed description

#### Example

```
               opto   sham  ... frame_height  n_frames
session                     ...                       
161215_KM91   False  False  ...          600    535200
161216_KM91   False  False  ...          600    639000
170118_KM91   False  False  ...          600    755400
170119_KM91   False  False  ...          600    731400
...             ...    ...  ...          ...       ...
200330_267CR  False  False  ...          550    525600
200331_245CR  False  False  ...          550    535800
200331_267CR  False  False  ...          550    477600
200401_267CR  False  False  ...          550    420000

[115 rows x 9 columns]
```

#### Index

The session name, as used virtually throughout the dataset. For instance,
this is the name of the subdirectory for all files `sessions/SESSION_NAME/*`.
It is also used to index the neural metadata in
`neural_metadata/big_waveform_info_df`.

#### Columns

* `'opto'` : boolean indicating whether optogenetics was used. If False,
  no laser was used. If True, the laser was used (but see `sham`).
* `'sham'` : boolean indicating whether sham optogenetics was used. If True,
  the laser was used, but it wasn't pointed at the brain.
* `'task'` : one of ``{'discrimination', 'detection'}``, indicating which
  task was performed.
* `'twoshapes'` : boolean indicating whether the more difficult flatter
  shapes were used.
* `'mouse'` : name of the mouse
* `'frame_rate'` : frame rate of the video, in frames per second
* `'frame_width'` : frame width of the video
* `'frame_height'` : frame height of the video
* `'n_frames'` : duration of the video, in frames

## `'neural_session_df'`

### Format

pickled `pandas.DataFrame`

### Short description

Behavioral metadata about each session in the dataset for which there is
neural data.

### Detailed description

This is a strict subset of the rows in `session_df`. The only use of this
file is to identify which sessions have neural data.

#### Example

```
               opto   sham  ... frame_height  n_frames
session                     ...                       
180116_KM131  False  False  ...          550    394800
180117_KM131   True  False  ...          550    562800
180118_KM131   True  False  ...          550    445800
180119_KM131   True  False  ...          550    576600
...             ...    ...  ...          ...       ...
200330_267CR  False  False  ...          550    525600
200331_245CR  False  False  ...          550    535800
200331_267CR  False  False  ...          550    477600
200401_267CR  False  False  ...          550    420000

[44 rows x 9 columns]
```

#### Index

See `session_df`.

#### Columns

See `session_df`.
