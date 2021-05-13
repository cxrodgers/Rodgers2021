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

| File | Description |
| --- | --- |
| `example_frames/` | Example images from the behavioral video |
| `flatter_shapes/` | Behavioral data on the more difficult shapes |
| `glm_example/` | Example GLM fits from the neural encoding analysis |
| `glm_fits/` | Coefficients and fitting results from each GLM |
| `flatter_shapes/` | Behavioral data on the more difficult shapes |
| `gradual_trims/` | Behavioral data on gradually trimming off rows |
| `lesion/` | Behavioral data after cortical lesion |
| `sessions/` | A large amount of various behavioral and neural data about each session |
| `single_whisker/` | Behavioral data on trimming to a single whisker |
| `training_time/` | Metadata about mice and their training times |
| `trainset/` | Quantifications of whisker tracking accuracy |
| `whisker_trim/` | Behavioral data on trimming all whiskers |
| `neural_session_df` | Metadata about the sessions with neural data |
| `session_df` | Metadata about every session |

Note that the vast majority of the data is contained within the `sessions/`
directory, in sub-directories named by each session. Each session has a
name like `161215_KM91`, in which the first six digits indicate the date
of the session, and the characters after the `_` indicate the mouse's name.

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

# Detailed description of each file

This section describes the format and content of each individual file
in the dataset.

Because the bulk of the data is contained within the `sessions/` directory,
that directory is described first. It contains 115 subdirectories, one for
each session. The token `SESSION_NAME` is used here to indicate the name of each
subdirectory.

## `sessions/SESSION_NAME/`

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
