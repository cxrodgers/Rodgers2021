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

## `sessions/SESSION_NAME/spikes`

### Format

pickled `pandas.DataFrame`

### Short description

This file only exists for sessions with neural data, and it contains
the spike times for each single unit that was recorded.

### Detailed description

#### Example

```
                time  cluster    sample
0         216.406900      833       207
1         216.410900      833       327
2         216.416300       71       489
3         216.416967      194       509
...              ...      ...       ...
1131147  2845.105467      833  78861164
1131148  2845.106967      193  78861209
1131149  2845.108033       50  78861241
1131150  2845.109233      472  78861277

[1131151 rows x 3 columns]

```

#### Index

Arbitrary

#### Columns

* `cluster` : the id of this single unit. This is unique within each session,
  but not across sessions. This matches the index in `big_waveform_info_df`.
  Because the word `cluster` is ambiguous, I plan to replace it with the word
  `neuron`.
* `sample` : the sample number within the recording, always using a sample
  rate of 30 kHz.
* `time` : the time of the spike, in the _neural timebase_.

Note that sample 30000 is not guaranteed to be time 1.0 s, but rather some
arbitrary offset start time plus 1.0 s. This offset corresponds to the time
between starting the Open Ephys display and clicking the record button. The
analysis code relies exclusively on `time` and not on `sample`.
