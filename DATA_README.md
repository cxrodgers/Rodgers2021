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
