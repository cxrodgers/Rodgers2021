## main2* is all about prepping whisker positions
# This one defines and dumps "big_C2_tip_whisk_cycles", the timing of each cycle
# This filters, applies Hilbert transform, and extracts peaks and troughs.
# Then it extracts metrics about each cycle, like amplitude, set_point,
# C3vC2, speed, etc.
#
# Reconstruction is worst for 200CR, I think mainly because of double pumps,
# which cannot really be captured accurately.
#
# Dumps
# big_C2_tip_whisk_cycles
# whisking_parameters_by_frame (contains phase)


import json
import os
import numpy as np
import scipy.signal
import pandas


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load data
print("loading data")
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))
big_tip_pos = pandas.read_hdf(os.path.join(params['patterns_dir'], 'big_tip_pos'))

lo_cut = 8 # 4 is too low
hi_cut = 50 # doesn't matter so much
filter_order = 2 # doesn't matter so much
fsamp = 200


## Filter
print("filtering")
# Extract C2 tip angle
tip_angle = big_tip_pos.xs(
    'tip', level='joint', axis=1).xs(
    'angle', level='metric', axis=1)
C2_tip_angle = tip_angle.xs('C2', level='whisker', axis=1)

# Bandpass every session * trial separately
blo, alo = scipy.signal.butter(filter_order, hi_cut / (fsamp / 2), 'low')
bhi, ahi = scipy.signal.butter(filter_order, lo_cut / (fsamp / 2), 'high')

# Apply the filter (to each row of the DataFrame)
filtered_whisking_signal = C2_tip_angle.values.copy()
filtered_whisking_signal = scipy.signal.filtfilt(
    blo, alo, filtered_whisking_signal) 
filtered_whisking_signal = scipy.signal.filtfilt(
    bhi, ahi, filtered_whisking_signal) 
filtered_whisking_signal = pandas.DataFrame(
    filtered_whisking_signal,
    index=C2_tip_angle.index,
    columns=C2_tip_angle.columns)


## Hilbert
print("hilbert")

# Pad the length of each row. Hilbert is extremely slow on prime numbers!
padded_len = scipy.fft.next_fast_len(filtered_whisking_signal.shape[1])
npad = padded_len - filtered_whisking_signal.shape[1]
padded_fws = np.concatenate([
    np.zeros((filtered_whisking_signal.shape[0], npad // 2)),
    filtered_whisking_signal.values,
    np.zeros((filtered_whisking_signal.shape[0], npad - (npad // 2))),
    ], axis=1)

# Hilbert each row
analytic_aow = scipy.signal.hilbert(padded_fws)

# Unpad
analytic_aow = analytic_aow[:, 
    (npad // 2):((npad // 2) + filtered_whisking_signal.shape[1])
    ]

# Extract envelope, phase, etc
ampl_envelope = np.abs(analytic_aow)
phase = np.angle(analytic_aow)
unwrapped = np.unwrap(phase)
inst_freq = np.diff(unwrapped) * fsamp / (2 * np.pi)

# DataFrame this for later
phase_df = pandas.DataFrame(phase, 
    index=filtered_whisking_signal.index, 
    columns=filtered_whisking_signal.columns,
    )

## Make the phase non-decreasing
# Make unwrapped non-decreasing
nd_unwrapped = np.hstack([
    unwrapped[:, :ncol+1].max(1)[:, None] 
    for ncol in range(unwrapped.shape[1])])

# Make unwrapped start with value between 0 and 2 * pi
nd_unwrapped = nd_unwrapped - (2 * np.pi * 
    np.floor(nd_unwrapped[:, 0] / (2 * np.pi)))[:, None]

# If it starts between pi and 2pi, then ceil it to 2pi+.001, so that it
# will always start with a pi-crossing (max retraction) before a 
# 2pi-crossing (max protraction)
nd_unwrapped[(
    (nd_unwrapped > np.pi) &
    (nd_unwrapped <= 2 * np.pi)
    )] = 2 * np.pi + .001


## Identify retract and protract cycle numbers
print("identifying cycles")
# There's a -1 here because we started with 0 to 2pi instead of -pi to pi
retract_cycle = pandas.DataFrame(
    np.floor((nd_unwrapped - np.pi) / (2 * np.pi)).astype(np.int),
    index=C2_tip_angle.index, columns=C2_tip_angle.columns)
protract_cycle = pandas.DataFrame(
    np.floor((nd_unwrapped) / (2 * np.pi)).astype(np.int),
    index=C2_tip_angle.index, columns=C2_tip_angle.columns) - 1

# Stack and keep only the starts of the cycles
retract_starts = retract_cycle.stack().rename('cycle').reset_index(
    ).sort_values(['session', 'trial', 'cycle', 'frame']
    ).drop_duplicates(['session', 'trial', 'cycle'], 
    keep='first')
protract_starts = protract_cycle.stack().rename('cycle').reset_index(
    ).sort_values(['session', 'trial', 'cycle', 'frame']
    ).drop_duplicates(['session', 'trial', 'cycle'], 
    keep='first')
    
# Interdigitate the events
concatted = pandas.concat([retract_starts, protract_starts], axis=0, 
    keys=['trough', 'peak'], names=['event', 'junk_index']).reset_index(
    ).drop('junk_index', 1)
by_cycle = concatted.set_index(
    ['session', 'trial', 'cycle', 'event']
    )['frame'].sort_index()
assert not by_cycle.index.duplicated().any()    


## Error check and remove edge cases
# Always drop cycles -1 and 0 because they are partial
by_cycle = by_cycle.drop([-1, 0], level='cycle')

# Unstack the event
by_cycle = by_cycle.unstack('event')
assert not by_cycle['trough'].isnull().any()

# At most one null protraction per trial (should be the last one)
assert by_cycle['peak'].isnull().astype(np.int).sum(
    level=['session', 'trial']).max() == 1

# Drop the null protraction
by_cycle = by_cycle.dropna().astype(np.int)

# Check that retract is always before protract
# Equality seems possible
assert (by_cycle['trough'] < by_cycle['peak']).all()


## Resample the original signal at the peak and trough times, on all whiskers
print("resampling")
slicing_index_df = by_cycle.stack().rename('frame').reset_index()
to_slice = tip_angle.stack('frame')
sliced = to_slice.loc[
    pandas.MultiIndex.from_frame(
    slicing_index_df[['session', 'trial', 'frame']])]
sliced.index = pandas.MultiIndex.from_frame(
    slicing_index_df[[
    'session', 'trial', 'cycle', 'event']])
sliced = sliced.sort_index()


## Parameterize the cycles
# Extract amplitude as C2_peak - C2_trough
C2_peak = sliced.xs('peak', level='event').loc[:, 'C2']
C2_trough = sliced.xs('trough', level='event').loc[:, 'C2']
amplitude = C2_peak - C2_trough

# Floor amplitude at zero
amplitude[amplitude < 0] = 0

# Extract set_point as C2_trough
set_point = C2_trough

# Extract mid_point as mean(C2_peak, C2_trough)
mid_point = (C2_peak + C2_trough) / 2.0

# Extract spread as C3 - C1, meaned over peak and trough
# C2 and C3 are more correlated than C1 and C2
# That means that subtracting off C2 is a more effective way to decorrelate C3
# So C3vC2 is actually more independent of set_point than C1vC2 is
spread = (sliced['C3'] - sliced['C1']).mean(level=['session', 'trial', 'cycle'])
C3vC2 = (sliced['C3'] - sliced['C2']).mean(level=['session', 'trial', 'cycle'])
C1vC2 = (sliced['C1'] - sliced['C2']).mean(level=['session', 'trial', 'cycle'])

# Concat all params
big_C2_tip_whisk_cycles = pandas.DataFrame.from_dict({
    'amplitude': amplitude,
    'set_point': set_point,
    'mid_point': mid_point,
    'spread': spread,
    'C3vC2': C3vC2,
    'C1vC2': C1vC2,
    'peak_frame_wrt_rwin': by_cycle['peak'],
    'start_frame_wrt_rwin': by_cycle['trough'],
    })
big_C2_tip_whisk_cycles.columns.name = 'param'


## Reconstruct
print("reconstructing")
# Extract params at peak and trough times (using same values for peak and trough)
at_peak = big_C2_tip_whisk_cycles.drop(
    'start_frame_wrt_rwin', axis=1).rename(
    columns={'peak_frame_wrt_rwin': 'frame'}).set_index(
    'frame', append=True).droplevel('cycle')
at_trough = big_C2_tip_whisk_cycles.drop(
    'peak_frame_wrt_rwin', axis=1).rename(
    columns={'start_frame_wrt_rwin': 'frame'}).set_index(
    'frame', append=True).droplevel('cycle')

# Interdigitate peak and trough
# Consider smoothing each peak with its adjacent troughs, and vice versa
interdigitated = pandas.concat([at_peak, at_trough]).sort_index()

# Get frames alone on columns
interdigitated = interdigitated.unstack(['session', 'trial']).T
interdigitated = interdigitated.reindex(big_tip_pos.columns.levels[-1], axis=1)

# Interpolate
interpolated = interdigitated.interpolate(
    method='linear', limit_direction='both', axis=1)

# Get session * trial on index
interpolated = interpolated.unstack('param').swaplevel(axis=1).sort_index(axis=1)

# Append the original phase
phase_df.columns = pandas.MultiIndex.from_tuples(
    [('phase', frame) for frame in phase_df.columns])
interpolated = pandas.concat([interpolated, phase_df], axis=1)

# Reconstruct
C2_reconstructed = (
    interpolated['mid_point'] + 
    np.cos(interpolated['phase']) * interpolated['amplitude'] / 2.0
    )


## Compare
print("comparing")
# Evaluate rsquared, excluding the first and last bit, because this can't
# capture huge deviations in the first and last cycle
rsquared = pandas.Series([
    scipy.stats.linregress(orig[50:-50], rec[50:-50]).rvalue ** 2
    for orig, rec in zip(C2_tip_angle.values, C2_reconstructed.values)
    ], index=C2_tip_angle.index)


## Add columns for absolute frame number
big_C2_tip_whisk_cycles = big_C2_tip_whisk_cycles.reset_index()

# Join rwin_frame
big_C2_tip_whisk_cycles = pandas.merge(
    big_C2_tip_whisk_cycles, 
    big_tm[['rwin_frame']].reset_index(), 
    on=['session', 'trial'])

# Convert to absolute
big_C2_tip_whisk_cycles['peak_frame'] = (
    big_C2_tip_whisk_cycles['rwin_frame'] + 
    big_C2_tip_whisk_cycles['peak_frame_wrt_rwin'])
big_C2_tip_whisk_cycles['start_frame'] = (
    big_C2_tip_whisk_cycles['rwin_frame'] + 
    big_C2_tip_whisk_cycles['start_frame_wrt_rwin'])

# set index
big_C2_tip_whisk_cycles = big_C2_tip_whisk_cycles.set_index(
    ['session', 'trial', 'cycle']
    ).sort_index()


## Add stop frame
# Get cycle alone on index
unstacked_start = big_C2_tip_whisk_cycles['start_frame'].unstack('cycle').T

# Shift and restack
big_C2_tip_whisk_cycles['stop_frame'] = unstacked_start.shift(-1).T.stack()

# Drop null stop_frames (last of each trial, I think)
big_C2_tip_whisk_cycles = big_C2_tip_whisk_cycles.dropna(
    subset=['stop_frame']).copy()
big_C2_tip_whisk_cycles['stop_frame'] = (
    big_C2_tip_whisk_cycles['stop_frame'].astype(np.int))


## Add duration
big_C2_tip_whisk_cycles['duration'] = (
    big_C2_tip_whisk_cycles['stop_frame'] - 
    big_C2_tip_whisk_cycles['start_frame'])


## Calculate some kinematic whisk cycle parameters
# Duration has a long positive tail (mean: 17.6, median: 15, mode: 10)
# and a shoulder at 25.
# I think the main peak is ~11 samples (18Hz) and the shoulder corresponds
# to a double-length cycle at about 9Hz, but this is pretty smudgy.

# Add protract_time
big_C2_tip_whisk_cycles['protract_time'] = (
    big_C2_tip_whisk_cycles['peak_frame_wrt_rwin'] - 
    big_C2_tip_whisk_cycles['start_frame_wrt_rwin'])

# Add retract_time
big_C2_tip_whisk_cycles['retract_time'] = (
    big_C2_tip_whisk_cycles['duration'] - 
    big_C2_tip_whisk_cycles['protract_time']
)

# Define instantaneous frequency as 200/duration (units: Hz)
big_C2_tip_whisk_cycles['inst_frequency'] = (
    200.0 / big_C2_tip_whisk_cycles['duration'])

# Protraction speed (degrees / s)
big_C2_tip_whisk_cycles['protraction_speed'] = (
    big_C2_tip_whisk_cycles['amplitude'] / 
    (big_C2_tip_whisk_cycles['protract_time'] / 200.)
)

# Retraction speed (degrees / s)
big_C2_tip_whisk_cycles['retraction_speed'] = (
    big_C2_tip_whisk_cycles['amplitude'] / 
    (big_C2_tip_whisk_cycles['retract_time'] / 200.)
)


## Dump
# Parameters by cycle
big_C2_tip_whisk_cycles.columns.name = 'metric'
big_C2_tip_whisk_cycles.to_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))

# Dump the peak and trough frames of each cycle
# This is not actually ever used again, so comment this out
#~ by_cycle.to_pickle(
    #~ os.path.join(params['patterns_dir'], 'peak_and_trough_frames'))

# Dump all of the whisking parameters by frame
# Phase is directly from hilbert, the others are interpolated between
# peaks and troughs
interpolated.to_pickle(
    os.path.join(params['patterns_dir'], 'whisking_parameters_by_frame'))


