## Simple schematic of task timing
"""
1H	
    TRIAL_TIMELINE	
    N/A	
    Trial timeline
"""
import numpy as np
import matplotlib.pyplot as plt
import my.plot

my.plot.manuscript_defaults()
my.plot.font_embed()


## CONSTS
# Distances
CLOSE_DISTANCE = 5.5 # np.sqrt(2 * 4**2)
MED_DISTANCE = CLOSE_DISTANCE + 2.7
FAR_DISTANCE = CLOSE_DISTANCE + 5.4

# Time plotting start
T_PLOT_START = -2.1
T_PLOT_STOP = 1

# Time servo starts moving
T_SERVO_START = -2

# Times the servo reaches its position
T_FAR = -.9
T_MEDIUM = -.7
T_CLOSE = -.5

# Lick times
T_EARLY_LICK = -.3
T_LICK = .3

# Time the servo starts moving back
T_RETURN = .5

SPEED = -(FAR_DISTANCE - CLOSE_DISTANCE) / (T_FAR - T_CLOSE)
AWAY_DISTANCE = SPEED * (T_CLOSE - T_SERVO_START) + CLOSE_DISTANCE


## Make handles
f, axa = plt.subplots(
    5, 1, figsize=(3.5, 2.4), sharex=True, 
    gridspec_kw={'height_ratios': [6, 0.5, 1, 1, 1]})
f.subplots_adjust(bottom=.25, left=.12, top=.97, right=.97)

# Plot kwargs for the traces
trace_kwargs = {'color': 'k', 'clip_on': False, 'lw': 1,}


## Plot shape position over trial
ax_shape = axa[0]

# The approach trajectory, which is the same for all positions
ax_shape.plot(
    [T_PLOT_START, T_SERVO_START], # time before moving
    [AWAY_DISTANCE, AWAY_DISTANCE], # at start pos
    **trace_kwargs)

ax_shape.plot(
    [T_SERVO_START, T_CLOSE], # time moving to closest pos
    [AWAY_DISTANCE, CLOSE_DISTANCE], # from start to closest pos
    **trace_kwargs)

# Time at each position
ax_shape.plot(
    [T_CLOSE, T_RETURN], # time at closest post
    [CLOSE_DISTANCE, CLOSE_DISTANCE], # at closest pos
    **trace_kwargs)

ax_shape.plot(
    [T_MEDIUM, T_RETURN], # time at medium pos
    [MED_DISTANCE, MED_DISTANCE], # at medium pos
    **trace_kwargs)

ax_shape.plot(
    [T_FAR, T_RETURN], # time at far pos
    [FAR_DISTANCE, FAR_DISTANCE], # at far pos
    **trace_kwargs)    

# Return trajectory from each position
ax_shape.plot(
    [T_RETURN, 0.7],
    [CLOSE_DISTANCE, CLOSE_DISTANCE + SPEED * (0.7 - T_RETURN)],
    'k:', lw=1)

ax_shape.plot(
    [T_RETURN, 0.99],
    [MED_DISTANCE, MED_DISTANCE + SPEED * (0.99 - T_RETURN)],
    'k:', lw=1)

ax_shape.plot(
    [T_RETURN, 0.99],
    [FAR_DISTANCE, FAR_DISTANCE + SPEED * (0.99 - T_RETURN)],
    'k:', lw=1)

# Scale bar
line, = ax_shape.plot([-1.1, -1.1], [CLOSE_DISTANCE, FAR_DISTANCE], 'k-', lw=.8)
ax_shape.text(line.get_xdata().mean() - .05, line.get_ydata().mean(), '5.4 mm', ha='right', va='center', size=12)

# Label the positions
ax_shape.text(1, CLOSE_DISTANCE, 'close', size=12, ha='center', va='center')
ax_shape.text(1, MED_DISTANCE, 'med.', size=12, ha='center', va='center')
ax_shape.text(1, FAR_DISTANCE, 'far', size=12, ha='center', va='center')

# ylabel
ax_shape.set_ylabel('distance\nto shape', size=12)
ax_shape.set_ylim((CLOSE_DISTANCE - 1, AWAY_DISTANCE))


## Plot rwin 
ax_rwin = axa[2]
ax_rwin.plot(
    [T_PLOT_START, 0, 0, T_LICK + .05, T_LICK + .05, T_PLOT_STOP],
    [0, 0, 1, 1, 0, 0],
    **trace_kwargs)
ax_rwin.text(T_PLOT_START + .3, .4, 'response window', size=12, ha='center', va='center')


## Plot lick
ax_lick = axa[3]
ax_lick.plot(
    [T_PLOT_START, 
    T_EARLY_LICK, T_EARLY_LICK, T_EARLY_LICK + .05, T_EARLY_LICK + .05,
    T_LICK, T_LICK, T_LICK + .05, T_LICK + .05, 
    T_PLOT_STOP],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    **trace_kwargs)
ax_lick.text(T_PLOT_START + .3, .4, 'licks', size=12, ha='center', va='center')


## Plot reward
ax_reward = axa[4]
ax_reward.plot(
    [T_PLOT_START, T_LICK + .05, T_LICK + .05, T_LICK + .1, T_LICK + .1, T_PLOT_STOP],
    [0, 0, 1, 1, 0, 0],
    **trace_kwargs)
ax_reward.text(T_PLOT_START + .3, .4, 'reward', size=12, ha='center', va='center')


## Rwin line spanning all axes
axa[-1].plot([0, 0], [-.75, 11], 'k--', clip_on=False, lw=1)


## Pretty
for ax in axa:
    if ax is axa[-1]:
        my.plot.despine(ax, which=('top', 'left', 'right'))
        ax.set_yticks([])
        ax.set_xlabel('time in trial (s)')
    else:
        my.plot.despine(ax, which=('top', 'bottom', 'left', 'right'))
        ax.set_yticks([])
    
    if ax is not axa[0]:
        ax.set_ylim((0, 1))

# Move the time-axis downward
axa[-1].spines['bottom'].set_position(('outward', 4)) 
ax.set_xlim((-2.2, 1.2))
ax.set_xticks((-2, -1, 0, 1))

plt.show()

f.savefig('TRIAL_TIMELINE.svg')
f.savefig('TRIAL_TIMELINE.png', dpi=300)