## Simulate trial balancing to show it works

"""
S3B, right; S3C
    PLOT_SIMULATED_DATA_AND_ACCURACY
    N/A
    Plot simulated trial balancing data and accuracy
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import my.plot
import sklearn.linear_model

np.random.seed(0)

my.plot.manuscript_defaults()
my.plot.font_embed()


## Simulate some data
# This sets the ratio of correct to incorrect
correct_ratio = 2
n_incorrect = 30

trial_params = pandas.DataFrame.from_dict({
    'stimulus': ['A', 'A', 'B', 'B'],
    'choice': ['A', 'B', 'A', 'B'],
    'n_trials': [
        n_incorrect * correct_ratio, n_incorrect, 
        n_incorrect, n_incorrect * correct_ratio,
        ],
    }).set_index(['stimulus', 'choice'])

# Generate trial_matrix
res_l = []
for stimulus, choice in trial_params.index:
    n_trials = trial_params.loc[(stimulus, choice), 'n_trials']
    res = pandas.DataFrame(
        [(stimulus, choice)] * n_trials, columns=['stimulus', 'choice'])
    res_l.append(res)
trial_matrix = pandas.concat(res_l, ignore_index=True)
trial_matrix.index.name = 'trial'

# Designate folds, should be okay to alternate since trial types are 
# slowly varying
trial_matrix['set'] = np.mod(trial_matrix.index, 2)
trial_matrix['set'] = trial_matrix['set'].replace(
    {0: 'train', 1: 'test'})

# Designate weight
trial_matrix['mouse_correct'] = trial_matrix['stimulus'] == trial_matrix['choice']
trial_matrix['weight'] = trial_matrix['mouse_correct'].replace(
    {True: 1, False: correct_ratio})


## Define neuron weights
# The simulation params
# stim_choice_ratio: 0 = purely choice, 1 = purely stim
# noise_ratio: 0 = all signal (stim and choice combined); 1 = all noise
# Could also add a "gain" parameter, but this likely doesn't matter
neuron_weights = pandas.MultiIndex.from_product([
    pandas.Series(np.array([.2, .4, .6]), name='noise_ratio'),
    pandas.Series(np.linspace(0, 1, 5), name='stim_choice_ratio'),
    ]).to_frame().reset_index(drop=True)

# The actual weights of noise, stim, and choice
# These are constrained to add to 1
neuron_weights['noise_weight'] = neuron_weights['noise_ratio']
neuron_weights['stim_weight'] = (
    (1 - neuron_weights['noise_weight']) * neuron_weights['stim_choice_ratio'])
neuron_weights['choice_weight'] = (
    1 - neuron_weights['noise_weight'] - neuron_weights['stim_weight'])
neuron_weights.index.name = 'neuron'


## Iterate over repetitions
# Store data here
all_neuron_preds_l = []
all_neuron_coefs_l = []
all_neuron_preds_keys_l = []
response_data_l = []
response_data_keys_l = []

# This needs to be somewhat high because of the coin flip issue
n_repetitions = 50

# Iterate over repetitions
for n_repetition in range(n_repetitions):
    ## Generate data
    # Merge trial matrix and neuron weights
    to_merge1 = trial_matrix.reset_index()
    to_merge2 = neuron_weights.reset_index()
    to_merge1['key'] = 0
    to_merge2['key'] = 0
    merged = pandas.merge(to_merge1, to_merge2, on='key', how='outer')

    # Draw
    merged['response'] = (
        -merged['stim_weight'] * (merged['stimulus'] == 'A').astype(np.int) + 
        merged['stim_weight'] * (merged['stimulus'] == 'B').astype(np.int) + 
        -merged['choice_weight'] * (merged['choice'] == 'A').astype(np.int) +
        merged['choice_weight'] * (merged['choice'] == 'B').astype(np.int)
        )
    merged['response'] += (
        merged['noise_weight'] * np.random.standard_normal(len(merged)))

    # Drop the metadata and include only the responses
    response_data = merged.set_index(
        ['neuron', 'trial'])['response'].unstack('neuron')

    
    ## Implement trial dropping, consistently for all decodes and neurons
    # Do only on the train
    train_trial_matrix = trial_matrix[trial_matrix['set'] == 'train'].copy()
    test_trial_matrix = trial_matrix[trial_matrix['set'] == 'test'].copy()
    
    # Size of the smallest group
    gobj = train_trial_matrix.groupby(['stimulus', 'choice'])
    dropped_n = gobj.size().min()
    
    # Choose
    dropped_sub_tm_l = []
    for keys, sub_tm in gobj:
        # Just take the first N because there's no temporal ordering here
        dropped_sub_tm = sub_tm.iloc[:dropped_n]
        dropped_sub_tm_l.append(dropped_sub_tm)
    
    # Append test
    dropped_sub_tm_l.append(test_trial_matrix)
    
    # Generate new dropped_tm
    dropped_tm = pandas.concat(dropped_sub_tm_l).sort_index()


    ## Iterate first over how to decode, then over neurons
    for decode_label in ['stimulus', 'choice']:
        for fit_method in ['naive', 'balanced', 'trial dropping']:
            
            ## Iterate over neurons
            preds_l = []
            coefs_l = []
            preds_keys_l = []
            for neuron in neuron_weights.index:
                # Get neuron_data
                neuron_data = response_data.loc[:, neuron]

                if fit_method == 'trial dropping':
                    # Join on trial matrix
                    neuron_data = dropped_tm.join(neuron_data)
                else:
                    # Join on trial matrix
                    neuron_data = trial_matrix.join(neuron_data)
                    assert len(neuron_data) == response_data.shape[0]                          
                
                # Split
                train_data = neuron_data[neuron_data['set'] == 'train']
                test_data = neuron_data[neuron_data['set'] == 'test']

                
                ## Decode
                # Set up model
                # With only one feature, all this can do is threshold
                # And by design we know the best threshold (intercept) is always zero
                # So this could be done simply by assessing datapoints wrt zero
                model = sklearn.linear_model.LogisticRegression(
                    fit_intercept=False, C=1.0,
                    )

                # Fit
                if fit_method in ['naive', 'trial dropping']:
                    model.fit(
                        train_data.loc[:, [neuron]].values, 
                        train_data[decode_label].values,
                        )
                elif fit_method == 'balanced':
                    model.fit(
                        train_data.loc[:, [neuron]].values, 
                        train_data[decode_label].values,
                        sample_weight=train_data.loc[:, 'weight'].values,
                        )
                else:
                    1/0

                # Predict
                preds = model.predict(
                    test_data.loc[:, [neuron]].values, 
                    )
                preds = pandas.Series(preds, index=test_data.index, name='pred')
                
                # Store
                preds_l.append(preds)
                preds_keys_l.append(neuron)
                coefs_l.append(model.coef_.item())

            
            ## Concat results over neurons
            # index: trials; columns: neuron
            all_neuron_preds = pandas.concat(
                preds_l, keys=preds_keys_l, names=['neuron'], axis=1)
            all_neuron_coefs = pandas.Series(coefs_l, index=neuron_weights.index)

            
            ## Store
            all_neuron_preds_l.append(all_neuron_preds)
            all_neuron_coefs_l.append(all_neuron_coefs)
            all_neuron_preds_keys_l.append(
                (n_repetition, decode_label, fit_method))
    
    
    ## Store the generated responses
    response_data_l.append(response_data)
    response_data_keys_l.append((n_repetition))
    

## Concat
# Responses
all_responses = pandas.concat(
    response_data_l,
    keys=response_data_keys_l, 
    names=['n_rep'])

# Coefs
all_coefs = pandas.concat(
    all_neuron_coefs_l,
    keys=all_neuron_preds_keys_l, 
    axis=1,
    names=['n_rep', 'decode_label', 'fit_method']).T

# Predictions
all_preds = pandas.concat(
    all_neuron_preds_l, 
    keys=all_neuron_preds_keys_l, 
    names=['n_rep', 'decode_label', 'fit_method'])

# Stack into series
all_preds = all_preds.stack('neuron').rename('pred')
all_coefs = all_coefs.stack('neuron').rename('coef')


## Join trial data onto preds
all_preds = all_preds.to_frame().join(trial_matrix)

# Error check that we're only studying the test set
assert (all_preds['set'] == 'test').all()
all_preds = all_preds.drop('set', axis=1)

# Append what we were trying to predict
all_preds['target'] = all_preds['stimulus'].copy()
choice_mask = all_preds.index.get_level_values('decode_label') == 'choice'
all_preds.loc[choice_mask, 'target'] = all_preds.loc[
    choice_mask, 'choice'].copy()

# Error check that worked
mask = all_preds.index.get_level_values('decode_label') == 'stimulus'
assert (
    all_preds.loc[mask, 'target'] == 
    all_preds.loc[mask, 'stimulus']).all()
mask = all_preds.index.get_level_values('decode_label') == 'choice'
assert (
    all_preds.loc[mask, 'target'] == 
    all_preds.loc[mask, 'choice']).all()

# Score
all_preds['pred_correct'] = all_preds['pred'] == all_preds['target']
all_preds['weighted_pred_correct'] = (
    all_preds['pred_correct'].astype(np.float) * all_preds['weight'])


## Aggregate score over trials
def aggregate_score(preds):
    # Flat score
    flat_score = preds['pred_correct'].mean(
        level=[lev for lev in preds.index.names if lev != 'trial'])

    # Weighted score
    weighted_score_num = preds['weighted_pred_correct'].sum(
        level=[lev for lev in preds.index.names if lev != 'trial'])
    weighted_score_den = preds['weight'].sum(
        level=[lev for lev in preds.index.names if lev != 'trial'])
    weighted_score = weighted_score_num / weighted_score_den

    # Concat scores
    scores = pandas.concat(
        [flat_score, weighted_score], axis=1, keys=['flat', 'weighted'])

    return scores

# Ignoring outome
scores = aggregate_score(all_preds)

# Consider outcome
all_preds_by_outcome = all_preds.set_index(
    'mouse_correct', append=True).reorder_levels(
    ['mouse_correct', 'n_rep', 'decode_label', 'fit_method', 'neuron', 'trial']
    ).sort_index()
scores_by_outcome = aggregate_score(all_preds_by_outcome)

    
## Aggregate coefs over repetitions
coefs_meanrep = all_coefs.mean(
    level=[lev for lev in all_coefs.index.names if lev != 'n_rep'])
coefs_errrep = all_coefs.std(
    level=[lev for lev in all_coefs.index.names if lev != 'n_rep'])


## Aggregate scores over repetitions
# Note that in some cases sign flips make the score either 0 or 1 across reps

# Aggregate scores ignoring outcome
scores_meanrep = scores.mean(
    level=[lev for lev in scores.index.names if lev != 'n_rep'])
scores_errrep = scores.std(
    level=[lev for lev in scores.index.names if lev != 'n_rep'])

# Aggregate scores considering outcome
scores_meanrep_by_outcome = scores_by_outcome.mean(
    level=[lev for lev in scores_by_outcome.index.names if lev != 'n_rep'])
scores_errrep_by_outcome = scores_by_outcome.std(
    level=[lev for lev in scores_by_outcome.index.names if lev != 'n_rep'])


## Plots
PLOT_SIMULATED_DATA_AND_ACCURACY = True

if PLOT_SIMULATED_DATA_AND_ACCURACY:
    neuron = 9 
    bins = np.linspace(-2, 2, 31)
    bincenters = (bins[1:] + bins[:-1]) / 2    

    # Get data for this neuron
    neuron_data = all_responses.loc[:, neuron].unstack('n_rep')
    
    # Histogram neuron data by trial type
    grouping_key_names = ['stimulus', 'choice']
    gobj = trial_matrix.groupby(grouping_key_names)
    counts_l = []
    counts_keys_l = []
    for grouping_keys, sub_tm in gobj:
        # Get data for this neuron on corresponding trials
        this_neuron_data = neuron_data.loc[sub_tm.index]
        
        # Histo it
        counts, edges = np.histogram(this_neuron_data.values, bins=bins)
        
        # Store
        counts_l.append(counts)
        counts_keys_l.append(grouping_keys)
    
    # Concatenate histo over trial types
    counts_df = pandas.DataFrame(
        np.array(counts_l), 
        index=pandas.MultiIndex.from_tuples(
        counts_keys_l, names=grouping_key_names)).T
    counts_df.index.name = 'bin'
    
    # Convert to trials per repetition 
    counts_df = counts_df / n_repetitions


    ## Plot
    f = plt.figure(figsize=(6, 3))
    axa = [
        f.add_axes([.15, .175, .325, .6]),
        f.add_axes([.625, .5, .35, .35]),
        ]

    ## Plot each distribution
    ax = axa[0]
    for stimulus in ['A', 'B']:
        for choice in ['A', 'B']:
            # Get plot kwargs
            if stimulus == 'A' and choice == 'A':
                color = 'r'
                ls = '-'
            elif stimulus == 'A' and choice == 'B':
                color = 'r'
                ls = '--'
            elif stimulus == 'B' and choice == 'A':
                color = 'b'
                ls = '-'
            elif stimulus == 'B' and choice == 'B':
                color = 'b'
                ls = '--'
            
            # Slice
            this_histo = counts_df.loc[:, (stimulus, choice)]
            
            # Plot
            ax.plot(bincenters, this_histo, color=color, ls=ls, alpha=.5)
    
    # Pretty ax
    my.plot.despine(ax)
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=9)
    ax.set_yticks((0, 4, 8))
    ax.set_xlabel('observed feature')
    ax.set_ylabel('# of trials')
    
    #~ # Legend
    #~ f.text(.44, .92, 'choice A solid', color='k', ha='right', va='center', size=8)
    #~ f.text(.44, .9, 'choice B dashed', color='k', ha='right', va='center', size=8)
    #~ f.text(.44, .88, 'stim A', color='r', ha='right', va='center', size=8)
    #~ f.text(.44, .86, 'stim B', color='b', ha='right', va='center', size=8)    
    
    
    ## Plot coefs obtained
    neuron_mcoefs = coefs_meanrep.xs(neuron, level='neuron')
    neuron_ecoefs = coefs_errrep.xs(neuron, level='neuron')
    
    xticks = [0, 1, 2, 5, 6, 7]
    axa[1].bar(
        x=xticks,
        height=neuron_mcoefs.values,
        yerr=neuron_ecoefs.values,
        facecolor='w',
        edgecolor='k',
        lw=1,
        )
    axa[1].plot([xticks[-1] + 1 , xticks[0] - 1], [0, 0], 'k:', lw=.75)
    axa[1].set_xticks(xticks)
    xticklabels = neuron_mcoefs.index.get_level_values('fit_method')
    xticklabels = [xticklabel if xticklabel != 'trial dropping' else 'dropping' 
        for xticklabel in xticklabels]
    axa[1].set_xticklabels(xticklabels, rotation=90)
    axa[1].set_xlim((xticks[0] - 1, xticks[-1] + 1))
    axa[1].set_ylim(ymin=-1)
    axa[1].set_ylim(ymax=5)
    axa[1].set_yticks((0, 5))
    axa[1].set_ylabel('evidence (logits)')
    my.plot.despine(axa[1])
    axa[1].set_xlabel('method')
    
    axa[1].text(xticks[1], axa[1].get_ylim()[1], 'stimulus', ha='center', va='bottom')
    axa[1].text(xticks[4], axa[1].get_ylim()[1], 'choice', ha='center', va='bottom')

    f.savefig('PLOT_SIMULATED_DATA_AND_ACCURACY.svg')
    f.savefig('PLOT_SIMULATED_DATA_AND_ACCURACY.png', dpi=300)

plt.show()