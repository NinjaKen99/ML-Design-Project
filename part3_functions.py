# Import libraries
import numpy as np

# Imports from another file
from fixed_parameters import ES_train, RU_train, TAGS
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data
from part2_functions import estimate_transition_params
from part2_functions import estimate_emission_params_log

# Functions
def k_viterbi(transition_parameters, emission_parameters, words_observed, sentence, k):
    #Initialisation
    memo = []
    memo.append({'Start': [(0, [])]})
    n = len(sentence)
   
  
    # Get the list of tags for transition calculation
    tag_list = TAGS
    # Iterate through each position in the sentence
    for j in range(n):
        memo.append({})
        lowerword = sentence[j]
        memo.append({})   # Initialize the memoization dictionary for this position
        
        # Check if the word is in the training set, else set to '#UNK#'
        word = lowerword if lowerword in words_observed else '#UNK#'

        # Calculate scores for each possible next tag, similar to previous viterbi
        for next_tag in tag_list:
            if word in emission_parameters[next_tag]:
                emission_prob = emission_parameters[next_tag][word]
            else:
                emission_prob = -float("inf")

            entries = []
            for prev_label in memo[j].keys():
                prev_entries = memo[j][prev_label]
                for viterbi_score, path in prev_entries:
                    if prev_label in transition_parameters and next_tag in transition_parameters[prev_label]:
                        transition_prob = transition_parameters[prev_label][next_tag]
                    else:
                        transition_prob = -float('inf')
                    
                    # Calculate the new score based on Viterbi and transition probabilities
                    new_score = viterbi_score + emission_prob + transition_prob
                    if new_score > -float('inf'):
                        new_path = path.copy()
                        new_path.append(prev_label)
                        entries.append((new_score, new_path))
            entries.sort(key = lambda x: x[0])
            while len(entries) > k:
                entries.pop(0)
            if len(entries) > 0:
                memo[j + 1][next_tag] = entries

        # Unexpected Transition Scenario where all possible combinations lead to a probability of 0 (log-prob = -inf)
        # Default behavior is to assign 'O' as a label and set the log-prob as the max in the previous step
        if len(list(memo[j+1].keys())) < 1:
            entries = []
            for prev_label in transition_parameters.keys():
                prev_entries = memo[j].get(prev_label, [])
                for viterbi_score, path in prev_entries:
                    new_path = path.copy()
                    new_path.append(prev_label)
                    entries.append((viterbi_score, new_path))
            entries.sort(key = lambda x: x[0])
            while len(entries) > k:
                entries.pop(0)
            if len(entries) > 0:
                memo[j + 1]['O'] = entries
            print('Unexpected Transition Scenario') # occurs when there are no valid transitions from the previous tag to any of the possible next tags. This can happen if the emission probabilities for all possible next tags are very low or if the transition probabilities from the previous tag to all possible next tags are also very low.


    #Termination step
    entries = []
    for prev_label in transition_parameters.keys():
        prev_entries = memo[n].get(prev_label, [])
        for v, path in prev_entries:
            a = transition_parameters.get(prev_label, {}).get('Stop', -float('inf'))
            new_score = v + a
            if new_score > -float('inf'):
                new_path = path.copy()
                new_path.append(prev_label)
                entries.append((new_score, new_path))
    entries.sort(key = lambda x: x[0])
    while len(entries) > k:
        entries.pop(0)
    memo.append({'Stop': entries})

    #Get k best output sequence
    output = memo[-1]['Stop'][0][1]
    output.pop(0)
    return output

#____________________TESTING____________________#
# run functions below

#For ES
# Load training data for words and tags
words, tags = ES_train

# Estimate transition parameters based on the training tags
transition_params = estimate_transition_params(tags)

# Estimate emission parameters and observed words using training data
emission_params, words_observed = estimate_emission_params_log(words, tags)

# Run k_viterbi twice, with 2nd position and 8th position for RU


ES_words_2 = open_unlabelled_data('ES/dev.in')
with open('ES/dev.p3.2nd.out', 'w', encoding="utf-8") as f_out:
    for word in ES_words_2:
        ES_labels = k_viterbi(transition_params, emission_params, words_observed, word, 1)
        for i in range(len(word)):
 
            f_out.write(word[i] + ' ' + ES_labels[i] + '\n')
print("------------------------DONE------------------------")


ES_words_8 = open_unlabelled_data('ES/dev.in')
print("ES 2ND DONE")
with open('ES/dev.p3.8th.out', 'w', encoding='utf-8') as f_out:
    for word in ES_words_8:
    
        ES_labels = k_viterbi(transition_params, emission_params, words_observed, word, 8)
        for i in range(len(word)):
            f_out.write(word[i] + ' ' + ES_labels[i] + '\n')
        f_out.write('\n')
print("Donerino 🤪")
#----------------------------------------------------------------------------------------#
