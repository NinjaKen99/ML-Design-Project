# Import libraries
import numpy as np

# Imports from another file
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data

# Functions

#Modified viterbi_algo to find k-th best
#Instead of storing just a tuple in memo, it now stores list of tuples of the top-k log-probs
#Instead of storing prev_label, it stores path up to that point
def k_viterbi_algo(transition_parameters, emission_parameters, vocab, sentence, k):
    #Initialisation
    n = len(sentence)
    memo = []
    memo.append({'START': [(0, [])]})

    label_list = list(transition_parameters.keys())
    label_list.remove('START')
    label_list.remove('STOP')

    for j in range(n):
        memo.append({})
        #Check if word in training set, else set to #UNK#
        if vocab.get(sentence[j], False):
            observation = sentence[j]
        else:
            observation = '#UNK#'

        #Calc scores
        for next_label in label_list:
            b = emission_parameters[next_label].get(observation, -float('inf'))

            entries = []
            for prev_label in transition_parameters.keys():
                prev_entries = memo[j].get(prev_label, [])
                for v, path in prev_entries:
                    a = transition_parameters.get(prev_label, {}).get(next_label, -float('inf'))
                    new_score = v + b + a
                    #only add entry if meaningful for easier debugging
                    if new_score > -float('inf'):
                        new_path = path.copy()
                        new_path.append(prev_label)
                        entries.append((new_score, new_path))
            entries.sort(key = lambda x: x[0])
            while len(entries) > k:
                entries.pop(0)
            #only add entry if meaningful for easier debugging
            if len(entries) > 0:
                memo[j + 1][next_label] = entries
        
        #edge case where by some miracle all possible combi lead to probability = 0 (log-prob = -inf) and nothing is recorded
        #Current default is assign 'O' as label and set log-prob as max in prev step (with corresponding label)
        if len(list(memo[j+1].keys())) < 1:
            entries = []
            for prev_label in transition_parameters.keys():
                prev_entries = memo[j].get(prev_label, [])
                for v, path in prev_entries:
                    new_path = path.copy()
                    new_path.append(prev_label)
                    entries.append((v, new_path))
            entries.sort(key = lambda x: x[0])
            while len(entries) > k:
                entries.pop(0)
            if len(entries) > 0:
                memo[j + 1]['O'] = entries
            print('edge case encountered')

    #Termination
    entries = []
    for prev_label in transition_parameters.keys():
        prev_entries = memo[n].get(prev_label, [])
        for v, path in prev_entries:
            a = transition_parameters.get(prev_label, {}).get('STOP', -float('inf'))
            new_score = v + a
            if new_score > -float('inf'):
                new_path = path.copy()
                new_path.append(prev_label)
                entries.append((new_score, new_path))
    entries.sort(key = lambda x: x[0])
    while len(entries) > k:
        entries.pop(0)
    memo.append({'STOP': entries})

    #Get k-th likely sequence
    seq = memo[-1]['STOP'][0][1]
    seq.pop(0)
    return seq

#Predict labels for .in file and print to .out file
def predict_labels_p3(emission_parameters, transition_parameters, vocab, k, input_file, output_file):
    observations = read_unlabelled(input_file)
    with open(output_file, 'w') as f_out:
        for observation in observations:
            labels = k_viterbi_algo(transition_parameters, emission_parameters, vocab, observation, k)
            for i in range(len(observation)):
                f_out.write(observation[i] + ' ' + labels[i] + '\n')
            f_out.write('\n')

#____________________TESTING____________________#
# run funtions below

#Testing
observations, labels = read_labelled('Data/ES/train')
transition_parameters = estimate_transition_parameters(labels)
emission_parameters, vocab = estimate_emission_parameters(observations, labels)
x = 23
k = 4

predict = k_viterbi_algo(transition_parameters, emission_parameters, vocab, observations[x], k)
assert(len(labels[x]) == len(predict))
print(labels[x])
print(predict)

# memo = k_viterbi_algo(transition_parameters, emission_parameters, vocab, observations[x], k)
# print(len(observations[x]))
# print(len(memo))
# print(memo)