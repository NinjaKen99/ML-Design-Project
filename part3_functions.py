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
    n = len(sentence)
    memo = []
    memo.append({'Start': [(0, [])]})

    label_list = TAGS

    for j in range(n):
        memo.append({})
        #Check if word in training set, else set to #UNK#
        lowerword = sentence[j]
        memo.append({})  # Initialize the memoization dictionary for this position
        
        # Check if the word is in the training set, else set to '#UNK#'
        word = lowerword if lowerword in words_observed else '#UNK#'

        #Calc scores
        for next_label in label_list:
            if word in emission_parameters[next_label]:
                emission_prob = emission_parameters[next_label][word]
            else:
                emission_prob = -float("inf")

            entries = []
            for prev_label in memo[j].keys():
                prev_entries = memo[j][prev_label]
                for v, path in prev_entries:
                    if prev_label in transition_parameters and next_label in transition_parameters[prev_label]:
                        transition_prob = transition_parameters[prev_label][next_label]
                    else:
                        transition_prob = -float('inf')
                    new_score = v + emission_prob + transition_prob
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
            print('Unexpected Transition Scenario') # occurs when there are no valid transitions from the previous tag to any of the possible next tags. This can happen if the emission probabilities for all possible next tags are very low or if the transition probabilities from the previous tag to all possible next tags are also very low.


    #Termination
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

    #Get k-th likely sequence
    seq = memo[-1]['Stop'][0][1]
    seq.pop(0)
    return seq



#Predict labels for .in file and print to .out file
# def predict_labels_p3(emission_parameters, transition_parameters, vocab, k, input_file, output_file):
#     observations = read_unlabelled(input_file)
#     with open(output_file, 'w') as f_out:
#         for observation in observations:
#             labels = k_viterbi_algo(transition_parameters, emission_parameters, vocab, observation, k)
#             for i in range(len(observation)):
#                 f_out.write(observation[i] + ' ' + labels[i] + '\n')
#             f_out.write('\n')

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
print("Donerino")
#----------------------------------------------------------------------------------------#

# #For RU
# # Load training data for words and tags
# words, tags = RU_train

# # Estimate transition parameters based on the training tags
# transition_params = estimate_transition_params(tags)

# # Estimate emission parameters and observed words using training data
# emission_params, words_observed = estimate_emission_params_log(words, tags)

# RU_words_2 = open_unlabelled_data('RU/dev.in')
# with open('RU/dev.p3.2nd.out', 'w', encoding="utf-8") as f_out:
#     for word in RU_words_2:
#         RU_labels = k_viterbi(transition_params, emission_params, words_observed, word, 2)
#         for i in range(len(word)):
#             print(len(word),len(RU_labels))
#             f_out.write(word[i] + ' ' + RU_labels[i] + '\n')
#         f_out.write('\n')

# RU_words_8 = open_unlabelled_data('RU/dev.in')
# with open('RU/dev.p3.8th.out', 'w', encoding="utf-8") as f_out:
#     for word in RU_words_2:
#         RU_labels = k_viterbi(transition_params, emission_params, words_observed, word, 8)
#         for i in range(len(word)):
#             f_out.write(word[i] + ' ' + RU_labels[i] + '\n')
#         f_out.write('\n')

# print("Done!")
#Testing
# words, tags = RU_train
# transition_parameters = estimate_transition_params(tags)
# emission_parameters, vocab = estimate_emission_params_log(words, tags)
# x = 23
# k = 4
# words, tags = ES_train
# predict = k_viterbi_algo(transition_parameters, emission_parameters, vocab, words[x], k)
# assert(len(tags[x]) == len(predict))
# print(tags[x])
# print(predict)
