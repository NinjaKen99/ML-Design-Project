# Import libraries
import numpy as np

# Imports from another file
from fixed_parameters import ES_train, RU_train, TAGS
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data

# Functions

def estimate_transition_params(tags):
    # Initialize the transition parameters dictionary with 'Start' and 'Stop' tags
    transition_params = {'Start': {'Total': 0}}
    
    # Count transitions between tags
    for sentence in range(len(tags)):  # For each sentence
        for word_iterator in range(len(tags[sentence])):  # For each word in the sentence
            # If it's the first word in the sentence, use 'Start' as the previous tag
            if word_iterator == 0:
                previous_tag = 'Start' # Use 'Start' as the previous tag
            else:
                previous_tag = tags[sentence][word_iterator - 1] # Use the previous tag
            
            # Initialize the count for the current tag under the previous tag
            if previous_tag not in transition_params.keys(): # If the previous tag is not a key in transition_params
                transition_params[previous_tag] = {'Total': 0} # Initialize the previous tag with a count for smoothing
            
            # Get the current count of transitions from previous tag to current tag
            current_count = transition_params[previous_tag].get(tags[sentence][word_iterator], 0)
            
            # Increment the count for the current tag transition
            transition_params[previous_tag][tags[sentence][word_iterator]] = current_count + 1
            
            # Increment the total count for transitions from the previous tag
            transition_params[previous_tag]['Total'] += 1
        
        # Count the transition from the last tag in the sentence to 'Stop'
        current_count = transition_params[tags[sentence][-1]].get('Stop', 0) # Get the current count of transitions from the last tag to 'Stop'
        transition_params[tags[sentence][-1]]['Stop'] = current_count + 1 # Increment the count for the current tag transition
        transition_params[tags[sentence][-1]]['Total'] += 1 # Increment the total count for transitions from the last tag

    # Calculate the log-probabilities for transitions
    for tag in transition_params.keys():  # For each tag
        for transition_word in transition_params[tag].keys():  # For each transition word
            if transition_word != 'Total': # If it's not the total count
                transition_params[tag][transition_word] = np.log(transition_params[tag][transition_word] / transition_params[tag]['Total']) # Calculate the log-probability by dividing the count by the total count and taking the log

    
    return transition_params

def estimate_emission_params_log(words, tags, k=1):
    # Initialize dictionaries to store emission log-probabilities and observed words
    emission_params = {}  # Stores emission log-probabilities for each tag and word
    words_observed = {}   # Stores observed words
    
    # Loop through each sentence and its words in the tagged data
    for sentence in range(len(tags)):
        for tag_for_word in range(len(tags[sentence])):
            # Check if the tag is already a key in emission_params
            if tags[sentence][tag_for_word] not in emission_params.keys():
                # Initialize emission_params[tag] with counts for smoothing
                emission_params[tags[sentence][tag_for_word]] = {'Total': k, '#UNK#': k}
            
            # Get the current count of the word's emission for the tag
            current_count = emission_params[tags[sentence][tag_for_word]].get(words[sentence][tag_for_word], 0)
            
            # Increment the emission count for the word's tag and word
            emission_params[tags[sentence][tag_for_word]][words[sentence][tag_for_word].lower()] = current_count + 1
            
            # Increment the total emission count for the tag
            emission_params[tags[sentence][tag_for_word]]['Total'] += 1
            
            # Track that the word has been observed
            words_observed[words[sentence][tag_for_word].lower()] = True
    
    # Calculate emission log-probabilities for each tag and word
    for tag in emission_params.keys():
        for emission_word in emission_params[tag].keys(): # Iterate through each word observed for the tag
            if emission_word != 'Total':
                # Calculate and store emission log-probability using np.log
                emission_params[tag][emission_word] = np.log(emission_params[tag][emission_word] / emission_params[tag]['Total'])
    
    # Return the calculated emission probabilities and observed words
    return emission_params, words_observed

# Find max length among all sentences in a list of tags
def find_max_sentence_length(tags):
    max_length = 0  # Initialize the maximum length to 0
    
    # Loop through each sentence in the list of tags
    for sentence in tags:
        # Compare the length of the current sentence to the maximum length
        if len(sentence) > max_length:
            max_length = len(sentence)  # Update the maximum length if necessary

    return max_length

#-------------------------#

# Apply Laplace smoothing to emission probabilities
def smooth_emission_params(emission_params, words_observed, k=0.1):
    smoothed_emission_params = {}
    for label in emission_params:
        smoothed_emission_params[label] = {}
        for word in emission_params[label]:
            smoothed_emission_params[label][word] = emission_params[label][word] + np.log(k / (emission_params[label]['Total'] + k * len(words_observed)))
    return smoothed_emission_params

# Apply Laplace smoothing to transition probabilities
def smooth_transition_params(transition_params, k=0.1):
    smoothed_transition_params = {}
    for prev_label in transition_params:
        smoothed_transition_params[prev_label] = {}
        for next_label in transition_params[prev_label]:
            if next_label != 'Total':
                smoothed_transition_params[prev_label][next_label] = transition_params[prev_label][next_label] + np.log(k / (transition_params[prev_label]['Total'] + k * len(transition_params)))
    return smoothed_transition_params


#-------------------------#

# Viterbi Algorithm implementation
# This function returns a list of dictionaries which correspond to (index=position, key=label)
# with value = tuple (log-prob, prev_tag)
# N value is the maximum length of the sentence
def viterbi_algo(transition_params, emission_params, words_observed, sentence):
    # Initialization
    n = len(sentence)
    memo = []
    memo.append({'Start': (0, '')})  # Initial state

    # Get the list of tags for transition calculation
    tag_list = TAGS
    # Iterate through each position in the sentence
    for j in range(n):
        lowerword = sentence[j].lower()
        memo.append({})  # Initialize the memoization dictionary for this position
        
        # Check if the word is in the training set, else set to '#UNK#'
        word = lowerword if lowerword in words_observed else '#UNK#'


        # Calculate scores for each possible next tag
        for next_tag in tag_list:
            # Get the emission probability for the current word and next tag
            emission_prob = emission_params[next_tag].get(word, -float('inf'))
            
            # Initialize the maximum score and the corresponding previous tag
            max_score = -float('inf')
            new_tag = '' 
            
            # Iterate through each previous tag to calculate transition scores
            for prev_tag in memo[j].keys():
                # Get the Viterbi score from the memo dictionary for the previous tag
    
                viterbi_score = memo[j][prev_tag][0]

                # Get the transition probability from the previous tag to the next tag
                if next_tag in transition_params[prev_tag]:
                    transition_prob = transition_params[prev_tag][next_tag]
                else:
                    transition_prob = -float('inf')
                
                # Calculate the new score based on Viterbi and transition probabilities
                new_score = viterbi_score + emission_prob + transition_prob ####
                
                # Update the maximum score and corresponding previous tag if needed
                if new_score > max_score: max_score = new_score; new_tag = prev_tag
            
            # Only add an entry if it is meaningful for easier debugging
            if new_tag != '':
                memo[j+1][next_tag] = (max_score, new_tag)
                print("memo at pre tag: ", memo)
        
        # Unexpected Transition Scenario where all possible combinations lead to a probability of 0 (log-prob = -inf)
        # Default behavior is to assign 'O' as a label and set the log-prob as the max in the previous step
        if len(list(memo[j+1].keys())) < 1:
            max_log_prob = -float('inf') 
            new_tag = ''
            for prev_tag in memo[j].keys():
                if memo[j][prev_tag][0] > max_log_prob:
                    max_log_prob = memo[j][prev_tag][0] 
                    new_tag = prev_tag
            memo[j+1]['O'] = (max_log_prob, new_tag)
            print('Unexpected Transition Scenario') # occurs when there are no valid transitions from the previous tag to any of the possible next tags. This can happen if the emission probabilities for all possible next tags are very low or if the transition probabilities from the previous tag to all possible next tags are also very low.

    # Termination step
    max_log_prob = -float('inf')
    new_tag = ''
    for prev_tag in transition_params.keys():
        # Get the Viterbi score from the memo dictionary for the previous tag
        viterbi_score = memo[n].get(prev_tag, (-float('inf'),''))[0]
        
        # Get the transition probability from the previous tag to 'Stop' tag
        transition_prob = transition_params.get(prev_tag, {}).get('Stop', -float('inf'))
        
        # Calculate the new score based on Viterbi and transition probabilities
        new_score = viterbi_score + transition_prob
        
        # Update the maximum score and corresponding previous tag if needed
        if new_score > max_log_prob:
            max_log_prob = new_score
            new_tag = prev_tag
    memo.append({'Stop': (max_log_prob, new_tag)})

    # Backtracking to find the sequence of tags
    sequence = []
    current_label = 'Stop'
    for memo_position in range(n+1, 1, -1):
        try:
            current_label = memo[memo_position][current_label][1]
        except KeyError:
            # Handle cases where backtracking fails
            print(f'memo_position: {memo_position}, current_label: {current_label}, sequence: {sequence}')
            print(memo)
            print(sentence)
            break
        sequence.append(current_label)

    return list(reversed(sequence))  # Return the reversed sequence of tags

# Predicts and assigns tags to words in the given input data file and saves the result to an output file.
def predict_tags_and_save(emission_params, transition_params, words_observed, data_input_path, data_output_path):
    print("input",data_input_path,"\noutput",data_output_path)
    # Read unlabelled words from the input data file
    words = open_unlabelled_data(data_input_path)
    # print("words",words)
    
    # Open the output data file for writing
    # with open(data_output_path, 'w', encoding="utf-8") as output_file:
    #     for word_list in words:
    #         # print("word",word_list)
    #         # Predict tags for each word using the Viterbi algorithm
    #         tags = viterbi_algo(transition_params, emission_params, words_observed, word_list)
            
    #         # Write the predicted word and its tag to the output file
    #         for i in range(len(word_list)):
    #             # print(word[i] + ' ' + tags[i] + '\n')
    #             # ??? I am printing something out, but when i try to write into the file, it doesn't write anything???
    #             output_file.write(word_list[i] + ' ' + tags[i] + '\n')
            
    #         # Write a newline character to separate sentences in the output file
    #         # print('\n')
    #         output_file.write('\n')


#____________________TESTING____________________#
# run funtions below

### Tester code for estimate_transition_params ###

# Load training data for words and tags
words, tags = RU_train

# Estimate transition parameters based on the training tags
transition_params = estimate_transition_params(tags)

# Apply Laplace smoothing to transition parameters
# smoothed_transition_params = smooth_transition_params(transition_params)

# Print the list of keys (tags) in the transition parameters dictionary
print("Tags in transition_params:", transition_params)

# Example tag for testing
tag = 'B-negative'

# Print the keys (transitions) in the specified tag's dictionary
print("Transitions for tag", tag, ":", transition_params[tag].keys())

# Print the transition probability from 'Start' to the specified tag
print("Transition probability from 'Start' to", tag, ":", transition_params['Start'][tag])

# Find and print the maximum sentence length in the training tags
print("Maximum sentence length:", find_max_sentence_length(tags))

### End of tester code for estimate_transition_params ###

#-------------------------#

### Tester code for viterbi_algo ###

# Load training data (words and tags) for testing
words, tags = RU_train

# Estimate transition parameters using training tags
transition_params = estimate_transition_params(tags)

# Estimate emission parameters and observed words using training data
emission_params, words_observed = estimate_emission_params_log(words, tags)

# Apply Laplace smoothing to emission parameters
# smoothed_emission_params = smooth_emission_params(emission_params, words_observed)

# Choose a specific sentence index for testing
x = 30

# Apply the Viterbi algorithm to predict tags for the selected sentence
#predict = viterbi_algo(transition_params, emission_params, words_observed, words[x])
def k_viterbi(transition_parameters, emission_parameters, words_observed, sentence, k):
    #Initialisation
    n = len(sentence)
    memo = []
    memo.append({'Start': [(0, [])]})

    label_list = TAGS

    for j in range(n):
        memo.append({})
        #Check if word in training set, else set to #UNK#
        #for some reason, if i use sentence[j].lower(), it will affect the results
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
            #for some reason, when i refactor it with the code at the bottom, the answer changes.
            # entries.sort(key=lambda x: x[0])
            # entries = entries[:k] if len(entries) > k else entries
            # if entries:
            #     memo[j + 1][next_label] = entries

                
        
        #edge case where by some miracle all possible combi lead to probability = 0 (log-prob = -inf) and nothing is recorded
        #Current default is assign 'O' as label and set log-prob as max in prev step (with corresponding label)
        if len(list(memo[j+1].keys())) < 1:
            entries = []
            for prev_label in memo[j].keys():
                prev_entries = memo[j][prev_label]
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
    for prev_label in memo[j].keys():
        prev_entries =  memo[j][prev_label]
        for v, path in prev_entries:
            if prev_label in transition_parameters and 'Stop' in transition_parameters[prev_label]:
                transition_prob = transition_parameters[prev_label]['Stop']
            else:
                transition_prob = -float('inf')
            new_score = v + transition_prob
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


def k_best_viterbi(transition_params, emission_params, words_observed, sentence, k):
    n = len(sentence)
    memo = []
    memo.append({'Start': [(0, [])]})  # Initial state

    tag_list = list(transition_params.keys())
    tag_list.remove('Start')  # Remove 'Start' tag from the list

    for j in range(n):
        lowerword = sentence[j].lower()
        memo.append({})  # Initialize the memoization dictionary for this position
        
        # Check if the word is in the training set, else set to '#UNK#'
        word = lowerword if lowerword in words_observed else '#UNK#'

        # Calculate scores for each possible next tag
        for next_tag in tag_list:
            emission_prob = emission_params[next_tag].get(word, -float('inf'))
            
            best_candidates = []
            for prev_tag in memo[j].keys():
                prev_candidates = memo[j][prev_tag]
                for v, path in prev_candidates:
                    viterbi_score = v

                    if next_tag in transition_params[prev_tag]:
                        transition_prob = transition_params[prev_tag][next_tag]
                    else:
                        transition_prob = -float('inf')

                    new_score = viterbi_score + emission_prob + transition_prob

                    if new_score > -float('inf'):
                        new_path = path.copy()
                        new_path.append(prev_tag)
                        best_candidates.append((new_score, new_path))

            best_candidates.sort(key=lambda x: x[0], reverse=True)
            while len(best_candidates) > k:
                best_candidates.pop()
            if len(best_candidates) > 0:
                memo[j + 1][next_tag] = best_candidates
        
        if len(list(memo[j+1].keys())) < 1:
            best_candidates = []
            for prev_tag in memo[j].keys():
                prev_candidates = memo[j][prev_tag]
                for v, path in prev_candidates:
                    new_path = path.copy()
                    new_path.append(prev_tag)
                    best_candidates.append((v, new_path))

            best_candidates.sort(key=lambda x: x[0], reverse=True)
            while len(best_candidates) > k:
                best_candidates.pop()
            if len(best_candidates) > 0:
                memo[j + 1]['O'] = best_candidates
            print('Unexpected Transition Scenario')

    best_candidates = []
    for prev_tag in memo[n].keys():
        prev_candidates = memo[n][prev_tag]
        for v, path in prev_candidates:
            viterbi_score = v
            transition_prob = transition_params.get(prev_tag, {}).get('Stop', -float('inf'))
            new_score = viterbi_score + transition_prob

            if new_score > -float('inf'):
                new_path = path.copy()
                new_path.append(prev_tag)
                best_candidates.append((new_score, new_path))

    best_candidates.sort(key=lambda x: x[0], reverse=True)
    while len(best_candidates) > k:
        best_candidates.pop()

    memo.append({'Stop': best_candidates})

    results = []
    for _, path in memo[-1]['Stop']:
        sequence = path[1:]
        sequence.reverse()
        results.append(sequence)

    return results

k_predict = k_viterbi(transition_params,emission_params,words_observed,words[x],6)
print("k_predict", k_predict)


# Assert that the length of predicted tags matches the length of actual tags
#assert len(tags[x]) == len(predict)

# Print the actual tags and predicted tags for comparison
print("Actual Tags:", tags[x])
#print("Predicted Tags:", predict)

### End of tester code for viterbi_algo ###

#-------------------------#

### Writing predictions to output files ###

words, tags = ES_train
transition_params = estimate_transition_params(tags)
emission_params, words_observed = estimate_emission_params_log(words, tags)
predict_tags_and_save(emission_params, transition_params, words_observed, 'ES/dev.in', 'ES/dev.p2.out')

words, tags = RU_train
transition_params = estimate_transition_params(tags)
emission_params, words_observed = estimate_emission_params_log(words, tags)
predict_tags_and_save(emission_params, transition_params, words_observed, 'RU/dev.in', 'RU/dev.p2.out')

### End of writing predictions to output files ###