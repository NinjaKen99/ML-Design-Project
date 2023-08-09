from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown
import numpy as np

#extract only the labels
def get_labels_only(data):
    labels = []
    sent_label = []
    dataset = data.split("\n")
    for line in dataset:
        if line != '':
            pair = line.split(' ')
            sent_label.append(pair[1])
        else:
            if sent_label:  # Check if sent_label is not empty before appending
                labels.append(sent_label[:])
                sent_label = []
    return labels

def estimate_transition_parameters_v2(labels):
    # Initialize transition parameters with START and STOP
    transition_params = {'START': {'Total': 0}, 'STOP': {'Total': 0}}
    
    # Iterate through each sentence
    for i in range(len(labels)):
        # Initialize the previous label as START
        
        # Iterate through each label in the sentence
        for j in range(len(labels[i])):
            # If it's the first label in the sentence
            if j == 0:
                prev_label = 'START'
            elif j == len(labels[i])-1:
                prev_label = 'STOP'
            else:
                # Set the previous label to the label before the current one
                prev_label = labels[i][j-1]
            
            # Check if the previous label is already in transition_params
            if prev_label not in transition_params.keys():
                transition_params[prev_label] = {'Total': 0}
            
            # Get the current count and update it
            curr_label = labels[i][j]
            count = transition_params[prev_label].get(curr_label, 0)
            transition_params[prev_label][curr_label] = count + 1
            transition_params[prev_label]['Total'] += 1

    # Return the calculated transition parameters
    return transition_params

def log_transition_conversion(transition_params):
    for i in transition_params.keys():
        for j in transition_params[i].keys():
            transition_params[i][j] = np.log(transition_params[i][j]/transition_params[i]['Total'])

labels = get_labels_only(ES_train)
print(estimate_transition_parameters_v2(labels))

#-------------------------------------------------------------------------------------------------------------------------

def initialize_matrix(num_tags,sentence_length):
    init = []
    for i in range (num_tags):
        init.append([-99999]*sentence_length)
    print("init",init)
    return init

def viterbi_algorithm(sentence, transition_params, emission_params):
    num_tags = len(transition_params)
    sentence_length = len(sentence)
    viterbi_matrix = initialize_matrix(num_tags, sentence_length)
    
    # Initialization
    for tag in range(num_tags):
        viterbi_matrix[tag][0] = transition_params['START'][tag] + emission_params[tag][sentence[0]]
    
    # Recursive Step
    for j in range(1, sentence_length):
        for curr_tag in range(num_tags):
            for prev_tag in range(num_tags):
                prob = viterbi_matrix[prev_tag][j - 1] + transition_params[prev_tag][curr_tag] + emission_params[curr_tag][sentence[j]]
                if prob > viterbi_matrix[curr_tag][j]:
                    viterbi_matrix[curr_tag][j] = prob
    
    # Termination
    for tag in range(num_tags):
        viterbi_matrix[tag][sentence_length - 1] += transition_params[tag]['STOP']
    
    # Backtracking to find the most likely sequence
    best_sequence = []
    curr_tag = viterbi_matrix.argmax(axis=0)[-1]
    for j in range(sentence_length - 1, -1, -1):
        best_sequence.insert(0, curr_tag)
        curr_tag = viterbi_matrix[:, j].argmax()
    
    return best_sequence






#print(log_transition_conversion(estimate_transition_parameters_v2))




