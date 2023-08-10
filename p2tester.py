from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown
from part1_functions import produce_tag
import numpy as np

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
            if j !='Total':
                transition_params[i][j] = np.log(transition_params[i][j]/transition_params[i]['Total'])
    return transition_params


def viterbi_algorithm(sentence, transition_params, emission_params):
    sentence_length = len(sentence)
    num_tags = len(transition_params)
    
    # Initialize the Viterbi matrix and backpointers
    viterbi_matrix = np.full((num_tags, sentence_length), -np.inf)
    backpointers = np.zeros((num_tags, sentence_length), dtype=int)
    
    # Initialize the first column with START tag probabilities
    for tag_index, tag in enumerate(transition_params):
        viterbi_matrix[tag_index, 0] = transition_params['START'].get(tag, -np.inf)
    
    # Fill in the Viterbi matrix and backpointers
    for j in range(1, sentence_length):
        for curr_tag_index, curr_tag in enumerate(transition_params):
            max_score = -np.inf
            max_prev_tag_index = 0
            for prev_tag_index, prev_tag in enumerate(transition_params):
                transition_score = viterbi_matrix[prev_tag_index, j-1] + transition_params[prev_tag].get(curr_tag, -np.inf)
                emission_score = emission_params[curr_tag].get(sentence[j], -np.inf)
                score = transition_score + emission_score
                if score > max_score:
                    max_score = score
                    max_prev_tag_index = prev_tag_index
            viterbi_matrix[curr_tag_index, j] = max_score
            backpointers[curr_tag_index, j] = max_prev_tag_index
    
    # Find the best sequence using backpointers
    best_sequence = []
    curr_tag_index = viterbi_matrix[:, -1].argmax()
    for j in range(sentence_length - 1, -1, -1):
        best_sequence.insert(0, list(transition_params.keys())[curr_tag_index])
        curr_tag_index = backpointers[curr_tag_index, j]
    
    return best_sequence

# Example usage
sentence = ['hello', 'konnichiwa']
demo_emission = {'O': {'hello': 0.05555, 'konnichiwa': 0.003333333},
    'B-Positive': {'somethingsomething': -7.2442, 'nothingnothing': -2.33333}
}

labels = get_labels_only(ES_train)
ES_tagset, ES_eset = produce_tag(ES_train, ES_dev_in, TAGS)
transition_params = estimate_transition_parameters_v2(labels)
new_transition_params = log_transition_conversion(transition_params)

best_sequence = viterbi_algorithm(sentence, transition_params, ES_eset)
print(best_sequence)

