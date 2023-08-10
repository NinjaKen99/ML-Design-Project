from fixed_parameters import TAGS, NUMBER_OF_TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import ES_dev_p2_out
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

def unlabelled_Observation(data):
    observation = []
    sent_observ = []
    dataset = data.split("\n")
    for line in dataset:
        if line != '':
            sent_observ.append(line)
        else:
            if sent_observ:  # Check if sent_label is not empty before appending
                observation.append(sent_observ[:])
                sent_observ = []
    return observation

def log_emission_conversion(emission_params):
    log_emission_params = {}
    for tag, word_probs in emission_params.items():
        log_emission_params[tag] = {}
        for word, prob in word_probs.items():
            log_emission_params[tag][word] = np.log(prob)
    return log_emission_params

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
            if j != 'Total':
                transition_params[i][j] = np.log(transition_params[i][j]/transition_params[i]['Total'])
    return transition_params

def initialize_matrix(num_tags,sentence_length):
    init = []
    for i in range (num_tags):
        init.append([-99999]*sentence_length)
    #print("init",init)
    return init

def extract_vocabulary(emission_params):
    vocabulary = set()
    for tag_dict in emission_params.values():
        for word in tag_dict.keys():
            vocabulary.add(word)
    return vocabulary
    
def viterbi_all_sequences(vocab, sentence, transition_params, emission_params):
    index_to_tags = {0:'B-positive',1:'B-negative',2:'B-neutral',3:'I-positive',4:'I-negative',5:'I-neutral',6:'O'}
    num_tags = NUMBER_OF_TAGS
    sentence_length = len(sentence)
    viterbi_matrix = initialize_matrix(num_tags, sentence_length)
    
    # Initialization
    for j in range(sentence_length):  
        for curr_tag in range(num_tags):
            if sentence[j] in vocab and index_to_tags[curr_tag] in transition_params['START']:
                try:
                    viterbi_matrix[curr_tag][j] = transition_params['START'][index_to_tags[curr_tag]] + emission_params[index_to_tags[curr_tag]][sentence[j]]
                except:
                    viterbi_matrix[curr_tag][j] = -99999
    
    # Recursive Step
    for j in range(1, sentence_length):
        for curr_tag in range(num_tags):
            for prev_tag in range(num_tags):
                if sentence[j] in vocab and index_to_tags[curr_tag] in transition_params[index_to_tags[prev_tag]] and sentence[j] in emission_params[index_to_tags[curr_tag]]:
                    prob = viterbi_matrix[prev_tag][j - 1] + transition_params[index_to_tags[prev_tag]][index_to_tags[curr_tag]] + emission_params[index_to_tags[curr_tag]][sentence[j]]
                    if prob > viterbi_matrix[curr_tag][j]:
                        viterbi_matrix[curr_tag][j] = prob
    
    # Termination
    for tag in range(num_tags):
        if 'STOP' in transition_params.get(index_to_tags[tag], {}):
            viterbi_matrix[tag][sentence_length - 1] += transition_params[index_to_tags[tag]]['STOP']
        else:
            viterbi_matrix[tag][sentence_length - 1] = -np.inf 
    
    # Collect all sequences
    viterbi_matrix = np.array(viterbi_matrix)  
    all_sequences = []
    
    for curr_tag in range(num_tags):
        score = viterbi_matrix[curr_tag][sentence_length - 1]
        sequence = []
        curr = curr_tag
        
        for j in range(sentence_length - 1, -1, -1):
            sequence.insert(0, index_to_tags[curr])
            curr = np.argmax(viterbi_matrix[:, j])
        
        all_sequences.append((score, sequence))
    
    # Sort sequences by score
    all_sequences.sort(reverse=True, key=lambda x: x[0])
    
    return [seq for _, seq in all_sequences]
#-------------------------------------------------------------------------------------------------------------------------
labels = get_labels_only(ES_train)
transition_param = estimate_transition_parameters_v2(labels)
log_transition_param = log_transition_conversion(transition_param)

ES_tagset, ES_eset = produce_tag(ES_train, ES_dev_in, TAGS)
log_emission_param = log_emission_conversion(ES_eset)
dev_in_observation = unlabelled_Observation(ES_dev_in)
emission_vocab = extract_vocabulary(log_emission_param)

modified_dev = dev_in_observation[:2]

k = 2
with ES_dev_p2_out as f_out: # change syntax to same as part 1 writing
    for i in modified_dev:
        k_sequence = viterbi_all_sequences(emission_vocab,i,log_transition_param,log_emission_param)
        print("\n")
        print("############k_sequences:############", k_sequence)

        # for seq in k_sequence:
        #     for j in range(len(i)):
        #         f_out.write(i[j] + ' ' + sequence[j] + '\n')
        #     f_out.write('\n')
