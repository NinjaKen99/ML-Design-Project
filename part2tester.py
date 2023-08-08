from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown
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

def read_labelled(file_name):
    observations = []
    labels = []
    with open(file_name, 'r', encoding = 'utf-8') as f:
        temp_o = []
        temp_l = []
        for line in f:
            if line.isspace():
                observations.append(temp_o[:])
                labels.append(temp_l[:])
                temp_o.clear()
                temp_l.clear()
                #print("clearing the space")
            else:
                observation, label = line.strip().rsplit(maxsplit=1)
                temp_o.append(observation)
                temp_l.append(label)
    return observations, labels


def estimate_transition_parameters(labels):
    transition_parameters = {'START': {'#TOTAL#': 0}, 'STOP': {'#TOTAL#': 0}}
    #Count transitions
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if j == 0:
                prev_label = 'START'
            else:
                prev_label = labels[i][j-1]
            
            if prev_label not in transition_parameters.keys():
                transition_parameters[prev_label] = {'#TOTAL#': 0}
                
            curr_count = transition_parameters[prev_label].get(labels[i][j], 0)
            transition_parameters[prev_label][labels[i][j]] = curr_count + 1
            transition_parameters[prev_label]['#TOTAL#'] += 1
        
        curr_count = transition_parameters[labels[i][-1]].get('STOP', 0)
        transition_parameters[labels[i][-1]]['STOP'] = curr_count + 1
        transition_parameters[labels[i][-1]]['#TOTAL#'] += 1

    #Calc log-probs        
    for i in transition_parameters.keys():
        for j in transition_parameters[i].keys():
            if j != '#TOTAL#':
                transition_parameters[i][j] = np.log(transition_parameters[i][j] / transition_parameters[i]['#TOTAL#'])
    
    return transition_parameters

obs,labels = read_labelled('ES/train')
labell = get_labels_only(ES_train)

#print(labels)
print(estimate_transition_parameters(labell))
