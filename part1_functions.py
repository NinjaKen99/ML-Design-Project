# Imports from another file
from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown

# Functions
def estimate_emission_parameter_v1(data, WORD, TAG): # part a
    dataset = data.split("\n")
    counter = {"Count": 0, "Word": 0}
    for line in dataset:
        pair = line.split(" ")
        word, tag = pair[0], pair[1]
        if (tag == TAG):
            counter["Count"] += 1
            if (word == WORD):
                counter["Word"] += 1
    result = counter["Word"]/counter["Count"]
    return result

def estimate_emission_parameter_v2(data, WORD, TAG): # part b
    dataset = data.split("\n")
    counter = {"Count": 0, "Word": 0, "Unknown": 1}
    for line in dataset:
        pair = line.split(" ")
        word, tag = pair[0], pair[1]
        if (tag == TAG):
            counter["Count"] += 1
            if (word == unknown):
                counter["Unknown"] += 1
            elif (word == WORD):
                counter["Word"] += 1
    if (WORD == unknown):
        return counter["Unknown"] / (counter["Count"] + counter["Unknown"])
    return counter["Word"] / (counter["Count"] + counter["Unknown"])

def estimate_emission_parameter_v3(data, TAG): # modified for part c
    dataset = data.split("\n")
    counter = {"Count": 0, "Unknown": 1}
    for line in dataset:
        pair = line.split(" ")
        word, tag = pair[0], pair[1]
        if (tag == TAG):
            counter["Count"] += 1
            if (word == unknown):
                counter["Unknown"] += 1
            else:
                counter[word] += 1
    return counter

# Precision function
def precision(correctly_predicted_entities, total_predicted_entities):
    return correctly_predicted_entities / total_predicted_entities

# Recall function
def recall(correctly_predicted_entities, gold_entities):
    # gold entities refers to the entities with correct labels (from dev.out)
    return correctly_predicted_entities / gold_entities

# F_score function
def f_score(precision, recall):
    return 2 / ((1 / precision) + (1 / recall))

#____________________TESTING____________________#
# run funtions below