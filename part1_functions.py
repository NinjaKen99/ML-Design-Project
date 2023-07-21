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
                if (word not in counter.keys()):
                    counter[word] = 1
                else: counter[word] += 1
    # new dictionary with emission parameter for each word that exists
    emission_parameters = {"Unknown": counter["Unknown"]/(counter["Count"] + counter["Unknown"])}
    total = counter["Count"]
    k = counter["Unknown"]
    for keys, values in counter.items():
        if (keys != "Count" or keys != "Unknown"):
            emission_parameters[keys] = counter[keys] / (counter["Count"] + counter["Unknown"])        
    return emission_parameters # return dictionary with all emission parameters

def produce_tag(data, TAGS):
    tag_dict = {}
    word_dict = {}
    for tag in TAGS:
        tag_dict[tag] = estimate_emission_parameter_v3(data, tag)
        for key in tag_dict[tag].keys():
            word_dict[key] = None
    for word in word_dict.keys():
        emission_parameter = 0
        y_star = None
        for tag in TAGS:
            try:
                if (tag_dict[tag][word] > emission_parameter):
                    y_star = tag
            finally:
                pass
        word_dict[word] = y_star
    return word_dict

#____________________TESTING____________________#
# run funtions below

print(produce_tag(RU_train, TAGS))