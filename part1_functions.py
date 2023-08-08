# Imports from another file
from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown

# Functions
# def estimate_emission_parameter_v1(data, WORD, TAG): # part a
#     dataset = data.split("\n")
#     counter = {"Count": 0, "Word": 0}
#     for line in dataset:
#         pair = line.split(" ")
#         word, tag = pair[0], pair[1]
#         if (tag == TAG):
#             counter["Count"] += 1
#             if (word == WORD):
#                 counter["Word"] += 1
#     result = counter["Word"]/counter["Count"]
#     return result

# def estimate_emission_parameter_v2(data, WORD, TAG): # part b
#     dataset = data.split("\n")
#     counter = {"Count": 0, "Word": 0, "Unknown": 1}
#     for line in dataset:
#         pair = line.split(" ")
#         word, tag = pair[0], pair[1]
#         if (tag == TAG):
#             counter["Count"] += 1
#             if (word == unknown):
#                 counter["Unknown"] += 1
#             elif (word == WORD):
#                 counter["Word"] += 1
#     if (WORD == unknown):
#         return counter["Unknown"] / (counter["Count"] + counter["Unknown"])
#     return counter["Word"] / (counter["Count"] + counter["Unknown"])

# def estimate_emission_parameter_v3(data, TAG): # modified for part c
#     dataset = data.split("\n")
#     counter = {"Count": 0, "Unknown": 1}
#     for line in dataset:
#         pair = line.split(" ")
#         word, tag = pair[0], pair[1]
#         if (tag == TAG):
#             counter["Count"] += 1
#             if (word == unknown):
#                 counter["Unknown"] += 1
#             else:
#                 counter[word] += 1
#     return counter

#--------------------------------------------#


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

# Sentiment analysis
def sentiment_analysis(file, emission_parameters,gold_tags):
    word_tag_list = [] # combine word and its tag into a string, then append to a list
    total_correct_predictions = 0
    total_predicted_entities = 0
    total_gold_entities = 0
    
    for line, gold_tag in zip(file,gold_tags):
        word = line.strip()
        word_tag_pair = ""
        if line != "\n":
            if line == "\n":
                tag_for_word = max(emission_parameters[word],key=emission_parameters[word].get)
            else:
                tag_for_word = max(emission_parameters["#UNK"],key=emission_parameters["#UNK"].get)
            word_tag_pair = word + " " + tag_for_word
        word_tag_list.append(word_tag_pair)
        
        if tag_for_word == gold_tag:
            total_correct_predictions += 1
        if tag_for_word != "O":
            total_predicted_entities += 1
        if gold_tag != "O":
            total_gold_entities += 1
            
        p = precision(total_correct_predictions, total_predicted_entities)
        r = recall(total_correct_predictions, total_gold_entities)
        f = f_score(p, r)
        
    return word_tag_list, p, r, f

#____________________TESTING____________________#
# run funtions below