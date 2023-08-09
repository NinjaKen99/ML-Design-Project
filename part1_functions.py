# Importing libraries
from collections import defaultdict
from copy import deepcopy

# Imports from another file
from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown

# Functions

#-----------------------------#
def calculate_emission_counts(data):  # Part a: For training
    
    # Initialize dictionaries to store tag counts and word emission counts
    count_emission = {}  # Stores the count of each word for each tag
    count_tag = {'O': 0, 'B-positive': 0, 'B-neutral': 0, 'B-negative': 0, 'I-positive': 0, 'I-neutral': 0, 'I-negative': 0}
    new = {'O': 0, 'B-positive': 0, 'B-neutral': 0, 'B-negative': 0, 'I-positive': 0, 'I-neutral': 0, 'I-negative': 0}
    
    # Iterate through each line in the data (each line corresponds to a word and its tag)
    for line in data:
        # Account for gaps in file (Skip empty lines)
        if line != "\n":
            # Split line into word and tag components
            word_tag_pair = line.split(" ")
            word = ''.join([character for character in word_tag_pair[:-1]])  # Extract word
            tag = word_tag_pair[-1].strip()  # Extract tag and remove newline
            
            # If the word is encountered for the first time, initialize its emission counts
            if word not in count_emission:
                count_emission[word] = deepcopy(new)
            
            # Increment the emission count for the specific tag of the word
            count_emission[word][tag] += 1
            
            # Increment the total count for the specific tag
            count_tag[tag] += 1
    
    # Return the dictionaries containing emission counts and tag counts
    # print("count_emission:", count_emission)
    # print("count_tag:", count_tag)
    return count_emission, count_tag


#-----------------------------#

# def find_k (data_train, data_test): # part b
#     # Split file by line
#     dataset_train = data_train.split("\n")
#     dataset_test = data_test.split("\n")
#     word_list = []
#     k = 1 # initialise k to 1
#     for line in dataset_train:
#         if (line != ""):
#             # Split line into word and tag
#             pair = line.split(" ")
#             word = pair[0]
#             if word not in word_list:
#                 word_list.append(word)
#     for line in dataset_test:
#         if (line != ""):
#             # Split line into word and tag
#             pair = line.split(" ")
#             word = pair[0]
#             if word not in word_list:
#                 k += 1
#     return k

#-----------------------------#
  
def estimate_emission_parameters(data, k=1): # Modified for part c
    
    count_emission, count_tag = calculate_emission_counts(data)
    emission_parameters = {}
    tag_probability = {'O':0,'B-positive':0,'B-neutral':0,'B-negative':0,'I-positive':0,'I-neutral':0,'I-negative':0} # Stores the tag probability of each word

    # Create entry for "#UNK#" in emission_params
    emission_parameters["#UNK#"] = deepcopy(tag_probability)
    
    for tag in count_tag.keys():
        # denominator: used to normalise the emission probabilities
        denominator = k + count_tag[tag]
        for word in count_emission.keys():
            if (word not in emission_parameters):
                emission_parameters[word] = deepcopy(tag_probability)
            numerator = count_emission[word][tag] # numerator: represents the count of occurences of a specific word associated with a particular tag in training data
            emission_parameters[word][tag] = numerator/denominator
        emission_parameters["#UNK#"][tag] = k/denominator
    # print("emission parameters: ",emission_parameters)
    return emission_parameters

#-----------------------------#

# def produce_tag(data_train, data_test, TAGS): # part c: For training
#     # Create dictionary for storing data
#     tag_dict = {}
#     word_dict = {}
#     k = find_k(data_train, data_test)
#     # Create emission parameter tracking dictionary
#     for tag in TAGS:
#         tag_dict[tag] = estimate_emission_parameter_v3(data_train, k, tag)
#         # Create a word list using a dictionary
#         for key in tag_dict[tag].keys(): 
#             word_dict[key] = None
#     # Loop through word list to assign tag
#     for word in word_dict.keys():
#         emission_parameter = 0
#         y_star = None
#         for tag in TAGS:
#             try: # To ignore error if dictionary lacks the key
#                 if (tag_dict[tag][word] > emission_parameter):
#                     emission_parameter = tag_dict[tag][word]
#                     y_star = tag
#             except:
#                 pass
#         word_dict[word] = y_star # Predicted Tag for each training word
#     return word_dict

#-----------------------------#

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

# Sentiment analysis: For testing
# def sentiment_analysis(file, emission_parameters, gold_tags):
#     word_tag_list = [] # combine word and its tag into a string, then append to a list
#     total_correct_predictions = 0
#     total_predicted_entities = 0
#     total_gold_entities = count_gold_entities(gold_tags)
#     print(total_gold_entities)
#     # Get lines for each file
#     data_test = file.split("\n")
#     data_check = gold_tags.split("\n")
#     # Entity Tracker
#     predict_back = None
#     entity_wrong = False
    
#     for word, gold_line in zip(data_test, data_check):
#         word_tag_pair = ""
#         # Retrieve tag for word
#         if word != "\n":
#             word = word.strip()
            
#             # try : 
#             #     tag_for_word = max(emission_parameters[word])
#             # except:
#             #     tag_for_word = max(emission_parameters[unknown])
            
#             if word not in emission_parameters:
#                 tag_for_word = max(emission_parameters["#UNK#"], key = emission_parameters["#UNK#".get])
#             else:
#                 tag_for_word = max(emission_parameters[word], key = emission_parameters[word].get)
                        
#             # Assign tag to word
#             word_tag_pair = word + " " + tag_for_word
#             word_tag_list.append(word_tag_pair)
#             # Get golden tag
#             pair = gold_line.split(" ")
#             gold_tag = pair[1]
#             # Count number of predictions
#             if (predict_back == None and tag_for_word != "O"):
#                 total_predicted_entities += 1
#             elif (predict_back != None and tag_for_word != "O"):
#                 if (tag_for_word[2:] != predict_back[2:] or tag_for_word[0] == "B"):
#                     total_predicted_entities += 1
#             # Ignore if O
#             if (gold_tag == "O"):
#                 if tag_for_word != gold_tag:
#                     entity_wrong = True
#                     predict_back = tag_for_word
#                 else:
#                     entity_wrong = False
#                     predict_back = None
#             else:
#                 if (predict_back == None):
#                     if (tag_for_word == gold_tag):
#                         total_correct_predictions += 1
#                     else:
#                         entity_wrong = True
#                     predict_back = tag_for_word
#                 else:
#                     if (tag_for_word[0] == "B" or tag_for_word[2:] != predict_back[2:]): # check if different entity
#                         if (tag_for_word == gold_tag):
#                             total_correct_predictions += 1
#                             entity_wrong = False
#                         else:
#                             entity_wrong = True
#                     elif (not entity_wrong):
#                         if (tag_for_word != gold_tag):
#                             entity_wrong = True
#                             total_correct_predictions -= 1
#                     predict_back = tag_for_word
                
#         else: word_tag_list.append("") # Recreate empty lines
    
#     print(total_predicted_entities)        
#     p = precision(total_correct_predictions, total_predicted_entities)
#     r = recall(total_correct_predictions, total_gold_entities)
#     f = f_score(p, r)
        
#     return word_tag_list, p, r, f

#def count_gold_entities(data):
#    number = 0
#    previous_tag = None
#    gold_data = data.split("\n")
#    for line in gold_data:
#        if (line != ""):
#            pair = line.split(" ")
#            tag = pair[1]
#            match tag:
#                case "O":
#                    previous_tag = None
#                case "I-positive" | "I-negative" | "I-neutral":
#                    if (previous_tag == None): 
#                        number += 1
#                        previous_tag = tag
#                    elif (previous_tag[2:] == tag[2:]):
#                        pass
#                    else:
#                        number += 1
#                        previous_tag = tag
#                case _:
#                    number += 1
#                    previous_tag = tag
#        else:
#            previous_tag = None
#    return number

def sentiment_analysis(data, emission_parameters,gold_tags):
    word_tag_list = [] # combine word and its tag into a string, then append to a list
    total_correct_predictions = 0
    total_predicted_entities = 0
    total_gold_entities = 0
    # Tracker
    prev_head = None
    prev_sent = None
    entity_correct = None
    prev_gold_head = None
    prev_gold_sent = None
    
    for line, gold_tag in zip(data,gold_tags):
        word = line.strip()
        word_tag_pair = ""
        if (line != "\n"):
            if word not in emission_parameters: 
                tag_for_word = max(emission_parameters["#UNK#"],key = emission_parameters["#UNK#"].get)
            else:
                tag_for_word = max(emission_parameters[word],key = emission_parameters[word].get)
            word_tag_pair = word + " " + tag_for_word
            # Entity Count
            if tag_for_word != "O":
                cur_head = tag_for_word[0]
                cur_sent = tag_for_word[2:]
                if cur_head == "B" or (cur_head == "I" and prev_head == "O") or (cur_head == "I" and prev_head != "O" and cur_sent != prev_sent):
                    total_predicted_entities += 1
                prev_head = cur_head
                prev_sent = cur_sent
            else:
                prev_head = "O"
                prev_sent = None
        word_tag_list.append(word_tag_pair)
        # Count Correct
        if (gold_tag != "\n"):
            gold_label = gold_tag.strip().split(" ")[1]
            if gold_label != "O":
                gold_head = gold_label[0]
                gold_sent = gold_label[2:]
                # if new entity
                if gold_head == "B" or (gold_head == "I" and prev_gold_head == "O") or (gold_head == "I" and prev_gold_head != "O" and gold_sent != prev_gold_sent):
                    total_gold_entities += 1
                    if gold_label == tag_for_word: 
                        total_correct_predictions += 1
                        entity_correct = True
                    else:
                        entity_correct = False
                elif gold_head == "I":
                    if gold_label != tag_for_word and entity_correct == True: 
                        total_correct_predictions -= 1
                        entity_correct = False
                prev_gold_head = gold_head
                prev_gold_sent = gold_sent
            else:
                prev_gold_head = "O"
                prev_gold_sent = None
                entity_correct = None
    
    print(total_predicted_entities)
    print(total_gold_entities)
    print(total_correct_predictions)
    p = precision(total_correct_predictions, total_predicted_entities)
    r = recall(total_correct_predictions, total_gold_entities)
        # f = f_score(p, r)
        
    # return word_tag_list, p, r, f
    return word_tag_list, p, r

#____________________TESTING____________________#
# run funtions below

# tagset: y* assigned to each word
# eset: emission parameter value for each tag, for each word
# ES_tagset, ES_eset = estimate_emission_parameters(ES_train, ES_dev_in, TAGS)
# RU_tagset, RU_eset = estimate_emission_parameters(RU_train, RU_dev_in, TAGS)

ES_emission_parameters_train = estimate_emission_parameters(ES_train)
RU_emission_parameters_train = estimate_emission_parameters(RU_train)
    

ES_dev_out, ES_precision, ES_recall = sentiment_analysis(ES_dev_in, ES_emission_parameters_train, ES_dev_out)
RU_dev_out, RU_precision, RU_recall= sentiment_analysis(RU_dev_in, RU_emission_parameters_train, RU_dev_out)

print(ES_precision, ES_recall)
print(RU_precision, RU_recall)

# Writing to Files
with open('ES/dev.p1.out.txt', 'w', encoding="utf-8") as f:
   f.write('\n'.join(ES_dev_out))
with open('RU/dev.p1.out.txt', 'w', encoding="utf-8") as f:
   f.write('\n'.join(RU_dev_out))

# Reading lines from dev.p1.out files
with open('ES/dev.p1.out.txt', 'r', encoding="utf-8") as f:
    ES_p1_dev_out = f.readlines()
with open('RU/dev.p1.out.txt', 'r', encoding="utf-8") as f:
    RU_p1_dev_out = f.readlines()
print(calculate_emission_counts(ES_p1_dev_out)[1])
print(calculate_emission_counts(RU_p1_dev_out)[1])