# Imports from another file
from fixed_parameters import TAGS
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import Invalid_Word as unknown

# Functions
def estimate_emission_parameter_v1(data, WORD, TAG): # Part a: For training
    # Split file by line
    dataset = data.split("\n")
    # Set up dictionary for counting
    counter = {"Count": 0, "Word": 0}
    
    for line in dataset:
        # Account for gaps in file (Skip)
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word, tag = pair[0], pair[1]
            # Perform Counting
            if (tag == TAG):
                counter["Count"] += 1
                if (word == WORD):
                    counter["Word"] += 1
    # Calculate emission parameter
    result = counter["Word"]/counter["Count"]
    return result

def find_k (data_train, data_test): # part b
    # Split file by line
    dataset_train = data_train.split("\n")
    dataset_test = data_test.split("\n")
    word_list = []
    k = 1 # initialise k to 1
    for line in dataset_train:
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word = pair[0]
            if word not in word_list:
                word_list.append(word)
    for line in dataset_test:
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word = pair[0]
            if word not in word_list:
                k += 1
    return k
            
def estimate_emission_parameter_v3(data, k, TAG): # Modified for part c
    # Split file by line
    dataset = data.split("\n")
    # Set up dictionary for counting and emission parameter for each word that exists
    counter = {"Count": 0}
    emission_parameters = {}
    for line in dataset:
        # Account for gaps in file (Skip)
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word, tag = pair[0], pair[1]
            # Perform Counting
            if (tag == TAG): # Count for y
                counter["Count"] += 1
                if (word not in counter.keys()): # Count for all x
                    counter[word] = 1 # Add entry if not exist
                    emission_parameters[word] = None
                else: counter[word] += 1
    total = counter["Count"] # Count(y)
    # Emission parameters for words in train
    for keys in emission_parameters.keys():
        emission_parameters[keys] = counter[keys] / (total + k)   
    # Emission parameter for unknown words
    emission_parameters[unknown] = k / (total + k)
    return emission_parameters # return dictionary with all emission parameters

def produce_tag(data_train, data_test, TAGS): # part c: For training
    # Create dictionary for storing data
    tag_dict = {}
    word_dict = {}
    k = find_k(data_train, data_test)
    # Create emission parameter tracking dictionary
    for tag in TAGS:
        tag_dict[tag] = estimate_emission_parameter_v3(data_train, k, tag)
        # Create a word list using a dictionary
        for key in tag_dict[tag].keys(): 
            word_dict[key] = None
    # Loop through word list to assign tag
    for word in word_dict.keys():
        emission_parameter = 0
        y_star = None
        for tag in TAGS:
            try: # To ignore error if dictionary lacks the key
                if (tag_dict[tag][word] > emission_parameter):
                    emission_parameter = tag_dict[tag][word]
                    y_star = tag
            except:
                pass
        word_dict[word] = y_star # Predicted Tag for each training word
    return word_dict

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
def sentiment_analysis(file, emission_parameters,gold_tags):
    word_tag_list = [] # combine word and its tag into a string, then append to a list
    total_correct_predictions = 0
    total_predicted_entities = 0
    total_gold_entities = count_gold_entities(gold_tags)
    # Get lines for each file
    data_test = file.split("\n")
    data_check = gold_tags.split("\n")
    # Entity Tracker
    predict_back = None
    
    for word, gold_line in zip(data_test, data_check):
        word_tag_pair = ""
        # Retrieve tag for word
        if word != "":
            try : 
                tag_for_word = emission_parameters[word]
            except:
                tag_for_word = emission_parameters[unknown]
            # Assign tag to word
            word_tag_pair = word + " " + tag_for_word
            word_tag_list.append(word_tag_pair)
            # Get golden tag
            pair = gold_line.split(" ")
            gold_tag = pair[1]
            # Ignore if O
            if (tag_for_word != "O"):
                # Ignore if same entity (predicted)
                if (tag_for_word != predict_back):
                    if tag_for_word == gold_tag:
                        total_correct_predictions += 1
                    total_predicted_entities += 1
                    predict_back = tag_for_word
            else: predict_back = None
                
        else: word_tag_list.append("") # Recreate empty lines
            
    p = precision(total_correct_predictions, total_predicted_entities)
    r = recall(total_correct_predictions, total_gold_entities)
    f = f_score(p, r)
        
    return word_tag_list, p, r, f

def count_gold_entities(data):
    number = 0
    previous_tag = None
    gold_data = data.split("\n")
    for line in gold_data:
        if (line != ""):
            pair = line.split(" ")
            tag = pair[1]
            if (tag == "O"):
                previous_tag = None
            elif (tag != previous_tag):
                number += 1
                previous_tag = tag
    return number

#____________________TESTING____________________#
# run funtions below

ES_tagset = produce_tag(ES_train, ES_dev_in, TAGS)
RU_tagset = produce_tag(RU_train, RU_dev_in, TAGS)

# ES_dev_out, ES_precision, ES_recall, ES_f_score = sentiment_analysis(ES_dev_in, ES_tagset, ES_dev_out)
# RU_dev_out, RU_precision, RU_recall, RU_f_score= sentiment_analysis(RU_dev_in, RU_tagset, RU_dev_out)

# print(ES_precision, ES_recall, ES_f_score)
# print(RU_precision, RU_recall, RU_f_score)

# # Writing to Files
# with open('ES/dev.p1.out.txt', 'w', encoding="utf-8") as f:
#    f.write('\n'.join(ES_dev_out))
# with open('RU/dev.p1.out.txt', 'w', encoding="utf-8") as f:
#    f.write('\n'.join(RU_dev_out))
   
# # Reading lines from dev.p1.out files
# with open('ES/dev.p1.out.txt', 'r', encoding="utf-8") as f:
#     ES_p1_dev_out = f.readlines()
# with open('RU/dev.p1.out.txt', 'r', encoding="utf-8") as f:
#     RU_p1_dev_out = f.readlines()