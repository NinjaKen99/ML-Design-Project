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

def estimate_emission_parameter_v2(data, WORD, TAG): # Part b: For testing
    # Split file by line
    dataset = data.split("\n")
    # Set up dictionary for counting
    counter = {"Count": 0, "Word": 0, "Unknown": 1} # Initialise k to 1
    
    for line in dataset:
        # Account for gaps in file (Skip)
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word, tag = pair[0], pair[1]
            # Perform Counting
            if (tag == TAG):
                counter["Count"] += 1
                if (word == unknown):
                    counter["Unknown"] += 1
                elif (word == WORD):
                    counter["Word"] += 1
    # Calculate emission parameter depending on word or unknown word
    if (WORD == unknown):
        return counter["Unknown"] / (counter["Count"] + counter["Unknown"])
    return counter["Word"] / (counter["Count"] + counter["Unknown"])

def estimate_emission_parameter_v3(data, TAG): # Modified for part c: For training
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
                    counter[word.encode('utf-8')] = 1 # Add entry if not exist
                    emission_parameters[word.encode('utf-8')] = None
                else: counter[word] += 1
    total = counter["Count"] # Count(y)
    for keys in emission_parameters.keys():
        emission_parameters[keys] = counter[keys] / total      
    return emission_parameters # return dictionary with all emission parameters

''' Faulty
def estimate_emission_parameter_v3(data, TAG): # Modified for part c: For training
    # Split file by line
    dataset = data.split("\n")
    # Set up dictionary for counting
    counter = {"Count": 0, "Unknown": 1}
    for line in dataset:
        # Account for gaps in file (Skip)
        if (line != ""):
            # Split line into word and tag
            pair = line.split(" ")
            word, tag = pair[0], pair[1]
            # Perform Counting
            if (word == unknown): # Count all unknowns
                    counter["Unknown"] += 1
            elif (tag == TAG): # Count for y
                counter["Count"] += 1
                if (word not in counter.keys()): # Count for all x
                    counter[word] = 1 # Add entry if not exist
                else: counter[word] += 1
    # new dictionary with emission parameter for each word that exists
    emission_parameters = {"Unknown": counter["Unknown"]/(counter["Count"] + counter["Unknown"])}
    total = counter["Count"]
    k = counter["Unknown"]
    for keys, values in counter.items():
        if (keys != "Count" or keys != "Unknown"):
            emission_parameters[keys] = counter[keys] / (counter["Count"] + counter["Unknown"])        
    return emission_parameters # return dictionary with all emission parameters
'''

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
            except:
                pass
        word_dict[word] = y_star
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
        
        if word_tag_pair == gold_tag:
            total_correct_predictions += 1
        if word_tag_pair != "O":
            total_predicted_entities += 1
        if gold_tag != "O":
            total_gold_entities += 1
            
        p = precision(total_correct_predictions, total_predicted_entities)
        r = recall(total_correct_predictions, total_gold_entities)
        f = f_score(p, r)
        
    return word_tag_list, p, r, f

# Writing to Files
with open('Data/RU/dev.p1.out', 'w', encoding="utf-8") as f:
   f.write('\n'.join(RU_devout_lines))
with open('Data/ES/dev.p1.out', 'w', encoding="utf-8") as f:
   f.write('\n'.join(ES_devout_lines))
   
# Reading lines from dev.p1.out files
with open('Data/ES/dev.p1.out', 'r', encoding="utf-8") as f:
    ES_p1_devout = f.readlines()
with open('Data/RU/dev.p1.out', 'r', encoding="utf-8") as f:
    RU_p1_devout = f.readlines()

#____________________TESTING____________________#
# run funtions below

tagset = produce_tag(ES_train, TAGS)
print(tagset)
