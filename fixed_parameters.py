##### List of parameters used that are not subject to change #####

### FUNCTIONS TO READ FILES ###

# open_labelled_data function is used to read files with labels
def open_labelled_data(data_path):
    words = []  # Initialize a list to store words for each sentence
    tags = []   # Initialize a list to store tags for each sentence
    
    with open(data_path, 'r', encoding="utf-8") as f:
        entity_words = []  # Temporary list to store words for a sentence
        entity_tags = []   # Temporary list to store tags for a sentence
        
        for line in f:
            if line.isspace():
                # When encountering an empty line, a sentence is complete
                # Append the temporary word and tag lists to the main lists
                words.append(entity_words[:])
                tags.append(entity_tags[:])
                entity_words.clear()  # Clear the temporary word list for the next sentence
                entity_tags.clear()   # Clear the temporary tag list for the next sentence
            else:
                # When a non-empty line is encountered, extract the word and tag
                word, tag = line.strip().rsplit(maxsplit=1)
                entity_words.append(word)  # Add the word to the temporary word list
                entity_tags.append(tag)    # Add the tag to the temporary tag list
    
    return words, tags  # Return the main lists containing words and tags for each sentence

# open_unlabelled_data function is used to read files without labels
def open_unlabelled_data(data_path):
    words = []  # Initialize a list to store words for each sentence
    
    with open(data_path, 'r', encoding="utf-8") as f:
        entity_words = []  # Temporary list to store words for a sentence
        
        for line in f:
            if line.isspace():
                # When encountering an empty line, a sentence is complete
                # Append the temporary word list to the main list
                words.append(entity_words[:])
                entity_words.clear()  # Clear the temporary word list for the next sentence
            else:
                # When a non-empty line is encountered, extract the word
                word = line.strip()
                entity_words.append(word)  # Add the word to the temporary word list
    
    return words  # Return the main list containing words for each sentence

#-----------------------------#

### FILES ###

# labelled: train, dev.out
# unlabelled: dev.in

# ES
ES_dev_out = open_labelled_data('ES/dev.out')
ES_train = open_labelled_data('ES/train')
ES_dev_in = open_unlabelled_data('ES/dev.in')

# RU
RU_dev_out = open_labelled_data('RU/dev.out')
RU_train = open_labelled_data('RU/train')
RU_dev_in = open_unlabelled_data('RU/dev.in')

#-----------------------------#


'''
### FILES ###
# ES 
with open("ES/dev.in", "r", encoding="utf-8") as f:
    ES_dev_in = f.readlines()
with open("ES/dev.out", "r", encoding="utf-8") as f:
    ES_dev_out = f.readlines()
with open("ES/train", "r", encoding="utf-8") as f:
    ES_train = f.readlines()

# RU 
with open("RU/dev.in", "r", encoding="utf-8") as f:
    RU_dev_in = f.readlines()
with open("RU/dev.out", "r", encoding="utf-8") as f:
    RU_dev_out = f.readlines()
with open("RU/train", "r", encoding="utf-8") as f:
    RU_train = f.readlines()

### TAGS ###
NUMBER_OF_TAGS = 7

B_Positive = "B-positive"
B_Negative = "B-negative"
B_Neutral = "B-neutral"
I_Positive = "I-positive"
I_Negative = "I-negative"
I_Neutral = "I-neutral"
Outside = "O"

TAGS = (B_Positive, B_Negative, B_Neutral, I_Positive, I_Negative, I_Neutral, Outside)

### WORDS ###
Invalid_Word = "#UNK#"
'''

