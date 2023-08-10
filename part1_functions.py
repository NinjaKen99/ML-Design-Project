# Imports from another file
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data

# Functions

def estimate_emission_params(words, tags, k=1):
    # Initialize dictionaries to store emission probabilities and observed words
    emission_params = {}  # Stores emission probabilities for each tag and word
    words_observed = {}   # Stores observed words
    
    # Loop through each sentence and its words in the tagged data
    for sentence in range(len(tags)):
        for tag_for_word in range(len(tags[sentence])):
            # Check if the tag is already a key in emission_params
            if tags[sentence][tag_for_word] not in emission_params.keys():
                # Initialize emission_params[tag] with counts for smoothing
                emission_params[tags[sentence][tag_for_word]] = {'Total': k, '#UNK#': k}
            
            # Get the current count of the word's emission for the tag
            current_count = emission_params[tags[sentence][tag_for_word]].get(words[sentence][tag_for_word], 0)
            
            # Increment the emission count for the word's tag and word
            emission_params[tags[sentence][tag_for_word]][words[sentence][tag_for_word]] = current_count + 1
            
            # Increment the total emission count for the tag
            emission_params[tags[sentence][tag_for_word]]['Total'] += 1
            
            # Track that the word has been observed
            words_observed[words[sentence][tag_for_word]] = True
    
    # Calculate emission probabilities for each tag and word
    for tag in emission_params.keys():
        for emission_observation in emission_params[tag].keys(): # Iterate through each word observed for the tag
            if emission_observation != 'Total':
                # Calculate and store emission probability
                emission_params[tag][emission_observation] = emission_params[tag][emission_observation] / emission_params[tag]['Total']
    
    # Return the calculated emission probabilities and observed words
    return emission_params, words_observed

#-----------------------------#

# Calculate and return the most likely tag (label) for a given word based on emission probabilities
def highest_probability_tag(emission_params, words_observed, word):
    max_prob = 0  # Initialize maximum probability
    # Check if the word is in the training set, otherwise set it to '#UNK#'
    if not words_observed.get(word, False):
        word = '#UNK#'
    # Iterate through each tag to find the most likely tag for the word
    for tag_iterator in emission_params.keys():
        if word in emission_params[tag_iterator].keys() and emission_params[tag_iterator][word] > max_prob:
            max_prob = emission_params[tag_iterator][word]
            tag = tag_iterator
    return tag

# Predict tags for an input file and write the predictions to an output file
def sentiment_analysis(emission_params, words_observed, data_input_path, data_output_file):
    with open(data_input_path, 'r', encoding="utf-8") as input_file:
        with open(data_output_file, 'w', encoding="utf-8") as output_file:
            for line in input_file:
                if line.isspace():
                    output_file.write('\n')  # Write a newline for empty lines
                else:
                    word = line.strip()  # Extract the word from the line
                    label = highest_probability_tag(emission_params, words_observed, word)  # Get the predicted tag
                    output_file.write(word + ' ' + label + '\n')  # Write word and predicted label to the output file

# Generate output predictions for different datasets and languages
# For the ES (Spanish) dataset
words, tags = ES_train
transition_params, words_observed = estimate_emission_params(words, tags)
sentiment_analysis(transition_params, words_observed, 'ES/dev.in', 'ES/dev.p1.out')

# For the RU (Russian) dataset
words, tags = RU_train
transition_params, words_observed = estimate_emission_params(words, tags)
sentiment_analysis(transition_params, words_observed, 'RU/dev.in', 'RU/dev.p1.out')

#____________________TESTING____________________#
# run funtions below


### Tester code for estimate_emission_params ###

# Extract words and tags from the training data
words, tags = ES_train

# Calculate emission probabilities and track observed words
transition_params, words_observed = estimate_emission_params(words, tags)

# Print the list of unique tags for which emission probabilities are calculated
print("Unique tags for emission probabilities:", list(transition_params.keys()))

# Specify a tag and an observation for testing emission probabilities
tag = 'B-negative'   # The tag for which emission probabilities are being tested
observation = '#UNK#'  # The observation (word) for which emission probability is being tested

# Print the number of different words observed for the specified tag
print("Number of different words observed for tag", tag, ":", len(transition_params[tag].keys()))

# Print the calculated emission probability for the specified observation and tag
print("Emission probability (e(x|y)) for observation", observation, "given tag", tag, ":",
      transition_params[tag][observation])

# Print the count estimate for the specified observation and tag using e(x|y) * count(y)
print("Count estimate for observation", observation, "given tag", tag, ":",
      transition_params[tag][observation] * transition_params[tag]['Total'])

### End of tester code for estimate_emission_params ###
