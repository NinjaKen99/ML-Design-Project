# Imports from another file
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data
import numpy as np

# Functions


def estimate_emission_params(words, tags, k=1):
    emission_params = {}  # Emission probabilities for each tag and word
    words_observed = {}   # Observed words

    # Iterate through each sentence and word in the tagged data
    for sentence, sentence_tags in zip(words, tags):
        for word, tag in zip(sentence, sentence_tags):
            # Initialize tag's emission_params if it's not in emission_params
            if tag not in emission_params:
                emission_params[tag] = {'Total': k, '#UNK#': k}

            # Increment emission count for the tag and word
            emission_params[tag][word] = emission_params[tag].get(word, 0) + 1
            emission_params[tag]['Total'] += 1

            # Track observed words
            words_observed[word] = True

    # Calculate emission probabilities
    for tag, tag_counts in emission_params.items():
        total_count = tag_counts['Total']
        for word in tag_counts:
            if word != 'Total':
                emission_params[tag][word] = tag_counts[word] / total_count

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
                    label = highest_probability_tag(
                        emission_params, words_observed, word)  # Get the predicted tag
                    # Write word and predicted label to the output file
                    output_file.write(word + ' ' + label + '\n')


# Generate output predictions for different datasets and languages
# For the ES (Spanish) dataset
words, tags = ES_train

emission_params, words_observed = estimate_emission_params(words, tags)

sentiment_analysis(emission_params, words_observed,
                   'ES/dev.in', 'ES/dev.p1.out')

# For the RU (Russian) dataset
words, tags = RU_train
transition_params, words_observed = estimate_emission_params(words, tags)
sentiment_analysis(transition_params, words_observed,
                   'RU/dev.in', 'RU/dev.p1.out')

#____________________TESTING____________________#
# run funtions below


### Tester code for estimate_emission_params ###

# Extract words and tags from the training data
words, tags = ES_train

# Calculate emission probabilities and track observed words
# emission_params, words_observed = estimate_emission_params(words, tags)

emission_params, words_observed = estimate_emission_params(
    words, tags)


# Print the list of unique tags for which emission probabilities are calculated
print("Unique tags for emission probabilities:", list(emission_params.keys()))

# Specify a tag and an observation for testing emission probabilities
tag = 'B-negative'   # The tag for which emission probabilities are being tested
# The observation (word) for which emission probability is being tested
observation = '#UNK#'

# Print the number of different words observed for the specified tag
print("Number of different words observed for tag",
      tag, ":", len(emission_params[tag].keys()))

# Print the calculated emission probability for the specified observation and tag
print("Emission probability (e(x|y)) for observation", observation, "given tag", tag, ":",
      emission_params[tag][observation])

# Print the count estimate for the specified observation and tag using e(x|y) * count(y)
print("Count estimate for observation", observation, "given tag", tag, ":",
      emission_params[tag][observation] * emission_params[tag]['Total'])

### End of tester code for estimate_emission_params ###
