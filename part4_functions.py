# Imports from another file
from fixed_parameters import ES_train, RU_train
from fixed_parameters import ES_dev_in, RU_dev_in
from fixed_parameters import ES_dev_out, RU_dev_out
from fixed_parameters import open_labelled_data, open_unlabelled_data
from part1_functions import estimate_emission_params, sentiment_analysis
import numpy as np



def estimate_emission_params_modified(words, tags, k=1):

    emission_params = {}  # Emission probabilities for each tag and sentiment
    words_observed = {}   # Observed words

    for sentence, sentence_tags in zip(words, tags):
        sentiment_counts = {'B-positive': 0, 'B-negative': 0,
                            'B-neutral': 0, 'I-positive': 0, 'I-negative': 0, 'I-neutral': 0}

        for tag in sentence_tags:
            if tag in sentiment_counts:
                sentiment_counts[tag] += 1

        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        for word, tag in zip(sentence, sentence_tags):
            if tag not in emission_params:
                emission_params[tag] = {'Total': k, '#UNK#': k}

            if tag.startswith('I-') or word == "!" or word == "?"  and dominant_sentiment == 'B-positive':
                tag = 'B-positive'
            elif tag.startswith('I-') or word == "!" or word == "?" and dominant_sentiment == 'B-negative':
                tag = 'B-negative'
            elif tag.startswith('I-') or word == "!" or word == "?" and dominant_sentiment == 'B-neutral':
                tag = 'B-neutral'

            elif tag.startswith('B-') or word == "!" or word == "?" and dominant_sentiment == 'I-positive':
                tag = 'I-positive'
            elif tag.startswith('B-') or word == "!" or word == "?"  and dominant_sentiment == 'I-negative':
                tag = 'I-negative' 
            elif tag.startswith('B-') or word == "!" or word == "?" and dominant_sentiment == 'I-neutral':
                tag = 'I-neutral'

            if tag not in emission_params:
                emission_params[tag] = {'Total': k, '#UNK#': k}

            emission_params[tag][word] = emission_params[tag].get(word, 0) + 1
            emission_params[tag]['Total'] += 1

              # Track observed words
            words_observed[word] = True

    for tag, tag_counts in emission_params.items():
        total_count = tag_counts['Total']
        for word in tag_counts:
            if word != 'Total':
                emission_params[tag][word] = tag_counts[word] / total_count

    return emission_params, words_observed



words, tags = ES_train

emission_params, words_observed = estimate_emission_params(words, tags)

sentiment_analysis(emission_params, words_observed,
                   'ES/test.in', 'ES/test.p4.out')

# For the RU (Russian) dataset
words, tags = RU_train
transition_params, words_observed = estimate_emission_params(words, tags)
sentiment_analysis(transition_params, words_observed,
                   'RU/test.in', 'RU/test.p4.out')