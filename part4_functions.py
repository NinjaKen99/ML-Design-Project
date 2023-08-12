def estimate_emission_params_vote_based(words, tags, k=1):
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

            if tag.startswith('I-') and dominant_sentiment == 'B-positive':
                tag = 'B-positive'
            elif tag.startswith('I-') and dominant_sentiment == 'B-negative':
                tag = 'B-negative'
            elif tag.startswith('I-') and dominant_sentiment == 'B-neutral':
                tag = 'B-neutral'

            elif tag.startswith('B-') and dominant_sentiment == 'I-positive':
                tag = 'I-positive'
            elif tag.startswith('B-') and dominant_sentiment == 'I-negative':
                tag = 'I-negative'
            elif tag.startswith('B-') and dominant_sentiment == 'I-neutral':
                tag = 'I-neutral'

            if tag not in emission_params:
                emission_params[tag] = {'Total': k, '#UNK#': k}

            emission_params[tag][word] = emission_params[tag].get(word, 0) + 1
            emission_params[tag]['Total'] += 1

    for tag, tag_counts in emission_params.items():
        total_count = tag_counts['Total']
        for word in tag_counts:
            if word != 'Total':
                emission_params[tag][word] = tag_counts[word] / total_count

    return emission_params, words_observed
