def viterbi(transition_parameters, emission_parameters, vocab, sentence):
    #Initialisation
    n = len(sentence)
    viterb = []
    viterb.append({'START': (0, '')})

    label_list = list(transition_parameters.keys())
    label_list.remove('START')
    label_list.remove('STOP')

    for j in range(n):
        viterb.append({})
        #Check if word in training set, else set to #UNK#
        if vocab.get(sentence[j], False):
            observation = sentence[j]
        else:
            observation = '#UNK#'

        #Calc scores
        for next_label in label_list:
            b = emission_parameters[next_label].get(observation, -float('inf'))

            max = -float('inf')
            new_label = ''
            for prev_label in transition_parameters.keys():
                v = viterb[j].get(prev_label, (-float('inf'),''))[0]
                a = transition_parameters.get(prev_label, {}).get(next_label, -float('inf'))
                new_score = v + b + a
                if new_score > max:
                    max = new_score
                    new_label = prev_label
            #only add entry if meaningful for easier debugging
            if new_label != '':
                viterb[j+1][next_label] = (max, new_label)
        
        #edge case where by some miracle all possible combi lead to probability = 0 (log-prob = -inf) and nothing is recorded
        #Current default is assign 'O' as label and set log-prob as max in prev step (with corresponding label)
        if len(list(viterb[j+1].keys())) < 1:
            max = -float('inf')
            new_label = ''
            for prev_label in transition_parameters.keys():
                if viterb[j].get(prev_label, (-float('inf'),''))[0] > max:
                    max = viterb[j].get(prev_label, (-float('inf'),''))[0]
                    new_label = prev_label
            viterb[j+1]['O'] = (max, new_label)
            print('edge case encountered')

    #Termination
    max = -float('inf')
    new_label = ''
    for prev_label in transition_parameters.keys():
        v = viterb[n].get(prev_label, (-float('inf'),''))[0]
        a = transition_parameters.get(prev_label, {}).get('STOP', -float('inf'))
        new_score = v + a
        if new_score > max:
            max = new_score
            new_label = prev_label
    viterb.append({'STOP': (max, new_label)})

    #backtracking
    seq = []
    curr_label = 'STOP'
    for i in range(n+1, 1, -1):
        try:
            curr_label = viterb[i][curr_label][1]
        except KeyError:
            print(f'i: {i}, curr_label: {curr_label}, seq: {seq}')
            print(viterb)
            print(sentence)
            break
        seq.append(curr_label)

    return list(reversed(seq))