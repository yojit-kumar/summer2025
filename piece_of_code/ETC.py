def pair_frequencies(S):
    L = len(S)
    freq_of_pairs = {}
    index_of_pairs = {}

    for i in range(L-1):
        pair = S[i:i+2]
        if pair not in freq_of_pairs:
            freq_of_pairs[pair] = 1
            index_of_pairs[pair] = i
        else:
            freq_of_pairs[pair] += 1
    return freq_of_pairs, index_of_pairs


def tie_break(pair, index_of_pairs):
    a,b = map(int, list(pair))
    scale = a+b
    smallest_digit = min(a,b)
    first_occurence = index_of_pairs[pair]

    return [scale, smallest_digit, first_occurence]

def selection(freq, index):
    
    max_freq = max(freq.values())
    candidates = [k for k,v in freq.items() if v == max_freq]
    
    if len(candidates) > 1:
        min_scale = min(tie_break(pair, index)[0] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[0] == min_scale]
        
    if len(candidates) > 1:
        min_digit = min(tie_break(pair, index)[1] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[1] == min_digit]
            
    if len(candidates) != 1:
        min_first_occurence = min(tie_break(pair,index)[2] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair,index)[2] == min_first_occurence]
    
        most_repeated = candidates[0]
    else:
        most_repeated = candidates[0]
 
    return most_repeated


def compress(S, t=0):
    while len(S) > 1:
        freq, index = pair_frequencies(S)
        most_repeated = selection(freq, index)
        new_symbol = str(t+3)
        S = S.replace(most_repeated, new_symbol)
        t += 1

        print(f"Step {t}; sequence {S}")
    return t



S0 = input("Enter initial sequence: ")
compress(S0)





