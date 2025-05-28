import numpy as np

# function to convert the input sequence into pairs and then calculate its frequencies
def pair_frequencies(S):
    L = len(S)
    freq_of_pairs = {} #dictionary to hold frequencies corresponding to pairs
    index_of_pairs = {} #dictionary to hold the first occurence of the pair, to be used in tie-breaking

    for i in range(L-1):
        pair = tuple(S[i:i+2])
        if pair not in freq_of_pairs:
            freq_of_pairs[pair] = 1
            index_of_pairs[pair] = i
        else:
            freq_of_pairs[pair] += 1
    return freq_of_pairs, index_of_pairs


#function to determine tie-breaks between tow similar pair candidate for substitution 
def tie_break(pair, index_of_pairs, symbol_scales):
    a,b = pair
    a_scale = symbol_scales.get(a,1)
    b_scale = symbol_scales.get(b,1)

    scale = a_scale + b_scale

    smallest_scale = min(a_scale,b_scale)

    first_occurence = index_of_pairs[pair]

    return [scale, smallest_scale, first_occurence]

#function to selce the pair to be substituted using successive tie-break steps
def selection(freq, index, symbol_scales):    
    max_freq = max(freq.values())
    candidates = [k for k,v in freq.items() if v == max_freq] #selecting all keys with maximum values
    
    if len(candidates) > 1:
        min_scale = min(tie_break(pair, index, symbol_scales)[0] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index, symbol_scales)[0] == min_scale] #selecting the keys with least scales
        
    if len(candidates) > 1:
        min_smallest_scale = min(tie_break(pair, index, symbol_scales)[1] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index, symbol_scales)[1] == min_smallest_scale] #selecting the keys with the smallest individual digit
            
    if len(candidates) != 1:
        min_first_occurence = min(tie_break(pair,index, symbol_scales)[2] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair,index,symbol_scales)[2] == min_first_occurence] # selecting the pair which occur first
    
    most_repeated = candidates[0]
 
    return most_repeated



#Checking the validity of the input and then converting the into a symbolic sequence 
def check_and_convert(S, num_bins=0):
    symbol_scales = {}

    if num_bins == 0: #sequence already a symbolic sequence
        if isinstance(S, str):
            type_of_elem = set(S)
            for sym in type_of_elem:
                if sym.isdigit():
                    symbol_scales[int(sym)] = 1
                else:
                    symbol_scales[sym] = 1
            S = [int(x) if x.isdigit() else x for x in S]
        
        else:
            S = np.array(S)
            S = [str(x) for x in S]

            for sym in set(S):
                symbol_scales[sym] = 1

            S = [int(x) if x.isdigit() else x for x in S]
            
        return S, symbol_scales

#        if len(type_of_elem) < 3:
#            if len(type_of_elem) == 1:
#                return S
#            else:
#                S = [0 if x==list(type_of_elem)[0] else 1 for x in S]
#                return S
#        else:
#            print("If the input has more than two symbols, it needs bins to resolve it. Try using etc(x, num_bins=2)")
#            return None
    
    else: #num_bins > 0
        if isinstance(S, str):
            print('check if the input is valid, more than two types of symbol in a string input')
            return None, None
        
        S = np.array(S)

        range_S = max(S) - min(S)
        delta = range_S / num_bins if max(S) > min(S) else 1

        S_new = []
        for x in S:
            if range_S == 0:
                bin_idx = 0
            else:
                bin_idx = min( int( (x - min(S)) / delta), num_bins - 1 )
            S_new.append(bin_idx)

        for i in range(num_bins):
            symbol_scales[i] = 1  

        return S_new, symbol_scales
    
#        num_bins = 2 #can only take num_bins as 2, otherwise it breaks the tie breaking method scale
#        mean_S = (max(S) + min(S))/2
#
#        S = [0 if x<=mean_S else 1 for x in S]
#        return S


#function to substitute the new symbol
def replace(S, most_repeated, new_symbol):
    S_new=[]
    i = 0
    while i < len(S):
        if i < len(S) - 1 and tuple(S[i:i+2]) == most_repeated:
            S_new.append(new_symbol)
            i+=2
        else:
            S_new.append(S[i])
            i+=1
    return S_new
        

#function to compress the sequence successivly substituting one kind of pair at a time
def calculate_etc(S, symbol_scales, verbose=False):
    t = 0

    new_symbol = max( [x if isinstance(x, int) else -1 for x in set(S)] ) + 1

    while len(S) > 1:
        freq, index = pair_frequencies(S)
        if not freq:
            break

        most_repeated = selection(freq, index, symbol_scales)
        new_symbol += 1

        a,b = most_repeated
        symbol_scales[new_symbol] = symbol_scales.get(a,1) + symbol_scales.get(b,1)

        S = replace(S, most_repeated, new_symbol)
        t += 1 #counting iterations
        
        if verbose:
            print(f"Step {t}; sequence {S}")
    return t



def etc(data, num_bins=0, normalized=False, verbose=False):
    S, symbol_scales = check_and_convert(data, num_bins)

    if S is None:
        print("Problem in check funcion")
        return None

    original_length = len(S)
    etc_value = calculate_etc(S, symbol_scales, verbose)

    if normalized:
        norm_etc = etc_value / (original_length - 1) if original_length > 1 else 0
        return norm_etc
    else:
        return etc_value
