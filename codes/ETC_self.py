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
def tie_break(pair, index_of_pairs):
    a,b = map(int, list(pair))
    if a == 0 and b == 0: #have to use an if clause as both 1 & 0 are of scale 1
        scale = a+b+2
    elif a == 0 or b == 0:
        scale = a+b+1
    else:
        scale = a+b #scale is roughly the length of the pattern
    smallest_digit = min(a,b)
    first_occurence = index_of_pairs[pair]

    return [scale, smallest_digit, first_occurence]

#function to selce the pair to be substituted using successive tie-break steps
def selection(freq, index):    
    max_freq = max(freq.values())
    candidates = [k for k,v in freq.items() if v == max_freq] #selecting all keys with maximum values
    
    if len(candidates) > 1:
        min_scale = min(tie_break(pair, index)[0] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[0] == min_scale] #selecting the keys with least scales
        
    if len(candidates) > 1:
        min_digit = min(tie_break(pair, index)[1] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[1] == min_digit] #selecting the keys with the smallest individual digit
            
    if len(candidates) != 1:
        min_first_occurence = min(tie_break(pair,index)[2] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair,index)[2] == min_first_occurence] # selecting the pair which occur first
    
        most_repeated = candidates[0]
    else:
        most_repeated = candidates[0]
 
    return most_repeated



#Checking the validity of the input and then converting the into a symbolic sequence 
def check_and_convert(S, num_bins=0):
    if num_bins == 0: #sequence already a symbolic sequence
        if not isinstance(S, str):
            S = str(S)

        type_of_elem = set(S)

        if len(type_of_elem) < 3:
            if len(type_of_elem) == 1:
                return S
            else:
                S = [0 if x==list(type_of_elem)[0] else 1 for x in S]
                return S
        else:
            print("If the input has more than two symbols, it needs bins to resolve it. Try using etc(x, num_bins=2)")
            return None
    
    else: #num_bins > 0
        if isinstance(S, str):
            print('check if the input is valid, more than two types of symbol in a string input')
            return None
        
        S = np.array(S)

        #range_S = max(S) - min(S)
        #delta = range(S) / num_bins
    
        #symbols = np.floor((S-min(S))/ delta).astype(int)
        #symbols = np.clip(symbols, 0, num_bins-1)

        #return ''.join(map(str, symbols))
        num_bins = 2 #can only take num_bins as 2, otherwise it breaks the tie breaking method scale
        mean_S = (max(S) + min(S))/2

        S = [0 if x<=mean_S else 1 for x in S]
        return S


#function to substitute the new symbol
def replace(S, most_repeated, new_symbol):
    S_new=[]
    i = 0
    while i < len(S):
        if tuple(S[i:i+2]) == most_repeated:
            S_new.append(new_symbol)
            i+=2
        else:
            S_new.append(S[i])
            i+=1
    return S_new
        

#function to compress the sequence successivly substituting one kind of pair at a time
def calculate_etc(S, verbose=False):
    t = 0
    while len(S) > 1:
        freq, index = pair_frequencies(S)
        most_repeated = selection(freq, index)
        new_symbol = str(t+2)
        S = replace(S, most_repeated, new_symbol)
        t += 1 #counting iterations
        
        if verbose:
            print(f"Step {t}; sequence {S}")
    return t

def etc(data, num_bins=0, normalized=False, verbose=False):
    S = check_and_convert(data, num_bins)

    if S is None:
        print("Problem in check funcion")
        return None

    original_length = len(S)
    etc_value = calculate_etc(S, verbose)

    if normalized:
        norm_etc = etc_value / (original_length - 1) if original_length > 1 else 0
        return norm_etc
    else:
        return etc_value
