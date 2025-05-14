import numpy as np

def partition(X, num_bins=2):
    X = np.array(X)

    range_x = max(X) - min(X)
    delta = range_x / num_bins

    symbols = np.floor((X-min(X))/ delta).astype(int)

    symbols = np.clip(symbols, 0, num_bins-1)

    return ''.join(map(str, symbols))

# function to convert the input sequence into pairs and then calculate its frequencies
def pair_frequencies(S):
    L = len(S)
    freq_of_pairs = {} #dictionary to hold frequencies corresponding to pairs
    index_of_pairs = {} #dictionary to hold the first occurence of the pair, to be used in tie-breaking

    for i in range(L-1):
        pair = S[i:i+2]
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


#Input Checking to validate if algorithm can be run
def check(S):
    #Converting if not string
    if not isinstance(S, str):
        S = ''.join(map(str, S))

    type_of_elem = set(S)
    
    if len(type_of_elem) < 3:
        if len(type_of_elem) == 1:
            return S
        else:
            binary_seq = "".join(['0' if x==list(type_of_elem)[0] else '1' for x in S])
        #print(f"Converted Sequence: {binary_seq}")
        return binary_seq
    else:
       print("Check to see if the input sequence has more than two symbols, Please enter the sequence without spaces")
       return None
            


#function to compress the sequence successivly substituting one kind of pair at a time
def calculate_etc(S, verbose=False):
    t = 0
    while len(S) > 1:
        freq, index = pair_frequencies(S)
        most_repeated = selection(freq, index)
        new_symbol = str(t+2)
        S = S.replace(most_repeated, new_symbol)
        t += 1 #counting iterations
        
        if verbose:
            print(f"Step {t}; sequence {S}")
    return t

def etc(data, num_bins=0, normalized=False, verbose=False):
    if num_bins > 0:
        S = partition(data, num_bins)
    else:
        if not isinstance(data, str):
            S = ''.join(map(str, data))
        else:
            S = data

    S = check(S)
    if S is None:
        return None

    original_length = len(S)
    etc_value = calculate_etc(S, verbose)

    if normalized:
        norm_etc = etc_value / (original_length - 1) if original_length > 1 else 0
        return norm_etc
    else:
        return etc_value

if __name__ == "__main__":
       # Example with a symbolic sequence
    test_sequence = "01010011"
    etc_value, norm_etc = etc(test_sequence, normalized=True, verbose=True)
    print(f"ETC: {etc_value}")
    print(f"Normalized ETC: {norm_etc:.4f}")
    
    # Example with a time series
    time_series = np.random.rand(20)  # Random values between 0 and 1
    etc_value, norm_etc = etc(time_series, num_bins=5, normalized=True, verbose=True)
    print(f"Time Series ETC: {etc_value}")
    print(f"Time Series Normalized ETC: {norm_etc:.4f}")
