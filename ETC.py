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
    list_of_elem = list(S)
    type_of_elem = []
    for x in list_of_elem:
        if len(type_of_elem) < 3:
            if x not in type_of_elem:
                type_of_elem.append(x)
        else:
            print("Check to see if the input sequence has more than two symbols, Please enter the sequence without spaces")
            break
    binary_seq = "".join(['0' if x==type_of_elem[0] else '1' for x in list_of_elem])
    print(f"Converted Sequence: {binary_seq}")
    
    return binary_seq


#function to compress the sequence successivly substituting one kind of pair at a time
def compress(S, t=0):
    S = check(S)
    while len(S) > 1:
        freq, index = pair_frequencies(S)
        most_repeated = selection(freq, index)
        new_symbol = str(t+2)
        S = S.replace(most_repeated, new_symbol)
        t += 1 #counting iterations

        print(f"Step {t}; sequence {S}")
    return t


#Running the function
S0 = input("Enter initial sequence: ")
ETC = compress(S0)
print(f"ETC: {ETC}")





