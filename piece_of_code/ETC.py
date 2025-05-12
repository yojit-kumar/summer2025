import numpy as np

#S0 = input('Enter:')
#L0 = len(S0)

#temp_memory = {}

#for i in range(L0):
#    j = S0[i:i+2]
#    paired_seq.append([j])
#    if j not in temp_memory:
#        temp_memory[j] = 1
#    else:
#       temp_memory[j] += 1

#most_repeated = max(temp_memory, key=temp_memory.get)

#new_paired_seq = [3 if x = most_repeated else x for x in paired_seq]

#S1 = S0.replace(most_repeated, '3')

#print(S1)

def compress(S, t):
    L = len(S)

    temp_memory = {}

    for i in range(L):
        j = S[i:i+2]

        if j not in temp_memory:
            temp_memory[j] = 1
        else:
            temp_memory[j] += 1

    most_repeated = max(temp_memory,key=temp_memory.get)
    n = str(t+3)
    S_new = S.replace(most_repeated, n)
    t += 1

    print(f'most_repeated: {most_repeated}')
    print(f'S_new: {S_new}')
    print(f'steps: {t}')
    
    if len(S_new) > 1:
        return compress(S_new, t)
    elif len(S_new) == 1:
        return "STOPPED"

S1 = input("Enter initial sequence: ")
t = 0

compress(S1, t)


