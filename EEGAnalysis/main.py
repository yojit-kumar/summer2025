import os
import pyedflib as edf
import ETC 
import pandas as pd


def process_data(volunteer_ids, channel_ids, root_dir='/home/user/eeg-motor-movementimagery-dataset-1.0.0/files/'):
    results = []

    for v in volunteer_ids:
        v_path = os.path.join(root_dir, v)


        task1 = os.path.join(v_path, f'{v}R01.edf')
        task2 = os.path.join(v_path, f'{v}R02.edf')

        f1 = edf.EdfReader(task1)
        f2 = edf.EdfReader(task2)

        labels = f1.getSignalLabels()
        label_idx = {label: idx for idx, label in enumerate(labels)}


        for ch in channel_ids:
            idx = label_idx[ch]
            
            signal1 = f1.readSignal(idx)[:9600]
            signal2 = f2.readSignal(idx)[:9600]

            etc_1 = etc_func(signal1)
            etc_2 = etc_func(signal2)

            results.append({
                'volunteer': v,
                'channel': ch,
                'ETC for EyesOpen': etc_1,
                'ETC for EyesClosed': etc_2,
                'Difference EO - EC': etc_1 - etc_2
                })

    return pd.DataFrame(results)

def etc_func(signal):
    seq = ETC.partition(signal, n_bins=2)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')

    return res

if __name__ == '__main__':

    volunteers=[]
    for n in range(1,110):
        if len(str(n)) == 1:
            S = 'S00'+f'{n}'
        elif len(str(n)) == 2:
            S = 'S0'+f'{n}'
        else:
            S = 'S'+f'{n}'
        volunteers.append(S)


    channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 
                'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
                'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
                'Fp1.', 'Fpz.', 'Fp2.', 
                'Af7.', 'Af3.', 'Afz.', 'Af4.','Af8.',
                'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..','F4..', 'F6..', 'F8..',
                'Ft7.', 'Ft8.',
                'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 
                'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 
                'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 
                'O1..', 'Oz..', 'O2..', 
                'Iz..']
    

    df = process_data(volunteers, channels)
    df.to_csv("EEGAnalysis_results2.csv", index=False)

    print(df)
