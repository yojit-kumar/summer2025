import os
import numpy as np
import pandas as pd
import ETC 
import mne



root_dir = '/home/user/eeg-motor-movementimagery-dataset-1.0.0/files/'
ALPHA_BAND = (8.,12.)
NOTCH_FREQ = 60



def compute_alpha(raw):
    """Applying a 60Hz notch filter to remove powerline noise and extract alpha band frequency"""

    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw_alpha = raw.copy().filter(*ALPHA_BAND, verbose=False)

    return raw_alpha



def process_data(volunteer_ids, root_dir):    
    results = []

    for v in volunteer_ids:
        v_path = os.path.join(root_dir, v)
        task1 = os.path.join(v_path, f'{v}R01.edf')
        task2 = os.path.join(v_path, f'{v}R02.edf')
        try:
            r1 = mne.io.read_raw_edf(task1, preload=True, verbose=False)
            r2 = mne.io.read_raw_edf(task2, preload=True, verbose=False)
        except Exception as e:
            print(f'Error with {v}: {e}')
            continue

        labels = r1.ch_names
        label_idx = {label: idx for idx, label in enumerate(labels)}

        if r1.times[-1] > 60:
            r1.crop(tmin=0,tmax=60)
        if r2.times[-1] > 60:
            r2.crop(tmin=0,tmax=60)

        r1 = compute_alpha(r1)
        r2 = compute_alpha(r2)

        r1_data = r1.get_data(tmax=60)
        r2_data = r2.get_data(tmax=60)

        array1=[]
        array2=[]

        for ch in labels:
            idx = label_idx[ch]
            signal1 = r1_data[idx,:]
            signal2 = r2_data[idx,:]
            
            array1.append(signal1)
            array2.append(signal2)

        array1 = np.array(array1).T
        array2 = np.array(array2).T

        etc_1 = [etc_func(x) for x in array1]
        etc_2 = [etc_func(y) for y in array2]

        for t in range(len(etc_1)):
            results.append({
                'volunteer': v,
                'times': t,
                'ETC for EyesOpen': etc_1[t],
                'ETC for EyesClosed': etc_2[t],
                })

    return pd.DataFrame(results)

def etc_func(signal):
    seq = ETC.partition(signal, n_bins=2)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')

    return res

if __name__ == '__main__':

    volunteers = [f"S{n:03d}" for n in range(1,110)]

    df = process_data(volunteers, root_dir)
    df.to_csv("etc_for_individual_times.csv", index=False)

    print(df)
