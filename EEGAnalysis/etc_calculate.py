import os
import numpy as np
import pandas as pd

#import pyedflib as edf
import ETC 

import mne

def compute_alpha(raw):
    raw.notch_filter(freqs=60, verbose=False)

    raw_alpha = raw.copy().filter(8.,12.,verbose=False)

    return raw_alpha


def process_data(volunteer_ids, channel_ids, root_dir='/home/user/eeg-motor-movementimagery-dataset-1.0.0/files/'):
    results = []

    for v in volunteer_ids:
        v_path = os.path.join(root_dir, v)


        task1 = os.path.join(v_path, f'{v}R01.edf')
        task2 = os.path.join(v_path, f'{v}R02.edf')

        #f1 = edf.EdfReader(task1)
        #f2 = edf.EdfReader(task2)

        r1 = mne.io.read_raw_edf(task1, preload=True, verbose=False)
        r2 = mne.io.read_raw_edf(task2, preload=True, verbose=False)

        #labels = f1.getSignalLabels()
        #label_idx = {label: idx for idx, label in enumerate(labels)}

        labels = r1.ch_names
        new_names = {}
        for ch in labels:
            clean_ch = ch.strip('.')
            if 'Fp' not in clean_ch:
                clean_ch = clean_ch.upper()
            if clean_ch.endswith('Z'):
                clean_ch = clean_ch[:-1] + 'z'
            
            new_names[ch] = clean_ch

        #d1 = {ch:ch.strip('.') for ch in labels}
        #d2 = {ch:ch.upper() if 'Fp' not in ch else ch for ch in list(d1.values())}
        #d3 = {ch:ch[:-1]+'z' if 'Z' in ch else ch for ch in list(d2.values())}
        #new_labels = list(d3.values())

        #new_names = {ch:new_labels[idx] for idx, ch in enumerate(labels)}
        
        r1.rename_channels(new_names)
        r2.rename_channels(new_names)

        r1.set_montage("standard_1020")
        r2.set_montage("standard_1020")

        labels = r1.ch_names
        label_idx = {label: idx for idx, label in enumerate(labels)}

        if r1.times[-1] > 60.0:
            r1 = r1.crop(tmin=0,tmax=60)
        if r2.times[-1] > 60.0:
            r2 = r2.crop(tmin=0,tmax=60)

        r1 = compute_alpha(r1)
        r2 = compute_alpha(r2)

        for ch in channel_ids:
            idx = label_idx[ch]
            
            #signal1 = f1.readSignal(idx)[:9600]
            #signal2 = f2.readSignal(idx)[:9600]

            signal1 = r1.copy().get_data()[idx,:]
            signal2 = r2.copy().get_data()[idx,:]

            etc_1 = etc_func(signal1)
            etc_2 = etc_func(signal2)

            results.append({
                'volunteer': v,
                'channel': ch,
                'ETC for EyesOpen': etc_1,
                'ETC for EyesClosed': etc_2,
                })

    return pd.DataFrame(results)

def etc_func(signal):
    seq = ETC.partition(signal, n_bins=2)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')

    return res

if __name__ == '__main__':

    volunteers = [f"S{n:03d}" for n in range(1,110)]


#All channels
    channels = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'] 


#Top Rank channels
#    channels = ['Afz.']


#Oscipital channels
#    channels = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

 
    df = process_data(volunteers, channels)
    df.to_csv("etc_for_individual_channels_filtered.csv", index=False)

    print(df)
