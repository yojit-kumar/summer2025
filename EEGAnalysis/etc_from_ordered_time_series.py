import os
import logging
from itertools import permutations

import numpy as np
import pandas as pd
import mne
import ETC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = '/home/user/eeg-motor-movementimagery-dataset-1.0.0/files/'
ALPHA_BAND = (8.0, 12.0)
NOTCH_FREQ = 60.0
TIME_WINDOW = 5 
DELAY = 2

# EEG Channel definitions
ALL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

OCCIPITAL_CHANNELS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']


def compute_alpha(raw):
    """Apply notch filter and extract alpha band."""
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw_alpha = raw.copy().filter(*ALPHA_BAND, verbose=False)
    return raw_alpha


def change_labels(raw1, raw2):
    """Standardize channel names and set montage."""
    labels = raw1.ch_names
    new_names = {}
    
    for ch in labels:
        clean_ch = ch.strip('.')
        if 'Fp' not in clean_ch:
            clean_ch = clean_ch.upper()
        if clean_ch.endswith('Z'):
            clean_ch = clean_ch[:-1] + 'z'
        new_names[ch] = clean_ch

    for raw in [raw1, raw2]:
        raw.rename_channels(new_names)
        raw.set_montage("standard_1020", verbose=False)


def permute_patterns(m):
    all_perms = list(permutations(range(m)))
    pattern_to_symbol = {perm: i for i, perm in enumerate(all_perms)}
    return pattern_to_symbol, len(all_perms)


def ordinal_patterns(pattern_to_symbol, ts, m):
    """Convert time series to ordinal pattern symbols."""
    ts = np.array(ts)
    n = len(ts)
       
    symbols = []
    i = 0
    while i < n - m:
        window = ts[i:i+m]
        ordinal_pattern = tuple(np.argsort(window))
        symbol = pattern_to_symbol[ordinal_pattern]
        symbols.append(symbol)
        i += DELAY 
    
    return symbols

def compute_etc(signal, bins=2):
    """Compute ETC for a symbolic sequence."""
    try:
        seq = ETC.partition(signal, n_bins=bins)
        result = ETC.compute_1D(seq, verbose=False)
        return result.get('NETC1D', 0.0)
    except Exception as e:
        logger.error(f"Error computing ETC: {e}")
        return 0.0


def load_and_preprocess_subject(volunteer_id, root_dir):
    """Load and preprocess EEG data for one subject."""
    try:
        # Load files
        volunteer_path = os.path.join(root_dir, volunteer_id)
        task1_file = os.path.join(volunteer_path, f'{volunteer_id}R01.edf')
        task2_file = os.path.join(volunteer_path, f'{volunteer_id}R02.edf')
        
        if not (os.path.exists(task1_file) and os.path.exists(task2_file)):
            logger.warning(f"Missing files for {volunteer_id}")
            return None, None, None
        
        raw1 = mne.io.read_raw_edf(task1_file, preload=True, verbose=False)
        raw2 = mne.io.read_raw_edf(task2_file, preload=True, verbose=False)
        
        # Standardize channel names
        change_labels(raw1, raw2)
        
        # Apply alpha band filtering
        raw1_alpha = compute_alpha(raw1)
        raw2_alpha = compute_alpha(raw2)
        
        # Crop if necessary
        for raw in [raw1_alpha, raw2_alpha]:
            if raw.times[-1] > 60:
                raw.crop(tmin=0, tmax=60)
        
        # Get data and channel names
        data1 = raw1_alpha.get_data()
        data2 = raw2_alpha.get_data() 
        channel_names = raw1_alpha.ch_names
        
        return data1, data2, channel_names
        
    except Exception as e:
        logger.error(f"Error processing {volunteer_id}: {e}")
        return None, None, None


def process_data(volunteer_ids, channel_ids, root_dir):
    """Process EEG data for all volunteers and channels."""
    results = []
    
    logger.info(f"Processing {len(volunteer_ids)} volunteers with {len(channel_ids)} channels each")
    
    for i, volunteer_id in enumerate(volunteer_ids, 1):
        logger.info(f"Processing {volunteer_id} ({i}/{len(volunteer_ids)})")
        
        # Load and preprocess data
        data1, data2, channel_names = load_and_preprocess_subject(volunteer_id, root_dir)
        
        if data1 is None:
            continue
        
        # Create channel index mapping
        channel_idx_map = {name: idx for idx, name in enumerate(channel_names)}
        
        # Process each requested channel
        for channel in channel_ids:
            if channel not in channel_idx_map:
                logger.warning(f"Channel {channel} not found for {volunteer_id}")
                continue
            
            try:
                idx = channel_idx_map[channel]
                signal1 = data1[idx, :]
                signal2 = data2[idx, :]


                pattern_to_symbol, n = permute_patterns(TIME_WINDOW)
                
                # Compute ordinal patterns
                symbols1 = ordinal_patterns(pattern_to_symbol, signal1, TIME_WINDOW)
                symbols2 = ordinal_patterns(pattern_to_symbol, signal2, TIME_WINDOW)
                
                # Compute ETC
                etc1 = compute_etc(symbols1, bins=n)
                etc2 = compute_etc(symbols2, bins=n)
                
                results.append({
                    'volunteer': volunteer_id,
                    'channel': channel,
                    'ETC_EyesOpen': etc1,
                    'ETC_EyesClosed': etc2,
                })
                
            except Exception as e:
                logger.error(f"Error processing channel {channel} for {volunteer_id}: {e}")
                continue
        
        # Progress update
        if i % 10 == 0:
            logger.info(f"Completed {i}/{len(volunteer_ids)} volunteers")
    
    logger.info(f"Processing complete. Total results: {len(results)}")
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    # Configuration
    volunteers = [f"S{n:03d}" for n in range(1, 110)]

    # Channel selection - choose one:
    channels = ALL_CHANNELS
    # channels = OCCIPITAL_CHANNELS  
    # channels = ['AFz']
    
    output_filename = f"etc_individual_channels_ordinal_patterns_timewindow{TIME_WINDOW}_delay{DELAY}.csv"
    
    try:
        # Process all data
        results_df = process_data(volunteers, channels, ROOT_DIR)
        
        # Save results
        results_df.to_csv(output_filename, index=False)
        logger.info(f"Results saved to {output_filename}")
        
        # Display summary
        print(f"\nProcessing Summary:")
        print(f"Total records: {len(results_df)}")
        print(f"Unique volunteers: {results_df['volunteer'].nunique()}")
        print(f"Unique channels: {results_df['channel'].nunique()}")
        print(f"\nFirst few results:")
        print(results_df.head())
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == '__main__':
    main()
