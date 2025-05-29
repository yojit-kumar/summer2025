import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split




df = pd.read_csv('etc_for_each_individual_channels.csv')

volunteer = sorted(df['volunteer'].unique())

X = []
Y = []

for v in volunteer:
    subdf = df[df['volunteer'] == v].sort_values(by='channel')

    # Extract 64-channel ETC vector for each task
    etc_open = subdf['ETC_Task1_EyesOpen'].values
    etc_closed = subdf['ETC_Task2_EyesClosed'].values

    if len(etc_open) == 64 and len(etc_closed) == 64:
        X.append(etc_open)
        Y.append(0)  # Eyes open

        X.append(etc_closed)
        Y.append(1)  # Eyes closed

print(X)
print(Y)

