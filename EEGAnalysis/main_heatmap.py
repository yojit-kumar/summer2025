import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns


df = pd.read_csv("EEGAnalysis_results2.csv")

df['delta'] = df['Difference EO - EC']

avg_delta = df.groupby('channel')['delta'].mean()


electrode_coords = {
    "Fp1.": (-2,  4), "Fpz.": ( 0,  4), "Fp2.": ( 2,  4),
    "Af7.": (-3,  3), "Af3.": (-2,  3), "Afz.": ( 0,  3), "Af4.": ( 2,  3), "Af8.": (3,   3),
    "F7..": (-4,  2), "F5..": (-3,  2), "F3..": (-2,  2), "F1..": (-1,  2), "Fz..": ( 0,  2), 
    "F2..": ( 1,  2), "F4..": ( 2,  2), "F6..": ( 3,  2), "F8..": ( 4,  2),
    "Ft7.": (-4,  1), "Fc5.": (-3,  1), "Fc3.": (-2,  1), "Fc1.": (-1,  1), "Fcz.": ( 0,  1), 
    "Fc2.": ( 1,  1), "Fc4.": ( 2,  1), "Fc6.": ( 3,  1), "Ft8.": ( 4,  1),
    "T9..": (-5,  0), "T7..": (-4,  0), "C5..": (-3,  0), "C3..": (-2,  0), "C1..": (-1,  0),"Cz..": ( 0,  0), 
    "C2..": ( 1,  0), "C4..": ( 2,  0), "C6..": ( 3,  0), "T8..": ( 4,  0), "T10..": (-5, 0),
    "Tp7.": (-4, -1), "Cp5.": (-3, -1), "Cp3.": (-2, -1), "Cp1.": (-1, -1),
    "Cpz.": ( 0, -1), "Cp2.": ( 1, -1), "Cp4.": ( 2, -1), "Cp6.": ( 3, -1), "Tp8.": ( 4, -1),
    "P7..": (-4, -2), "P5..": (-3, -2), "P3..": (-2, -2), "P1..": (-1, -2),
    "Pz..": ( 0, -2), "P2..": ( 1, -2), "P4..": ( 2, -2), "P6..": ( 3, -2), "P8..": ( 4, -2),
    "Po7.": (-3, -3), "Po3.": (-2, -3), "Poz.": ( 0, -3), "Po4.": ( 2, -3), "Po8.": ( 3, -3),
    "O1..": (-2, -4), "Oz..": ( 0, -4), "O2..": ( 2, -4),
    "Iz..": ( 0, -5)
}



heatmap = np.full((10,10), np.nan)

for ch, d in avg_delta.items():
    if ch in electrode_coords:
        x, y = electrode_coords[ch]
        heatmap[y,x] = d

plt.figure(figsize=(10,10))
plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Delta ETC EO - EC')
plt.title("Electrode Heat Map")

for ch, (x,y) in electrode_coords.items():
    if ch in avg_delta:
        plt.text(x,y,ch,ha='center',va='center',color='black')


plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("plot1.png")
plt.show()
