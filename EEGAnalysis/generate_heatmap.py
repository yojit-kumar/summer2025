import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns


df = pd.read_csv("etc_for_individual_channels.csv")

df['delta'] = df['Difference EO - EC']

avg_delta = df.groupby('channel')['delta'].mean()


electrode_coords = {
    "Fp1.": (3,9), "Fpz.": (5,9), "Fp2.": (7,9),
    "Af7.": (2,8), "Af3.": (3,8), "Afz.": (5,8), "Af4.": (7,8), "Af8.": (8,8),
    "F7..": (1,7), "F5..": (2,7), "F3..": (3,7), "F1..": (4,7), "Fz..": (5,7), 
    "F2..": (6,7), "F4..": (7,7), "F6..": (8,7), "F8..": (9,7),
    "Ft7.": (1,6), "Fc5.": (2,6), "Fc3.": (3,6), "Fc1.": (4,6), "Fcz.": (5,6), 
    "Fc2.": (6,6), "Fc4.": (7,6), "Fc6.": (8,6), "Ft8.": (9,6),
    "T9..": (0,5), "T7..": (1,5), "C5..": (2,5), "C3..": (3,5), "C1..": (4,5),"Cz..": (5,5), 
    "C2..": (6,5), "C4..": (7,5), "C6..": (8,5), "T8..": (9,5), "T10.": (10,5),
    "Tp7.": (1,4), "Cp5.": (2,4), "Cp3.": (3,4), "Cp1.": (4,4),
    "Cpz.": (5,4), "Cp2.": (6,4), "Cp4.": (7,4), "Cp6.": (8,4), "Tp8.": (9,4),
    "P7..": (1,3), "P5..": (2,3), "P3..": (3,3), "P1..": (4,3),
    "Pz..": (5,3), "P2..": (6,3), "P4..": (7,3), "P6..": (8,3), "P8..": (9,3),
    "Po7.": (2,2), "Po3.": (3,2), "Poz.": (5,2), "Po4.": (7,2), "Po8.": (8,2),
    "O1..": (3,1), "Oz..": (5,1), "O2..": (7,1),
    "Iz..": (5,0)
}



heatmap = np.full((11,11), np.nan)

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
plt.savefig("electrode_heatmap.png")
plt.show()
