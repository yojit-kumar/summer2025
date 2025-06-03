import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from codes import k_cross_validation
from codes import chaosnet
import ChaosFEX.feature_extractor as CFX



data_dir = os.path.dirname( os.getcwd() ) 

df = pd.read_csv(data_dir + '/etc_for_individual_channels_filtered.csv')

volunteers = df['volunteer'].unique()

X = []
Y = []

for v in volunteers:
    subdf = df[df['volunteer'] == v]

    etc_open = subdf['ETC for EyesOpen']
    etc_closed = subdf['ETC for EyesClosed']

    if len(etc_open) == 64 and len(etc_closed) == 64:
        X.append(etc_open)
        Y.append([0])  # Eyes open

        X.append(etc_closed)
        Y.append([1])  # Eyes closed
X = np.array(X)
Y = np.array(Y)




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train_norm = X_train
X_train_norm = (X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = X_test
X_test_norm = (X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)



print('Do you want to proceed with training?(y/n)')
choice = input('choice (default=n): ')

def training(choice='n'):
    if choice == 'y':        
        FOLD_NO=5
        INITIAL_NEURAL_ACTIVITY = np.arange(0.01, 0.99, 0.01)
        DISCRIMINATION_THRESHOLD = [0.499,0.969]
        EPSILON = np.arange(0.001,0.5000,0.001)
        k_cross_validation(FOLD_NO, X_train_norm, Y_train, X_test_norm, Y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)
    elif choice != 'n' and choice != '':
        print('Please choose a valid input')
        exit()

training(choice)

#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/CFX TUNING/RESULTS/' 
   
INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
DT = np.load(RESULT_PATH+"/h_B.npy")[0]

FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm, INA, 10000, EPSILON_1, DT)
FEATURE_MATRIX_VAL = CFX.transform(X_test_norm, INA, 10000, EPSILON_1, DT)            

mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_train, FEATURE_MATRIX_VAL)

f1 = f1_score(Y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
