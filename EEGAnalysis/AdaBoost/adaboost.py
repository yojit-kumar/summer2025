import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold



data_dir = os.path.dirname(os.getcwd())

df = pd.read_csv(data_dir+'/etc_for_individual_channels_filtered.csv')

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


X_train_norm = (X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)



print('Do you want to proceed with training?(y/n)')
choice = input('choice (default=n): ')

def training(choice='n'):
    if choice == 'y':
        n_estimator = [1, 10, 50, 100, 500, 1000, 5000, 10000]
        BESTF1 = 0
        FOLD_NO = 5
        KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
        KF.get_n_splits(X_train_norm) 
        print(KF) 
        for NEST in n_estimator:
            
                        
            FSCORE_TEMP=[]

            for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
                
                X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
                Y_TRAIN, Y_VAL = Y_train[TRAIN_INDEX], Y_train[VAL_INDEX]
            
                
                clf = AdaBoostClassifier(n_estimators=NEST, random_state=42)
                clf.fit(X_TRAIN, Y_TRAIN.ravel())
                Y_PRED = clf.predict(X_VAL)
                f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                FSCORE_TEMP.append(f1)
                print('F1 Score', f1)
            print("Mean F1-Score for N-EST = ", NEST," is  = ",  np.mean(FSCORE_TEMP)  )
            if(np.mean(FSCORE_TEMP) > BESTF1):
                BESTF1 = np.mean(FSCORE_TEMP)
                BESTNEST = NEST
                
                
        print("BEST F1SCORE", BESTF1)
        print("BEST NEST = ", BESTNEST)




        print("Saving Hyperparameter Tuning Results")
           
          
        PATH = os.getcwd()
        RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'


        try:
            os.makedirs(RESULT_PATH)
        except OSError:
            print ("Creation of the result directory %s not required" % RESULT_PATH)
        else:
            print ("Successfully created the result directory %s" % RESULT_PATH)

        np.save(RESULT_PATH+"/h_NEST.npy", np.array([BESTNEST]) ) 
        np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) )

    elif choice != 'n' and choice != '':
        print('Please choose a valid input')
        exit()

training(choice=choice)

#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
    

NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = AdaBoostClassifier(n_estimators = NEST, random_state=42)
clf.fit(X_train_norm, Y_train.ravel())


Y_pred = clf.predict(X_test_norm)
f1 = f1_score(Y_test, Y_pred, average='macro')


print('TRAINING F1 Score', F1SCORE)
print('TESTING F1 Score', f1)


np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

