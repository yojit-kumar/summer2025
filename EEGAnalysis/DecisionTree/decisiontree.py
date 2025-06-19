import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold




FILE = '/etc_for_individual_channels_filtered.csv'
SEQ_LENGTH = 64



data_dir = os.path.dirname(os.getcwd())

df = pd.read_csv(data_dir+FILE)

volunteers = df['volunteer'].unique()

X = []
Y = []

for v in volunteers:
    subdf = df[df['volunteer'] == v]

    etc_open = subdf['ETC for EyesOpen']
    etc_closed = subdf['ETC for EyesClosed']

    if len(etc_open) == SEQ_LENGTH and len(etc_closed) == SEQ_LENGTH:
        X.append(etc_open)
        Y.append([0])  # Eyes open

        X.append(etc_closed)
        Y.append([1])  # Eyes closed
X = np.array(X)
Y = np.array(Y)
print(len(X))
print(len(X[0]))


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train_norm = (X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)



print('Do you want to proceed with training?(y/n)')
choice = input('choice (default=n): ')

def training(choice='n'):
    if choice == 'y':
        clf = DecisionTreeClassifier(random_state=42)
        ccp_path = clf.cost_complexity_pruning_path(X_train, Y_train)
        alpha = ccp_path['ccp_alphas']
        BESTF1 = 0
        FOLD_NO = 5
        KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
        KF.get_n_splits(X_train_norm) 
        print(KF) 
        for MSL in range(1,11):
                        
            for MD in range(1,11):
                
                for CCP in alpha:

                    FSCORE_TEMP=[]
                
                    for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
                        
                        X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
                        Y_TRAIN, Y_VAL = Y_train[TRAIN_INDEX], Y_train[VAL_INDEX]
                    
                        
                        clf = DecisionTreeClassifier(min_samples_leaf = MSL, random_state=42, max_depth = MD, ccp_alpha = CCP)
                        clf.fit(X_TRAIN, Y_TRAIN.ravel())
                        Y_PRED = clf.predict(X_VAL)
                        f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                        FSCORE_TEMP.append(f1)
                        print('F1 Score', f1)
                    print("Mean F1-Score for MSL = ", MSL," MD = ", MD," CCP = ", CCP," is  = ",  np.mean(FSCORE_TEMP)  )
                    if(np.mean(FSCORE_TEMP) > BESTF1):
                        BESTF1 = np.mean(FSCORE_TEMP)
                        BESTMSL = MSL
                        BESTMD = MD
                        BESTCCP = CCP
                
        print("BEST F1SCORE", BESTF1)
        print("BEST MD = ", BESTMD)
        print("BEST MSL = ", BESTMSL)
        print("BEST CCP = ", BESTCCP)


        print("Saving Hyperparameter Tuning Results")
           
          
        PATH = os.getcwd()
        RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'


        try:
            os.makedirs(RESULT_PATH)
        except OSError:
            print ("Creation of the result directory %s not required" % RESULT_PATH)
        else:
            print ("Successfully created the result directory %s" % RESULT_PATH)

        np.save(RESULT_PATH+"/h_MSL.npy", np.array([BESTMSL]) ) 
        np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]) ) 
        np.save(RESULT_PATH+"/h_CCP.npy", np.array([BESTCCP]) ) 
        np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) )

    elif choice != 'n' and choice != '':
        print('Please choose a valid input')
        exit()

training(choice)

#Testing

PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
    

MSL = np.load(RESULT_PATH+"/h_MSL.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
CCP = np.load(RESULT_PATH+"/h_CCP.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = DecisionTreeClassifier(min_samples_leaf = MSL, random_state = 42, max_depth = MD, ccp_alpha = CCP)
clf.fit(X_train_norm, Y_train.ravel())

Y_pred = clf.predict(X_test_norm)
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')
prec = precision_score(Y_test, Y_pred, average='macro')
recall = recall_score(Y_test, Y_pred, average='macro')


print('TRAINING F1 Score', F1SCORE)

print('ACCURACY', acc)
print('TESTING F1 Score', f1)
print('PRECISION', prec)
print('RECALL', recall)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )
