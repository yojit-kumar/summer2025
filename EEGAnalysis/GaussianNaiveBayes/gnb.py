import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



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




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train_norm = (X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)


#Algorithm - Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train_norm, Y_train.ravel())

PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'

Y_pred = clf.predict(X_test_norm)
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')
prec = precision_score(Y_test, Y_pred, average='macro')
recall = recall_score(Y_test, Y_pred, average='macro')

print('ACCURACY', acc)
print('TESTING F1 Score', f1)
print('PRECISION', prec)
print('RECALL', recall)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )
