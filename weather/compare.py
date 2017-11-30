import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

weather = pickle.load(open('data/mldata.p'))
weather1 = pickle.load(open('data/mldata1.p'))
weather2 = pickle.load(open('data/mldata2.p'))

trainIn, testIn, trainOut, testOut = train_test_split(weather.data, weather.target, test_size = 0.5, train_size = 0.5)
trainIn1, testIn1, trainOut1, testOut1 = train_test_split(weather1.data, weather1.target, test_size = 0.5, train_size = 0.5)
trainIn2, testIn2, trainOut2, testOut2 = train_test_split(weather2.data, weather2.target, test_size = 0.5, train_size = 0.5)



scaler = StandardScaler()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)

scaler = StandardScaler()
scaler.fit(trainIn1)
trainIn1 = scaler.transform(trainIn1)
#testIn = scaler.transform(testIn)

scaler = StandardScaler()
scaler.fit(trainIn2)
trainIn2 = scaler.transform(trainIn2)
#testIn = scaler.transform(testIn)

clf = MLPClassifier(solver='adam',\
                    activation = 'tanh',\
                    alpha=1e-10,\
                    hidden_layer_sizes=(40,40,40),\
                     max_iter = 500,\
                      random_state = 2,\
                      tol = 1e-10,\
                      verbose = False,\
                       warm_start = True,\
                        epsilon= 1e-6,\
                        beta_1 = 0.9,\
                        beta_2 = 0.999,\
                        early_stopping = False)


clf1 = MLPClassifier(solver='adam',\
                    activation = 'tanh',\
                    alpha=1e-10,\
                    hidden_layer_sizes=(40,40,40),\
                     max_iter = 500,\
                      random_state = 2,\
                      tol = 1e-10,\
                      verbose = False,\
                       warm_start = True,\
                        epsilon= 1e-6,\
                        beta_1 = 0.9,\
                        beta_2 = 0.999,\
                        early_stopping = False)

clf2 = MLPClassifier(solver='adam',\
                    activation = 'tanh',\
                    alpha=1e-10,\
                    hidden_layer_sizes=(40,40,40),\
                     max_iter = 500,\
                      random_state = 2,\
                      tol = 1e-10,\
                      verbose = False,\
                       warm_start = True,\
                        epsilon= 1e-6,\
                        beta_1 = 0.9,\
                        beta_2 = 0.999,\
                        early_stopping = False)


clf = clf.fit(trainIn, trainOut)
clf1 = clf1.fit(trainIn1, trainOut1)
clf2 = clf2.fit(trainIn2, trainOut2)

thelist = clf.predict(testIn)
thelist1 = clf1.predict(testIn1)
thelist2 = clf2.predict(testIn2)


print("-------------Neural1-----------")
print(classification_report(testOut, thelist, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, thelist))


print("-------------Neural2-----------")

print(classification_report(testOut, thelist1, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, thelist1))


print("-------------Neural3-----------")

print(classification_report(testOut, thelist2, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, thelist2))


#------------------result optimisation with three-----------------------------------
newlist = np.zeros(len(thelist), dtype = str)



for i in range(len(thelist)):
    if thelist[i]==thelist1[i] and thelist[i]==thelist2[i]:
        if thelist[i]=='1':
            newlist[i] = '1'
        if thelist[i]=='0':
            newlist[i] = '0'
        if thelist[i]=='2':
            newlist[i] = '2'
    else:
        if thelist[i]==thelist1[i]:
            newlist[i]=thelist[i]
        elif thelist[i] == thelist2[i]:
            newlist[i] = thelist[i]
        elif thelist1[i] == thelist2[i]:
            newlist[i] = thelist1[i]
        else:
            randomnr = np.random.random()
            if randomnr>0.66:
                newlist[i] = thelist[i]
            elif randomnr>0.33 and randomnr<=0.66:
                newlist[i] = thelist1[i]
            else:
                newlist[i] = thelist2[i]

newlist.astype(str)
print("-------------------optimized-----------------")
print(classification_report(testOut, newlist, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, newlist))


#-------------------------------------
