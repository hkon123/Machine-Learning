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
from sklearn.ensemble import RandomForestClassifier


weather = pickle.load(open('data/mldata.p'))






trainIn, testIn, trainOut, testOut = train_test_split(weather.data, weather.target, test_size = 0.5, train_size = 0.5)

'''uncomment to use
#-----------------------------------over-sampling----------------------------------------
one=0
zero=0
two=0
for i in range(len(trainOut)):
    if trainOut[i] == '1':
        one+=1
    if trainOut[i]=='0':
        zero+=1
    if trainOut[i]=='2':
        two+=1

#print(one)
#print(zero)
#print(two)
#print()

add = 0
index = 0
while one>zero:
    if trainOut[index] == '0':
        trainIn=np.vstack((trainIn,trainIn[index]))
        trainOut = np.append(trainOut, ['0'])
        zero+=1
        add+=1
    index+=1
index = 0
add = 0
while one>two:
    if trainOut[index] == '2':
        trainIn=np.vstack((trainIn,trainIn[index]))
        trainOut = np.append(trainOut, ['2'])
        two+=1
        add+=1
    index+=1

#print(one)
#print(zero)
#print(two)

#---------------------------------------------------------------------------
'''

#--------------------------proscessing data-------------------------------
'''uncomment to use
#------normalize data------
scaler = Normalizer()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)
#---------------------------
'''

#------standardize data---------
scaler = StandardScaler()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)
#--------------------------------

#------------------------------------------------------------------------------


'''uncomment to use
#---------------------------------Grid-search-----------------------------------------
tuned_parameters = [{'solver':['adam'],\
 'activation':['tanh'],\
 'alpha':[1e-5],\
  'hidden_layer_sizes':[(5,2), (10,10), (10,10,10), (25,25), (25,25,25), (40,40,40), (60,60), (60,60,60)],\
   'random_state': [2],\
   'tol':[1e-10],\
   'max_iter':[500],\
   'warm_start':[True],\
   'epsilon':[1e-4] }]


scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(trainIn, trainOut)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = testOut, clf.predict(testIn)
    print(classification_report(y_true, y_pred))
    print()




#---------------------------------------------------------------------------
'''

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
                        #beta_1 = 0.9,\
                        #beta_2 = 0.999,\
                        #early_stopping = False,\
                        )


clf1 = KNeighborsClassifier(n_neighbors=50,\
                            weights = 'distance')

clf2 =RandomForestClassifier(max_depth=None,\
                            random_state=None,\
                            n_estimators = 200,\
                            min_samples_split = 3,\
                            #bootstrap = True,\
                            #oob_score = False,\
                            n_jobs = 5,\
                            verbose = 0,\
                            warm_start = False,\
                            class_weight = 'balanced_subsample')

#clf = MLPClassifier(solver='adam', alpha=0.000001, random_state=1)



print( trainOut)

clf = clf.fit(trainIn, trainOut)
clf1 = clf1.fit(trainIn, trainOut)
clf2 = clf2.fit(trainIn, trainOut)

thelist = clf.predict(testIn)
thelist1 = clf1.predict(testIn)
thelist2 = clf2.predict(testIn)

print("-------------Neural-----------")
print(classification_report(testOut, thelist, target_names = weather.getTargetNames()))

print(confusion_matrix(testOut, thelist))

print("----------------neighbors-------------------------------")
print(classification_report(testOut, thelist1, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, thelist1))
#print(classification_report(thelist, thelist1, target_names = weather.getTargetNames()))
#print(confusion_matrix(thelist, thelist1))

print("----------------random forest--------------------------------")
print(classification_report(testOut, thelist2, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, thelist2))


'''uncomment to use
#------------------result optimisation with two-----------------------------------
newlist = np.zeros(len(thelist), dtype = str)

for i in range(len(thelist)):
    if thelist2[i]==thelist1[i]:
        if thelist2[i]=='1':
            newlist[i] = '1'
        if thelist2[i]=='0':
            newlist[i] = '0'
        if thelist2[i]=='2':
            newlist[i] = '2'

    else:
        if thelist2[i] == '1':
            newlist[i] = thelist1[i]
        elif thelist1[i] == '1':
            newlist[i] = thelist2[i]
        else:
            if np.random.random()>0.5:
                newlist[i] = thelist2[i]
            else:
                newlist[i] = thelist1[i]

newlist.astype(str)
print("-------------------optimized-----------------")
print(classification_report(testOut, newlist, target_names = weather.getTargetNames()))
print(confusion_matrix(testOut, newlist))

#-------------------------------------
'''
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
