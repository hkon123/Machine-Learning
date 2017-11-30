import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

weather = pickle.load(open('data/mldata.p'))
#print(int(weather.target[0])+1)





trainIn, testIn, trainOut, testOut = train_test_split(weather.data, weather.target, test_size = 0.5, train_size = 0.5)
scaler = StandardScaler()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)

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

index = 0
while one>zero:
    if trainOut[index] == '0':
        trainIn=np.vstack((trainIn,trainIn[index]))
        trainOut = np.append(trainOut, ['0'])
        zero+=1
    index+=1
index = 0
while one>two:
    if trainOut[index] == '2':
        trainIn=np.vstack((trainIn,trainIn[index]))
        trainOut = np.append(trainOut, ['2'])
        two+=1
    index+=1

#print(one)
#print(zero)
#print(two)

#---------------------------------------------------------------------------
'''
scaler = StandardScaler()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)
'''
'''
#---------------------------------cross-validation-----------------------------------------
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

#clf = LinearSVC()#loss = 'squared_hinge',\
                #dual = False,\
                #multi_class = 'crammer_singer')

#clf = KNeighborsClassifier(n_neighbors=4,\
#                            weights = 'distance')

clf = RandomForestClassifier(max_depth=None,\
                            random_state=None,\
                            n_estimators = 200,\
                            min_samples_split = 3,\
                            #bootstrap = True,\
                            #oob_score = False,\
                            n_jobs = 5,\
                            verbose = 0,\
                            warm_start = False,\
                            #class_weight = 'balanced',\
                            )

#clf = MLPClassifier(solver='adam', alpha=0.000001, random_state=1)

'''
for i in range(0,len(trainOut)):
    trainOut[i] = int(trainOut[i])
print(len(testOut))
for i in range(0,len(testOut)):
    testOut[i] = int(testOut[i])


print(trainOut[0])
trainOut[0]= int(trainOut[0])+1
print(trainOut[0])
'''

print( trainOut)

clf = clf.fit(trainIn, trainOut)


thelist = clf.predict(testIn)

true = 0
false = 0
ones=0
zeros = 0
twos = 0
for i in range(0, len(thelist)):
    if thelist[i] == testOut[i]:
        true += 1
    else:
        false += 1
    if thelist[i]=='1':
        ones+=1
    if thelist[i]=='0':
        zeros+=1
    if thelist[i]=='2':
        twos+=1



print(clf.score(testIn,testOut))
print("number of true values: " + str(true))
print("number of false values: " + str(false))
print(classification_report(testOut, thelist, target_names = weather.getTargetNames()))
print(thelist)
print(testOut)
print("0's " + str(zeros))
print("1's " + str(ones))
print("2's " + str(twos))
print(confusion_matrix(testOut, thelist))
