import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.model_selection import GridSearchCV

## Step 1 - Import and Data loading

# Import the sklearn modules you intend to use as part of your Machine Learning analysis
# (e.g. classifiers, metrics, model selection)

import pickle
# ADD SKLEARN MODULES FOR YOUR CHOSEN CLASSIFICATION METHOD HERE
# e.g. to load the decision tree estimator use: from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the weather data you created by FeatureExtraction.py
weather = pickle.load(open('data/mldata.p'))

# Confirm that the data has loaded correctly by inspecting the data attributes in the `weather` object.

# ADD PRINT STATEMENTS HERE (SEE FeatureExtract.py FOR EXAMPLES)
print(weather.data[0])
print(weather.target[0])
## Step 2 - Define the training and testing sample
#
# Divide the weather data into a suitable training and testing sample.
# Start with a 50/50 split but make this easily adaptable for futher fitting evaluation.
#
# *Examples*:
trainIn, testIn, trainOut, testOut = train_test_split(weather.data, weather.target, test_size = 0.5, train_size = 0.5)  #`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)


#------------------------standardising features-------------------------------------

scaler = StandardScaler()
scaler.fit(trainIn)
trainIn = scaler.transform(trainIn)
testIn = scaler.transform(testIn)


#--------------------------------------------------------------------------------

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


'''uncomment to use
#-------------------------------Grid-search--------------------------------------------

tuned_parameters = [{'criterion':['gini','entropy'],\
                    'splitter':['best'],\
                    'min_samples_split': [2],\
                    'min_samples_leaf': [1],\
                    'random_state': [None],\
                    'class_weight': ['balanced'],\
                    'presort': [False],\
                    'max_depth':[ 9,10,11,12,25],\
                    'min_weight_fraction_leaf':[0],\
                    'max_leaf_nodes':[None]}]


scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5,
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



#-----------------------------------------------------------------------
'''





# DEFINE TRAINING AND TESTING SAMPLES AND TARGETS HERE

## Step 3 - Define the classification method

# This can be any of the estimators provided by Sklearn.
# I suggest you start with a *white box* method
# to better understand the process before trying something more advanced.

# DEFINE CLASSIFIER HERE
# e.g for a Decision tree: clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion = 'entropy',\
                            min_samples_split = 2,\
                            presort = False,\
                            max_depth = None,\
                            )


## Step 4 - Fit the training data

# Run the `fit` method of your chosen estimator using the training data (and corresponding targets) as input

# RUN FIT METHOD HERE
print(weather.getSelectedFeatures())
clf = clf.fit(trainIn,trainOut)

'''uncomment to use
#---------------Create in image of the decision tree-------------------

dot_data = export_graphviz(clf, out_file=None, feature_names=weather.getSelectedFeatures(), class_names=weather.getTargetNames(), filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("mini")

#------------------------------------------------------------------------
'''
## Step 5 - Define the expected and predicted datasets


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
print(clf.feature_importances_)
# Define `expected` as your *test* target values (i.e. **not** your *training* target values)
# and run the `predict` method on your chosen estimator using your *test data*

# DEFINE EXPECTED AND PREDICTED VALUES HERE

## Step 6 - Prediction Evaluation

# Use the `sklearn.metrics` module to compare the results using the expected and predicted datasets.

# Examples:
# - [Sklearn Model Evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html#)
# - [Handwritten Digits example](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)

# RUN PREDICTION EVALUATION METHODS HERE
