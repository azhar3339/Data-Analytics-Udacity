#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def print_statistics(data_dict):

    print "***********STATS START*******************"
    print "Total number of data points:", len(data_dict)
    print "Total number of features:", len(data_dict["TOTAL"])
    total_nans = {}
    no_of_pois = 0
    no_of_non_pois = 0
    for subject in data_dict:
        info = data_dict[subject]
        if info["poi"] == 1:
            no_of_pois += 1
        else:
            no_of_non_pois += 1
        for feature in info:
            if info[feature] == "NaN":
                if feature in total_nans:
                    total_nans[feature] += 1
                else:
                    total_nans[feature] = 1
    print "Total data points for POI:", no_of_pois
    print "Total data points for non POI:", no_of_non_pois
    print "Features with highest number of missing values"
    for feature in total_nans:
        if total_nans[feature] > 100:
            print feature, total_nans[feature]
    print "***********STATS END*******************"

def analyze_different_classifiers():

    classifier_names = [
    "Random Forests",
    "Gradient Boosting",
    "Decision Trees",
    "SVM",
    "Gaussian NB"]

    classifiers = [
        Pipeline([('scale', MinMaxScaler()),
                  ('kbest', SelectKBest()),
                  ('rf', RandomForestClassifier(random_state=42))]),
        Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest()),
                 ('gb', GradientBoostingClassifier(random_state=42))]),
        Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest()),
                 ('dt', DecisionTreeClassifier())]),
        Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest()),
                 ('svc', SVC())]),
        Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest()),
                 ('nb', GaussianNB())])
                  ]

    classifier_parms = [
        {"rf__n_estimators":range(10,110,10),
         "kbest__k":range(1,8)},
        {"gb__n_estimators":range(100,1100,100),
         "kbest__k":range(1,8)},
        {"dt__min_samples_split":range(10,110,10),
         "kbest__k":range(1,8)},
        {'svc__C': [1e3, 1e4, 1e5],
         'svc__gamma': [0.0001, 0.001, 0.01, 0.1],
         'svc__kernel':["rbf"],
         "kbest__k":range(1,8)},
        {"kbest__k":range(1,8)}
                        ]

    for name, classifier, parm in zip(classifier_names, classifiers, classifier_parms):
        print "Getting results for:", name
        clf = GridSearchCV(classifier, parm)
        clf = clf.fit(features, labels)
        test_classifier(clf.best_estimator_, my_dataset, features_list)

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'total_payments', ##Financial
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi'] ##Email

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##Print statistics
print_statistics(data_dict)



### Task 2: Remove outliers
####Check salary for extreme values
salary = []
count = 0
x_axis = []
for subject in data_dict:
    count += 1
    salary.append(data_dict[subject]["salary"])
    x_axis.append(count)
matplotlib.pyplot.scatter(x_axis,salary)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.title("Before Removing Total")
matplotlib.pyplot.show()

###Dropping the outlier
data_dict.pop("TOTAL", 0)

salary = []
count = 0
x_axis = []
for subject in data_dict:
    count += 1
    salary.append(data_dict[subject]["salary"])
    x_axis.append(count)
matplotlib.pyplot.scatter(x_axis,salary)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.title("After Removing Total")
matplotlib.pyplot.show()



### Task 3: Create new feature(s)

##Fraction of emails from and to poi instead of the raw count
for subject in data_dict:
    info = data_dict[subject]

    ##Check for NaNs and if found set the value to zero
    ##fraction_of_emails_from_poi
    if info["from_poi_to_this_person"] == "NaN" or info["to_messages"] == "NaN":
        info["fraction_of_emails_from_poi"] = 0
    else:
        info["fraction_of_emails_from_poi"] = info["from_poi_to_this_person"]/float(info["to_messages"])
    ##fraction_of_emails_to_poi
    if info["from_this_person_to_poi"] == "NaN" or info["from_messages"] == "NaN":
        info["fraction_of_emails_to_poi"] = 0
    else:
        info["fraction_of_emails_to_poi"] = info["from_this_person_to_poi"]/float(info["from_messages"])

###Adding the new features to features list
features_list.append("fraction_of_emails_from_poi")
features_list.append("fraction_of_emails_to_poi")

###Removing the features "from_this_person_to_poi", "from_messages", "from_poi_to_this_person", "to_messages"
###These are redundant now with the inclusion of new features
remove_these = ["from_this_person_to_poi", "from_messages",
                "from_poi_to_this_person", "to_messages"]

features_list = [x for x in features_list if x not in remove_these]
print features_list


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# analyze_different_classifiers()



###Extract the features for the best classifiers
clf = Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest(k=5)),
                 ('nb', GaussianNB())])
clf.fit(features, labels)

feature_support = clf.named_steps['kbest'].get_support()
scores = clf.named_steps['kbest'].scores_
print "features selected"
for feature, support, score in zip(features_list[1:], feature_support, scores):
    print feature, support, score

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
print "Found best classifier. Dumping the results"
#best classifier
clf = Pipeline([('scale', MinMaxScaler()),
                 ('kbest', SelectKBest(k=5)),
                 ('nb', GaussianNB())])
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
