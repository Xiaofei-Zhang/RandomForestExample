# RF_importance.py
# This program classifies the exsome mass data using random forest.
# Input file is a csv file. By default, colunm 1 are the labels, and column 2
# are the IDs. Row 1 is the feature names (lipids formula).
#
# Usage:
#	python RF_importance.py [OPTIONS]
#
# Author: Xiaofei Zhang
# Date: April 18 2017

from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
import sys
from sklearn import cross_validation
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


label0 = 'Normal'
label1 = 'Early'
label2 = 'Late'


def read_data(filename):
    """This function reads data from the input CSV file
       The first column is the labels, the second column is the IDs.
       Usfule features starts from 3rd column
    Args:
    - filename: (string) full path of the input CSV
    Returns:
    - X: (2D numpy array) the features
    - y: (2D numpy array) labels
    """
    data = np.genfromtxt(filename, delimiter=',', dtype = str)
    X = data[1:,2:].astype(np.float)
    y = data[1:,0]
    y[y==label0]='0' 
    y[y==label1]='1' 
    y[y==label2]='2'
    y.astype(np.float) 
    return X, y

def pred(X, y, opts):
    """Prediction
    """
    X_raw, y_raw = X, y
    array1 = np.genfromtxt(opts.data, delimiter=',', dtype=str)
    print(array1[:,0])
    print("Data Shape:")
    print(X_raw.shape)
    header = array1[0, 2:]
    clf = RandomForestClassifier(n_estimators=opts.n_estimators, max_features=opts.max_features, max_depth=None, min_samples_split=opts.min_samples_split, random_state=0)
    clf = clf.fit(X, y)
    avg_array = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    total_feature = X.shape[1]*(-1)
    total_idx_imp_ranked = avg_array.argsort()[total_feature:][::-1]
    top_idx = total_idx_imp_ranked[0:opts.top_feature]
    #top_idx = avg_array.argsort()[top_feature:][::-1]
    
    # Save the top feature data to file (importance ranked)
    idx_out = top_idx+2
    print(idx_out)
    head = np.asarray([0,1])
    final_idx = np.append(head, idx_out)
    data_out = array1[:, final_idx]
    print(data_out[:,0])
    np.savetxt(opts.out, data_out, delimiter=',', fmt = '%s')

    # Save the importance data to file
    total_feature = X.shape[1]
    imp_out = np.hstack((avg_array.reshape(-1,1), std.reshape(-1,1)))
    idx_list = np.arange(1,total_feature+1).reshape((-1, 1))
    sorted_imp = imp_out[imp_out[:,0].argsort()[::-1]]
    final_out = np.hstack((idx_list,header[total_idx_imp_ranked].reshape((-1,1)),sorted_imp))
    np.savetxt('imp.csv', final_out, delimiter = ',', fmt = '%s')

    #model = SelectFromModel(clf, prefit=True, threshold=opts.th)
    #X_new = model.transform(X)
    X_new = X_raw[:,top_idx]
    print("Data shape after feature selection:")
    print(X_new.shape)
    print("Selected feature indices and names")
    print(top_idx)
    print(header[top_idx])
    k_fold = cross_validation.StratifiedKFold(y, n_folds=opts.n_folds, shuffle=True)
    final_cv_scores = []
    final_cv_scores2 = []
    importance_dict2 = {}
    importance_id2 = 0
    for train, test in k_fold:
        print("================")
        print("Top %i features" % opts.top_feature)
        clf.fit(X_new[train], y[train])
        print(X_new[train].shape, y[train].shape)
        myscore = clf.score(X_new[train], y[train])
        importance_dict2[importance_id2] = clf.feature_importances_
        validationscore = clf.score(X_new[test], y[test])
        final_cv_scores.append(validationscore)
        #print clf
        print(("Fitting Score: "+ str(myscore)))
        print(("Validation Score: "+ str(validationscore)))
        print("Classification report:\n", classification_report(y[test],clf.predict(X_new[test])))
        print("Confusion matrix:\n", confusion_matrix(y[test], clf.predict(X_new[test])))
        print("=================")
        print("All features")
        clf.fit(X_raw[train], y[train])
        print(X_raw[train].shape, y[train].shape)
        myscore = clf.score(X_raw[train], y[train])
        importance_dict2[importance_id2] = clf.feature_importances_
        validationscore = clf.score(X_raw[test], y[test])
        final_cv_scores2.append(validationscore)
        #print clf
        print(("Fitting Score: "+ str(myscore)))
        print(("Validation Score: "+ str(validationscore)))
        print("Classification report:\n", classification_report(y[test],clf.predict(X_raw[test])))
        print("Confusion matrix:\n", confusion_matrix(y[test], clf.predict(X_raw[test])))
    print("Average validation top%i score is: %.4f" % (opts.top_feature, np.mean(np.asarray(final_cv_scores))))
    print("Average validation all score is: %.4f" % np.mean(np.asarray(final_cv_scores2)))
def main(args):

    parser = argparse.ArgumentParser(description = "Learning")
    # Path of the data csv
    parser.add_argument("--data", dest="data", type=str, default="./Lung_cancer.csv")
    parser.add_argument("--out", dest="out", type=str, default="./top_feature.csv")
    parser.add_argument("--clf", dest="clf", type=int, default=0)
    # Data scaling
    parser.add_argument("--n_estimators", dest="n_estimators", type=int, default=1000)
    parser.add_argument("--n_folds", dest="n_folds", type=int, default=5)
    parser.add_argument("--top_feature", dest="top_feature", type=int, default=20)
    parser.add_argument("--min_samples_split", dest="min_samples_split", type=int, default=2)
    parser.add_argument("--max_features", dest="max_features", type=str, default='sqrt')
    opts = parser.parse_args(args[1:])
    X, y = read_data(opts.data)
    pred(X,y, opts)
    return 0

if __name__ == '__main__':
    main(sys.argv)



