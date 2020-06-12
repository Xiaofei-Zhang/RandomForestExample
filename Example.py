from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score
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
    data = np.genfromtxt(filename, delimiter=',', dtype=str)
    X = data[1:,2:].astype(np.float)
    y = data[1:,0]
    y[y==label0]='0'
    y[y==label1]='1'
    y[y==label2]='2'
    y=y.astype(np.float)
    return X, y


def pred(X, y, opts):
    """Prediction
    """
    X_raw, y_raw = X, y
    clf = RandomForestClassifier(n_estimators=opts.n_estimators, max_features=opts.max_features, max_depth=None, min_samples_split=opts.min_samples_split, random_state=0)
    c0_idx = []
    c1_idx = []
    c2_idx = []
    for i in range(y.shape[0]):
        if y[i] == 0:
            c0_idx.append(i)
        elif y[i] == 1:
            c1_idx.append(i)
        else:
            c2_idx.append(i)
    X_c0_c1 = X[np.asarray(c0_idx+c1_idx),:]
    y_c0_c1 = y[np.asarray(c0_idx+c1_idx)]
    X_c0_c2 = X[np.asarray(c0_idx+c2_idx),:]
    y_c0_c2 = y[np.asarray(c0_idx+c2_idx)]
    X_c1_c2 = X[np.asarray(c1_idx+c2_idx),:]
    y_c1_c2 = y[np.asarray(c1_idx+c2_idx)]
    bootstrapped_scores0=[]
    bootstrapped_scores1=[]
    bootstrapped_scores2=[]
    bootstrapped_specificity0=[]
    bootstrapped_specificity1=[]
    bootstrapped_specificity2=[]
    bootstrapped_sensitivity0=[]
    bootstrapped_sensitivity1=[]
    bootstrapped_sensitivity2=[]
    print("----------------------------------------------")
    print('ROC_c0 ROC_c1 ROC_c2 pc0 pc1 pc2 rc0 rc1 rc2')
    print("----------------------------------------------")
    for i in range(opts.n_time):
        clf = RandomForestClassifier(n_estimators=opts.n_estimators, max_features=opts.max_features, max_depth=None, min_samples_split=opts.min_samples_split, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X_c0_c1, label_binarize(y_c0_c1, classes=[0,1])[:,0], test_size=1.0/opts.n_folds, random_state=i, stratify=label_binarize(y_c0_c1, classes=[0,1])[:,0])
        clf.fit(X_train, y_train)
        total_feature = X_train.shape[1]*(-1)
        avg_array = clf.feature_importances_
        total_idx_imp_ranked = avg_array.argsort()[total_feature:][::-1]
        top_idx = total_idx_imp_ranked[0:opts.top_feature]
        X_train = X_train[:,top_idx]
        X_test = X_test[:,top_idx]
        c0_score = clf.fit(X_train, y_train).predict_proba(X_test)
        roc_auc0=roc_auc_score(y_test, c0_score[:,1])
        rc0 = recall_score(y_test, c0_score[:,1].round(), pos_label=1)
        cm = confusion_matrix(y_test, c0_score[:,1].round())
        TN = float(cm[0][0])
        FP = float(cm[0][1])
        FN = float(cm[1][0])
        TP = float(cm[1][1])
        pc0 = TN/(TN+FP)
        clf = RandomForestClassifier(n_estimators=opts.n_estimators, max_features=opts.max_features, max_depth=None, min_samples_split=opts.min_samples_split, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X_c0_c2, label_binarize(y_c0_c2, classes=[0,2])[:,0], test_size=1.0/opts.n_folds, random_state=i, stratify=label_binarize(y_c0_c2, classes=[0,2])[:,0])
        clf.fit(X_train, y_train)
        total_feature = X_train.shape[1]*(-1)
        avg_array = clf.feature_importances_
        total_idx_imp_ranked = avg_array.argsort()[total_feature:][::-1]
        top_idx = total_idx_imp_ranked[0:opts.top_feature]
        X_train = X_train[:,top_idx]
        X_test = X_test[:,top_idx]
        c1_score = clf.fit(X_train, y_train).predict_proba(X_test)
        roc_auc1=roc_auc_score(y_test, c1_score[:,1])
        rc1 = recall_score(y_test, c1_score[:,1].round(), pos_label=1)
        cm = confusion_matrix(y_test, c1_score[:,1].round())
        TN = float(cm[0][0])
        FP = float(cm[0][1])
        FN = float(cm[1][0])
        TP = float(cm[1][1])
        pc1 = TN/(TN+FP)

        clf = RandomForestClassifier(n_estimators=opts.n_estimators, max_features=opts.max_features, max_depth=None, min_samples_split=opts.min_samples_split, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X_c1_c2, label_binarize(y_c1_c2, classes=[1,2])[:,0], test_size=1.0/opts.n_folds, random_state=i, stratify=label_binarize(y_c1_c2, classes=[1,2])[:,0])
        clf.fit(X_train, y_train)
        total_feature = X_train.shape[1]*(-1)
        avg_array = clf.feature_importances_
        total_idx_imp_ranked = avg_array.argsort()[total_feature:][::-1]
        top_idx = total_idx_imp_ranked[0:opts.top_feature]
        X_train = X_train[:,top_idx]
        X_test = X_test[:,top_idx]
        c2_score = clf.fit(X_train, y_train).predict_proba(X_test)
        roc_auc2=roc_auc_score(y_test, c2_score[:,1])
#        pc2 = precision_score(y_test, c1_score[:,1].round(), pos_label=1)
        rc2 = recall_score(y_test, c2_score[:,1].round(), pos_label=1)
        cm = confusion_matrix(y_test, c2_score[:,1].round())
        TN = float(cm[0][0])
        FP = float(cm[0][1])
        FN = float(cm[1][0])
        TP = float(cm[1][1])
        pc2 = TN/(TN+FP)
        
        print(roc_auc0, roc_auc1, roc_auc2, pc0, pc1, pc2, rc0, rc1, rc2)
        bootstrapped_scores0.append(roc_auc0)
        bootstrapped_scores1.append(roc_auc1)
        bootstrapped_scores2.append(roc_auc2)
        bootstrapped_specificity0.append(pc0)
        bootstrapped_specificity1.append(pc1)
        bootstrapped_specificity2.append(pc2)
        bootstrapped_sensitivity0.append(rc0)
        bootstrapped_sensitivity1.append(rc1)
        bootstrapped_sensitivity2.append(rc2)
    sorted_scores = np.array(bootstrapped_scores0)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("---------------------------------------------")
    print("auROC,std,lower_CI,upper_CI:")
    print("%s_vs_%s:"%(label0,label1))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_scores0), np.std(bootstrapped_scores0), confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_scores1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label0, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_scores1), np.std(bootstrapped_scores1), confidence_lower, confidence_upper))
    #print("Average AUROC for Normal VS PDAC: {:.4f} std: {:.4f} ".format(np.mean(bootstrapped_scores1), np.std(bootstrapped_scores1)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_scores2)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label1, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_scores2), np.std(bootstrapped_scores2), confidence_lower, confidence_upper))
    #print("Average AUROC for IPMN VS PDAC: {:.4f} std: {:.4f}".format(np.mean(bootstrapped_scores2), np.std(bootstrapped_scores2)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    print("---------------------------------------------")
    print("Specificity,std,lower_CI,upper_CI:")
    sorted_scores = np.array(bootstrapped_specificity0)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label0, label1))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_specificity0), np.std(bootstrapped_specificity0), confidence_lower, confidence_upper))
    #print("Average specificity for Normal VS IPMN: {:.4f} std: {:.4f}".format(np.mean(bootstrapped_specificity0), np.std(bootstrapped_specificity0)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_specificity1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label0, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_specificity1), np.std(bootstrapped_specificity1), confidence_lower, confidence_upper))
    #print("Average specificity for Normal VS PDAC: {:.4f} std: {:.4f} ".format(np.mean(bootstrapped_specificity1), np.std(bootstrapped_specificity1)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_specificity2)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label1, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_specificity2), np.std(bootstrapped_specificity2), confidence_lower, confidence_upper))
    #print("Average specificity for IPMN VS PDAC : {:.4f} std: {:.4f}".format(np.mean(bootstrapped_specificity2), np.std(bootstrapped_specificity2)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    print("---------------------------------------------")
    print("Sensitivity,std,lower_CI,upper_CI:")
    sorted_scores = np.array(bootstrapped_sensitivity0)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label0, label1))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_sensitivity0), np.std(bootstrapped_sensitivity0), confidence_lower, confidence_upper))
    #print("Average sensitivity for Normal VS IPMN: {:.4f} std: {:.4f}".format(np.mean(bootstrapped_sensitivity0), np.std(bootstrapped_sensitivity0)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_sensitivity1)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label0, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_sensitivity1), np.std(bootstrapped_sensitivity1), confidence_lower, confidence_upper))
    #print("Average sensitivity for Normal VS PDAC: {:.4f} std: {:.4f} ".format(np.mean(bootstrapped_sensitivity1), np.std(bootstrapped_sensitivity1)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    sorted_scores = np.array(bootstrapped_sensitivity2)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("%s_vs_%s:" % (label1, label2))
    print("%.4f %.4f %.4f %.4f" % (np.mean(bootstrapped_sensitivity2), np.std(bootstrapped_sensitivity2), confidence_lower, confidence_upper))
    #print("Average sensitivity for IPMN VS PDAC : {:.4f} std: {:.4f}".format(np.mean(bootstrapped_sensitivity2), np.std(bootstrapped_sensitivity2)))
    #print("Confidence interval: [{:0.4f} - {:0.4f}]".format(confidence_lower, confidence_upper))
    
def main(args):
    """Main function to calculate the average 1V1 auROC
    INPUTS:
    - args: (list of strings) command line arguments
    """
    # Setting up reading of command line options, storing defaults if not provided.
    parser = argparse.ArgumentParser(description = "Calculating 1V1 auROCs:")
    # Path of the data csv
    parser.add_argument("--data", dest="data", type=str, default="./top_feature.csv")
    parser.add_argument("--n_estimators", dest="n_estimators", type=int, default=500)
    parser.add_argument("--n_folds", dest="n_folds", type=float, default=5)
    parser.add_argument("--min_samples_split", dest="min_samples_split", type=int, default=2)
    parser.add_argument("--max_features", dest="max_features", type=str, default='sqrt')
    parser.add_argument("--top_feature", dest="top_feature", type=int, default=20)

    parser.add_argument("--n_time", dest="n_time", type=int, default=100)

    opts = parser.parse_args(args[1:])
    X, y = read_data(opts.data)
    pred(X,y, opts)
    return 0


if __name__ == '__main__':
    main(sys.argv)
