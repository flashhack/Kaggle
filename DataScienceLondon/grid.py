from __future__ import division
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import re
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rc('figure', figsize=(13, 7))
plt.style.use('ggplot')
def headtail(df, n = 5):
    print df.shape
    return df.head(n).append(df.tail(n))
import sys
print(sys.version)
print(pd.__version__)
print(np.__version__)

Y = pd.read_csv('trainLabels.csv', header=None)
Y = Y[0]
X = pd.read_csv('train.csv', header=None)

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc

from sklearn.grid_search import GridSearchCV
from operator import itemgetter

def report(grid_scores, n_top=10):
    sorted_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(sorted_scores):
        print "Model with rank: %d" % (i + 1)
        print "Mean validation score: %.3f (std: %.3f)" % (
              score.mean_validation_score,
              np.std(score.cv_validation_scores))
        print "Parameters: %s\n" % (score.parameters)

# parallel run: 43.5s
# single thread run: 2.3min
if __name__ == '__main__':        
    param_grid = {'estimator__C' : [2 **i for i in range(-5,17,2)], 'estimator__gamma' : [0] + [2 **i for i in xrange(-15,5,2)], 'estimator__kernel' : ['rbf']}
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=np.random.RandomState(0)))
    scorer = make_scorer(roc_auc_score, average='micro', greater_is_better=True, needs_threshold=True)
    grid_search = GridSearchCV(classifier, param_grid=param_grid, verbose=2, scoring=scorer,cv=5,n_jobs=-1).fit(X.values, Y.ravel())
    report(grid_search.grid_scores_)

#    param_grid = {'C' : [2 **i for i in range(-5,17,2)], 'gamma' : [0][2 **i for i in xrange(-15,5,2)], 'kernel' : ['rbf']}
#    svc = svm.SVC(kernel='rbf')
#    grid_search = GridSearchCV(svc, param_grid=param_grid, verbose=2,scoring='accuracy',cv=10,n_jobs=-1).fit(X.values, Y.ravel())
#    report(grid_search.grid_scores_)    
